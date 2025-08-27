# app_exec_views_cloud.py
# Full, cloud-ready Streamlit app for healthcare exec Q->Chart
# - Reads SUPABASE_DB_URL and OPENAI_API_KEY from Streamlit Secrets or env
# - Generates SQL with OpenAI; gracefully falls back to rules if the API is unavailable
# - Validates SQL (single SELECT, allowed views only), auto-adds LIMIT, and charts results

import os
import re
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
import plotly.express as px
import sqlglot
from sqlglot import exp  # used for robust SELECT check

# ---------- CONFIG ----------
# Read secrets (Streamlit Cloud) or environment variables (local)
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL") or st.secrets.get("SUPABASE_DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

# Whitelisted views the model is allowed to query
ALLOWED_VIEWS = {
    "v_provider_counts_by_county",
    "v_provider_counts_by_zip",
    "v_provider_counts_by_category",
    "v_facilities_by_county",
    "v_facilities_by_zip",
    "v_beds_by_county",
    "v_beds_by_zip",
    "v_physicians_by_county",
    "v_physicians_by_zip",
    "v_pas_by_county",
    "v_pas_by_zip",
    "v_rns_by_county",
    "v_rns_by_zip",
}

# Minimal schema the model sees (kept small & clean)
SCHEMA_DESC = """
You can ONLY query the following views (read-only). Column names are shown in parentheses.

v_provider_counts_by_county(state, county_fips, providers)
v_provider_counts_by_zip(state, zip, providers)
v_provider_counts_by_category(category, providers)

v_facilities_by_county(state, county_fips, facilities)
v_facilities_by_zip(state, zip, facilities)

v_beds_by_county(state, county_fips, beds, certified_beds)
v_beds_by_zip(state, zip, beds, certified_beds)

v_physicians_by_county(state, county_fips, physicians)
v_physicians_by_zip(state, zip, physicians)

v_pas_by_county(state, county_fips, physician_assistants)
v_pas_by_zip(state, zip, physician_assistants)

v_rns_by_county(state, county_fips, rns)
v_rns_by_zip(state, zip, rns)

Rules:
- Only SELECT queries.
- Never use tables directly, only these views.
- When filtering by state, use the two-letter code (e.g., 'TX').
- For county filtering you must use county_fips (numeric code).
- For zip filtering you must use 5-digit zip or a prefix with LIKE '77%'.
- Always include an ORDER BY when returning ranked results.
- Always add LIMIT 200 to the final query.
"""

# Few-shot examples to steer the model
FEW_SHOT = [
    {
        "q": "Top 10 Texas counties by physician count",
        "sql": "SELECT county_fips, physicians FROM v_physicians_by_county WHERE state='TX' ORDER BY physicians DESC LIMIT 10;",
    },
    {
        "q": "Provider counts by zip in Texas",
        "sql": "SELECT zip, providers FROM v_provider_counts_by_zip WHERE state='TX' ORDER BY providers DESC LIMIT 200;",
    },
    {
        "q": "Total certified beds by county in TX (show top 15)",
        "sql": "SELECT county_fips, certified_beds FROM v_beds_by_county WHERE state='TX' ORDER BY certified_beds DESC LIMIT 15;",
    },
    {
        "q": "Facilities by zip starting with 770 in Texas",
        "sql": "SELECT zip, facilities FROM v_facilities_by_zip WHERE state='TX' AND zip LIKE '770%' ORDER BY facilities DESC LIMIT 200;",
    },
    {
        "q": "Provider counts by provider category",
        "sql": "SELECT category, providers FROM v_provider_counts_by_category ORDER BY providers DESC LIMIT 200;",
    },
]

# ---------- OPENAI CLIENT & FALLBACK ----------
from openai import OpenAI
_client = None
if OPENAI_API_KEY:
    try:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        _client = None  # We'll still allow fallback

def _fallback_sql(user_question: str) -> str:
    """Very simple rules to keep the app usable if OpenAI is unavailable or rate-limited."""
    q = (user_question or "").lower()
    limit = "LIMIT 200;"

    # Beds
    if "bed" in q:
        if "zip" in q:
            return "SELECT zip, beds, certified_beds FROM v_beds_by_zip WHERE state='TX' ORDER BY certified_beds DESC " + limit
        return "SELECT county_fips, beds, certified_beds FROM v_beds_by_county WHERE state='TX' ORDER BY certified_beds DESC " + limit

    # Physicians
    if "physician" in q or "md" in q:
        if "zip" in q:
            return "SELECT zip, physicians FROM v_physicians_by_zip WHERE state='TX' ORDER BY physicians DESC " + limit
        return "SELECT county_fips, physicians FROM v_physicians_by_county WHERE state='TX' ORDER BY physicians DESC " + limit

    # PAs
    if ("physician assistant" in q) or (" pa " in f" {q} "):
        if "zip" in q:
            return "SELECT zip, physician_assistants FROM v_pas_by_zip WHERE state='TX' ORDER BY physician_assistants DESC " + limit
        return "SELECT county_fips, physician_assistants FROM v_pas_by_county WHERE state='TX' ORDER BY physician_assistants DESC " + limit

    # RNs
    if "rn" in q or "registered nurse" in q:
        if "zip" in q:
            return "SELECT zip, rns FROM v_rns_by_zip WHERE state='TX' ORDER BY rns DESC " + limit
        return "SELECT county_fips, rns FROM v_rns_by_county WHERE state='TX' ORDER BY rns DESC " + limit

    # Facilities
    if "facility" in q or "facilities" in q:
        if "zip" in q:
            return "SELECT zip, facilities FROM v_facilities_by_zip WHERE state='TX' ORDER BY facilities DESC " + limit
        return "SELECT county_fips, facilities FROM v_facilities_by_county WHERE state='TX' ORDER BY facilities DESC " + limit

    # Categories
    if "category" in q:
        return "SELECT category, providers FROM v_provider_counts_by_category ORDER BY providers DESC " + limit

    # Providers (default)
    if "zip" in q:
        return "SELECT zip, providers FROM v_provider_counts_by_zip WHERE state='TX' ORDER BY providers DESC " + limit
    return "SELECT county_fips, providers FROM v_provider_counts_by_county WHERE state='TX' ORDER BY providers DESC " + limit

def generate_sql(user_question: str) -> str:
    """Try OpenAI; if rate-limited/unavailable, use rules."""
    # If no client (missing key), skip straight to fallback
    if _client is None:
        st.warning("OpenAI key missing or invalid. Using a built-in fallback translator.")
        return _fallback_sql(user_question)

    system = (
        "You translate user questions into a single safe SQL SELECT statement. "
        "Only use the views listed. Do not invent columns. "
        "Do not use INSERT/UPDATE/DELETE/ALTER/DROP. Always add LIMIT 200."
    )
    examples = "\n\n".join([f"Q: {e['q']}\nSQL: {e['sql']}" for e in FEW_SHOT])
    prompt = f"""{SCHEMA_DESC}

{examples}

User question: {user_question}

Return ONLY the SQL query, nothing else.
"""
    try:
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        sql = resp.choices[0].message.content.strip()
        if sql.startswith("```"):
            sql = sql.strip("`")
            if sql.lower().startswith("sql"):
                sql = sql[3:]
        return sql.strip()
    except Exception:
        st.warning("LLM rate-limited or unavailable. Using a built-in fallback translator for now.")
        return _fallback_sql(user_question)

# ---------- VALIDATION ----------
FORBIDDEN = re.compile(r"\b(INSERT|UPDATE|DELETE|ALTER|DROP|TRUNCATE|GRANT|REVOKE|CALL|CREATE)\b", re.I)

def is_sql_safe(sql: str) -> tuple[bool, str]:
    if FORBIDDEN.search(sql):
        return False, "Query contains forbidden keywords."
    # Disallow stacked statements (anything beyond a single trailing ;)
    if ";" in sql.strip().rstrip(";"):
        return False, "Multiple statements are not allowed."
    try:
        # Strip trailing semicolon so the parser doesn't mis-read
        to_parse = sql.strip().rstrip(";")
        parsed = sqlglot.parse_one(to_parse, dialect="postgres")
    except Exception as e:
        return False, f"SQL parse error: {e}"
    # Robust SELECT check
    if not isinstance(parsed, exp.Select):
        return False, "Only SELECT statements are allowed."
    # Enforce view allowlist
    names = {t.name for t in parsed.find_all(sqlglot.exp.Table)}
    if not names.issubset(ALLOWED_VIEWS):
        return False, f"Query references non-allowed views: {names - ALLOWED_VIEWS}"
    return True, ""

def ensure_limit(sql: str, default_limit: int = 200) -> str:
    try:
        expr = sqlglot.parse_one(sql.strip().rstrip(";"), dialect="postgres")
        # Add LIMIT if missing
        has_limit = any(isinstance(node, sqlglot.exp.Limit) for node in expr.find_all(sqlglot.exp.Limit))
        if not has_limit:
            expr = expr.limit(default_limit)
        return expr.sql(dialect="postgres")
    except Exception:
        # Best-effort fallback
        if "limit" not in sql.lower():
            return sql.rstrip().rstrip(";") + f" LIMIT {default_limit};"
        return sql

# ---------- DB ----------
@st.cache_resource
def get_engine():
    if not SUPABASE_DB_URL:
        raise RuntimeError("Missing SUPABASE_DB_URL (set in Streamlit Secrets).")
    return create_engine(SUPABASE_DB_URL, pool_pre_ping=True)

def run_sql(sql: str) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)

# ---------- UI ----------
st.set_page_config(page_title="Healthcare Exec Q→Chart", layout="wide")
st.title("Healthcare Exec: Ask a Question → Get a Chart")

with st.sidebar:
    st.markdown("**Allowed views:**")
    for v in sorted(ALLOWED_VIEWS):
        st.code(v, language="sql")
    st.markdown("**Tips:**")
    st.write("- Use state codes like 'TX'")
    st.write("- For counties, filter by FIPS (numeric)")
    st.write("- For zips, use full 5-digit or prefix with LIKE '77%'")

# Quick-prompt buttons
st.caption("Quick prompts:")
col1, col2, col3 = st.columns(3)
qp = None
if col1.button("Top 10 TX counties by physicians"):
    qp = "Top 10 Texas counties by physician count"
if col2.button("Certified beds by county in TX"):
    qp = "Total certified beds by county in TX"
if col3.button("Providers by ZIP starting 770"):
    qp = "Provider counts by zip starting with 770 in Texas"

question = st.text_input(
    "Type your question (e.g., 'Top 10 Texas counties by physician count')",
    value=qp or "",
    placeholder="Top 10 Texas counties by physician count",
)
go = st.button("Run")

if go and question:
    with st.spinner("Thinking…"):
        raw_sql = generate_sql(question)
        st.subheader("Generated SQL")
        st.code(raw_sql, language="sql")

        safe_sql = ensure_limit(raw_sql)
        ok, reason = is_sql_safe(safe_sql)
        if not ok:
            st.error(f"Rejected SQL: {reason}")
        else:
            try:
                df = run_sql(safe_sql)
                if df.empty:
                    st.warning("No rows returned. Try a broader question.")
                else:
                    st.subheader("Results")
                    st.dataframe(df)

                    # Simple auto-chart heuristic
                    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
                    if numeric_cols and non_numeric_cols:
                        y = numeric_cols[0]
                        x = non_numeric_cols[0]
                        if any(k in x.lower() for k in ["year", "date", "month"]):
                            fig = px.line(df, x=x, y=y)
                        else:
                            fig = px.bar(df, x=x, y=y)
                        st.subheader("Chart")
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.exception(e)

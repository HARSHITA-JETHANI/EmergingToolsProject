import streamlit as st
import pandas as pd
import os
import json
import re
from groq import Groq
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
import io

# ================= CONFIG =================
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
st.set_page_config(page_title="AI AutoML Assistant", layout="wide")

# ================= SESSION =================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= UI =================
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#f5f7ff,#eef2ff,#fdf2ff);}
section[data-testid="stSidebar"] {background: linear-gradient(180deg,#e0c3fc,#8ec5fc);color:white;}
h1,h2,h3 {color:#5a4fcf;}
.stButton>button {background:#cdb4db;color:white;border-radius:10px;border:none;}
.stButton>button:hover {background:#b583d6;}
[data-testid="stFileUploader"] button {background:#ff77b7!important;color:white!important;border-radius:12px!important;}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("📊 Dataset Dashboard")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

st.title("AI AutoML Data Assistant 🤖")
st.markdown("### Your AI-powered Data Scientist")

# ================= LLM PLANNER =================
def get_plan(user_query, df):
    schema = {
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict()
    }

    prompt = f"""
You are an expert data analyst.

Dataset schema:
{schema}

User question:
"{user_query}"

Respond ONLY in JSON:

{{
  "analysis": "describe | head | tail | missing | correlation | histogram | boxplot | scatter | value_counts | variance_max | skew_max | text",
  "columns": ["col1","col2"],
  "answer": "Direct answer",
  "explanation": "Short explanation"
}}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content.strip()

    try:
        return json.loads(content)
    except:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass

    return {"analysis": "text", "columns": [], "answer": "Error", "explanation": content}

# ================= EXECUTION =================
def execute_plan(plan, df):
    analysis = plan["analysis"]
    cols = plan.get("columns", [])
    answer = plan.get("answer", "")
    explanation = plan.get("explanation", "")
    numeric_df = df.select_dtypes(include="number")

    if analysis == "describe":
        return df.describe(), answer, explanation

    if analysis == "head":
        return df.head(), answer, explanation

    if analysis == "tail":
        return df.tail(), answer, explanation

    if analysis == "missing":
        return df.isna().sum().to_frame("Missing Values"), answer, explanation

    if analysis == "correlation":
        fig, ax = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close()
        return buf, answer, explanation

    if analysis == "histogram":
        col = cols[0] if cols else numeric_df.columns[0]
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close()
        return buf, answer, explanation

    if analysis == "boxplot":
        col = cols[0] if cols else numeric_df.columns[0]
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close()
        return buf, answer, explanation

    if analysis == "scatter" and len(cols) >= 2:
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[cols[0]], y=df[cols[1]], ax=ax)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close()
        return buf, answer, explanation

    if analysis == "value_counts" and cols:
        return df[cols[0]].value_counts(), answer, explanation

    if analysis == "variance_max":
        col = numeric_df.var().idxmax()
        return None, col, explanation

    if analysis == "skew_max":
        col = numeric_df.skew().abs().idxmax()
        return None, col, explanation

    return None, answer, explanation

# ================= MAIN =================
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.sidebar.write("Rows:", df.shape[0])
    st.sidebar.write("Columns:", df.shape[1])
    st.sidebar.write("Missing Values:", int(df.isnull().sum().sum()))

    with st.expander("🔍 Dataset Preview"):
        st.dataframe(df.head(), use_container_width=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["type"] == "text":
                st.markdown(msg["content"])
            elif msg["type"] == "dataframe":
                st.dataframe(msg["content"])
            elif msg["type"] == "image":
                st.image(msg["content"])

    user_query = st.chat_input("Ask anything about your dataset...")

    if user_query:
        st.session_state.messages.append({"role": "user", "type": "text", "content": user_query})

        plan = get_plan(user_query, df)
        result, answer, explanation = execute_plan(plan, df)

        if answer:
            st.session_state.messages.append({"role": "assistant", "type": "text", "content": f"**Answer:** {answer}"})

        if isinstance(result, pd.DataFrame) or isinstance(result, pd.Series):
            st.session_state.messages.append({"role": "assistant", "type": "dataframe", "content": result})

        elif isinstance(result, io.BytesIO):
            st.session_state.messages.append({"role": "assistant", "type": "image", "content": result})

        if explanation:
            st.session_state.messages.append({"role": "assistant", "type": "text", "content": explanation})

        st.rerun()


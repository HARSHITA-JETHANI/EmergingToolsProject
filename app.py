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

# ================= KNOWLEDGE BASE =================
eda_knowledge = [
    "Variance measures how spread out a feature is; high variance means values vary significantly.",
    "Skewness indicates asymmetry; positive skew means a long right tail, negative skew means left tail.",
    "Outliers are extreme values that can distort statistical analysis and machine learning models.",
    "Correlation measures linear relationships between variables; values close to 1 or -1 indicate strong relationships.",
    "A heatmap is used to visualize correlations between numerical variables.",
    "Histograms show the distribution of a numerical feature.",
    "Boxplots help detect outliers and understand quartile distribution.",
    "Scatter plots show relationships between two numerical variables.",
]

cleaning_knowledge = [
    "Missing values can be handled using imputation (mean, median, mode) or removal.",
    "Categorical variables often require encoding such as one-hot encoding or label encoding.",
    "Normalization scales values between 0 and 1, useful for neural networks.",
    "Standardization centers data around mean 0 and standard deviation 1.",
    "Dropping columns with too many missing values can improve model performance.",
    "Constant columns (zero variance) should be removed as they provide no useful information.",
]

feature_knowledge = [
    "Feature importance helps identify which variables contribute most to predictions.",
    "Highly correlated features can cause multicollinearity and should be handled carefully.",
    "Interaction features combine multiple variables to improve model performance.",
    "Date features can be expanded into year, month, day for better analysis.",
]

ml_knowledge = [
    "Regression models predict continuous values such as price or temperature.",
    "Classification models predict categories such as yes/no or fraud/not fraud.",
    "Overfitting occurs when a model learns noise instead of patterns.",
    "Train-test split is used to evaluate model performance on unseen data.",
    "Accuracy, precision, recall, and F1-score are evaluation metrics for classification.",
]

knowledge_base = (
    eda_knowledge +
    cleaning_knowledge +
    feature_knowledge +
    ml_knowledge
)


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

    # 🔥 RAG CONTEXT (THIS WAS MISSING)
    dataset_context = generate_dataset_insights(df)
    all_docs = dataset_context + knowledge_base
    context = retrieve_context(user_query, all_docs)

    prompt = f"""
You are an expert data analyst.

You MUST use the context provided to answer accurately.

Context:
{context}

Dataset schema:
{schema}

User question:
"{user_query}"

IMPORTANT RULES:
- Choose the correct analysis type
- Use the correct column names
- If user asks for a specific column → use THAT column
- Do NOT hallucinate column names
- Answer must match dataset

Respond ONLY in JSON:

{{
  "analysis": "describe | head | tail | missing | correlation | histogram | boxplot | scatter | value_counts | variance_max | skew_max | text| not_useful_column | outliers | datatype_info | unique_values ",
  "columns": ["col1","col2"],
  "answer": "",
  "explanation": "Short explanation using context"
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

    return {
        "analysis": "text",
        "columns": [],
        "answer": "Could not understand query",
        "explanation": content
    }
def safe_execute(plan, df):
    try:
        if not isinstance(plan, dict):
            return None, "Invalid response", "AI did not return valid JSON"

        if "analysis" not in plan:
            return None, "Missing analysis type", "Plan incomplete"

        result = execute_plan(plan, df)

        if result is None:
            return None, "Execution failed", "Could not execute plan"

        return result

    except Exception as e:
        return None, "Error handled safely", str(e)

# ================= RAG =================

def generate_dataset_insights(df):
    insights = []
    numeric_df = df.select_dtypes(include="number")

    for col in numeric_df.columns:
        insights.append(
            f"{col} has mean {df[col].mean():.2f}, std {df[col].std():.2f}"
        )

        if df[col].nunique() == 1:
            insights.append(f"{col} is constant and not useful")

        if df[col].skew() > 1:
            insights.append(f"{col} is highly skewed")

    return insights

def retrieve_context(query, docs, k=5):
    query = query.lower()
    scored = []

    for doc in docs:
        score = sum(word in doc.lower() for word in query.split())
        scored.append((score, doc))

    scored.sort(reverse=True)
    return [doc for _, doc in scored[:k]]

# ================= EXECUTION =================
def execute_plan(plan, df):
    analysis = plan["analysis"]
    cols = plan.get("columns", [])
    answer = plan.get("answer", "")
    explanation = plan.get("explanation", "")
    numeric_df = df.select_dtypes(include="number")

    # 🛑 SAFETY
    if numeric_df.empty:
        return None, "No numeric columns found", "Cannot perform numerical analysis"

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
        var_series = numeric_df.var()
        col = var_series.idxmax()
        value = var_series.max()

        explanation = f"""
    {col} has the highest variance.
    Variance value ≈ {value:.2f}.
    This indicates a wide spread of values.
    Feature is highly informative for modeling.
    """
        return None, col, explanation

    if analysis == "skew_max":
        skew_series = numeric_df.skew().abs()
        col = skew_series.idxmax()
        value = skew_series.max()

        explanation = f"""
    {col} is the most skewed feature.
    Skewness ≈ {value:.2f}.
    Distribution is highly asymmetric.
    May require transformation (log/scale).
    """
        return None, col, explanation

    if analysis == "not_useful_column":
        zero_var = numeric_df.columns[numeric_df.nunique() == 1]
        if len(zero_var) > 0:
            return None, zero_var[0], "Column has constant values (zero variance)"

        missing_ratio = df.isna().mean()
        high_missing = missing_ratio[missing_ratio > 0.5]
        if len(high_missing) > 0:
            return None, high_missing.index[0], "Column has more than 50% missing values"

        for col in df.columns:
            if df[col].nunique() == len(df):
                return None, col, "Likely an ID column (all unique values)"

        return None, "None", "All columns contain useful information"

    # 🔥 FIXED INDENTATION (THIS WAS BUGGED BEFORE)
    if analysis == "outliers":
        col = cols[0] if cols else numeric_df.columns[0]
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        return outliers, col, "Detected outliers using IQR"

    if analysis == "datatype_info":
        return df.dtypes, "Column data types", ""
    
    if analysis == "unique_values":
        col = cols[0] if cols else df.columns[0]
        count = df[col].nunique()
        explanation = f"{col} contains {count} unique values"
        return None, f"{count}", explanation

    # 🛑 FINAL SAFETY NET (NO CRASH EVER)
    return None, "Could not determine", "Try rephrasing your question"

def format_explanation(explanation):
    if not explanation:
        return ""
    
    points = re.split(r"[.\n]", explanation)
    points = [p.strip() for p in points if p.strip()]
    
    return "\n".join([f"- {p}" for p in points[:4]])

# ================= MAIN =================
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.sidebar.write("Rows:", df.shape[0])
    st.sidebar.write("Columns:", df.shape[1])
    st.sidebar.write("Missing Values:", int(df.isnull().sum().sum()))

    with st.expander("🔍 Dataset Preview"):
        st.dataframe(df.head(), use_container_width=True)

    # 🧠 DISPLAY CHAT HISTORY
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["type"] == "text":
                st.markdown(msg["content"])
            elif msg["type"] == "dataframe":
                st.dataframe(msg["content"])
            elif msg["type"] == "image":
                st.image(msg["content"])

    # 💬 USER INPUT
    user_query = st.chat_input("Ask anything about your dataset...")

    if user_query:
        # 👉 show user message immediately
        st.session_state.messages.append({
            "role": "user",
            "type": "text",
            "content": user_query
        })

        with st.chat_message("assistant"):
            with st.spinner("Thinking like a data scientist... 🤖"):

                plan = get_plan(user_query, df)
                result, answer, explanation = safe_execute(plan, df)

                # 🎯 CLEAN OUTPUT (NO DUPLICATES)
                output_text = ""

                if answer and answer != "":
                    st.markdown(f"### ✅ Answer: {answer}")
                    output_text += f"**Answer:** {answer}\n\n"

                # 📊 SHOW RESULT
                if isinstance(result, pd.DataFrame) or isinstance(result, pd.Series):
                    st.dataframe(result)

                elif isinstance(result, io.BytesIO):
                    st.image(result)

                # 🧠 BULLET EXPLANATION
                formatted_expl = format_explanation(explanation)

                if formatted_expl:
                    st.markdown("### 🧠 Explanation")
                    st.markdown(formatted_expl)
                    output_text += f"**Explanation:**\n{formatted_expl}"

                # 💾 SAVE ONLY ONE CLEAN MESSAGE
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "text",
                    "content": output_text
                })

        st.rerun()


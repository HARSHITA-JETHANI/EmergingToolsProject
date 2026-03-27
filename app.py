import streamlit as st
import pandas as pd
import os
import json
import re
import numpy as np
from groq import Groq
from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import io
import plotly.express as px

# ── RAG imports (TF-IDF based, no internet needed) ──────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= STATIC KNOWLEDGE BASE =================
KNOWLEDGE_BASE = [
    # EDA
    "Variance measures how spread out a feature is; high variance means values vary significantly.",
    "Skewness indicates asymmetry; positive skew means a long right tail, negative skew means left tail.",
    "Outliers are extreme values that can distort statistical analysis and machine learning models.",
    "Correlation measures linear relationships between variables; values close to 1 or -1 indicate strong relationships.",
    "A heatmap visualizes correlations between numerical variables using color intensity.",
    "Histograms show the frequency distribution of a numerical feature.",
    "Boxplots help detect outliers and understand quartile distribution of data.",
    "Scatter plots show relationships between two numerical variables.",
    # Cleaning
    "Missing values can be handled using imputation with mean, median, or mode, or by removal.",
    "Categorical variables often require encoding such as one-hot encoding or label encoding.",
    "Normalization scales values between 0 and 1, useful for distance-based models and neural networks.",
    "Standardization centers data around mean 0 and standard deviation 1, useful for linear models.",
    "Dropping columns with too many missing values can improve model performance.",
    "Constant columns with zero variance should be removed as they provide no useful information.",
    # Features
    "Feature importance identifies which variables contribute most to predictions.",
    "Highly correlated features can cause multicollinearity and should be handled carefully.",
    "Date or period features can be expanded into year, month, quarter for richer analysis.",
    # ML
    "Regression models predict continuous values such as price or temperature.",
    "Classification models predict categories such as yes/no or fraud/not fraud.",
    "Overfitting occurs when a model learns noise instead of underlying patterns.",
    "Train-test split is used to evaluate model performance on unseen data.",
    "Accuracy, precision, recall, and F1-score are common evaluation metrics for classification.",
]


# ================= CONFIG =================
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
st.set_page_config(page_title="AI AutoML Assistant", layout="wide")

# ================= SESSION STATE =================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_index" not in st.session_state:
    st.session_state.rag_index = None   # (vectorizer, tfidf_matrix, all_docs)
if "df_hash" not in st.session_state:
    st.session_state.df_hash = None


# ================= UI =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
.stApp {
    background: linear-gradient(135deg, #f5f7ff, #eef2ff, #fdf2ff);
    font-family: 'Space Grotesk', sans-serif;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e0c3fc, #8ec5fc);
}
h1, h2, h3 { color: #5a4fcf; }
.stButton>button {
    background: #cdb4db; color: white;
    border-radius: 10px; border: none;
}
.stButton>button:hover { background: #b583d6; }
[data-testid="stFileUploader"] button {
    background: #ff77b7 !important; color: white !important;
    border-radius: 12px !important;
}
.answer-box {
    background: linear-gradient(135deg, #e0c3fc22, #8ec5fc22);
    border-left: 4px solid #5a4fcf;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
}
.explanation-box {
    background: #f8f5ff;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.9em;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("📊 Dataset Dashboard")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
st.title("AI AutoML Data Assistant 🤖")
st.markdown("### Your AI-powered Data Scientist")


# ╔══════════════════════════════════════════════════════════════╗
# ║                    REAL RAG PIPELINE                         ║
# ║                                                              ║
# ║  Step 1 — INDEX: Convert dataset into rich text documents   ║
# ║           + combine with static knowledge base              ║
# ║                                                              ║
# ║  Step 2 — RETRIEVE: TF-IDF cosine similarity search        ║
# ║           (bigrams, sublinear TF, English stopwords)        ║
# ║           Returns top-k docs most relevant to query         ║
# ║                                                              ║
# ║  Step 3 — AUGMENT: Inject retrieved docs into LLM prompt   ║
# ║           so the model answers from actual data, not        ║
# ║           generic training knowledge                         ║
# ╚══════════════════════════════════════════════════════════════╝

def build_dataset_documents(df: pd.DataFrame) -> list:
    """
    Convert the DataFrame into rich natural-language documents
    for the RAG index. Each document describes a column or a
    dataset-level insight — so a question like
    'which industry earns the most' can semantically match
    a doc about Series_title_2 containing industry names.
    """
    docs = []
    numeric_df = df.select_dtypes(include="number")
    cat_df     = df.select_dtypes(exclude="number")

    # ── Per-column documents ──────────────────────────────────
    for col in df.columns:
        n_missing  = int(df[col].isna().sum())
        pct_miss   = n_missing / len(df) * 100
        n_unique   = df[col].nunique()
        base       = (f"Column '{col}' has {n_unique} unique values "
                      f"and {pct_miss:.1f}% missing data.")

        if col in numeric_df.columns:
            mean = df[col].mean()
            std  = df[col].std()
            skew = df[col].skew()
            mn, mx = df[col].min(), df[col].max()
            docs.append(
                f"{base} Numeric — mean={mean:.2f}, std={std:.2f}, "
                f"min={mn:.2f}, max={mx:.2f}, skewness={skew:.2f}. "
                f"Use for statistics, distribution, outlier, variance, or correlation."
            )
        else:
            top_vals = df[col].value_counts().head(5).index.tolist()
            docs.append(
                f"{base} Categorical — top values: {top_vals}. "
                f"Use for value_counts, groupby, or category analysis."
            )

    # ── Dataset-level summary ─────────────────────────────────
    docs.append(
        f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns. "
        f"Numeric columns: {list(numeric_df.columns)}. "
        f"Categorical columns: {list(cat_df.columns)}. "
        f"Total missing values: {int(df.isna().sum().sum())}."
    )

    # ── Top correlations ──────────────────────────────────────
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr()
        pairs = []
        cols_l = list(corr.columns)
        for i in range(len(cols_l)):
            for j in range(i + 1, len(cols_l)):
                pairs.append((abs(corr.iloc[i, j]), cols_l[i], cols_l[j]))
        pairs.sort(reverse=True)
        pair_strs = [f"'{a}' and '{b}' (r={v:.2f})" for v, a, b in pairs[:3]]
        docs.append(
            f"Strongest correlations: {'; '.join(pair_strs)}. "
            f"Use correlation heatmap to visualize all relationships."
        )

    # ── Variance & skew summary ───────────────────────────────
    if not numeric_df.empty:
        var_col  = numeric_df.var().idxmax()
        skew_col = numeric_df.skew().abs().idxmax()
        docs.append(
            f"'{var_col}' has the highest variance — values most spread out. "
            f"'{skew_col}' is the most skewed and may need log transformation."
        )

    # ── High-missing columns ──────────────────────────────────
    miss = df.isna().mean()
    high_miss = miss[miss > 0.3]
    if not high_miss.empty:
        docs.append(
            f"High missing data (>30%): {list(high_miss.index)}. "
            f"Consider dropping or imputing before modeling."
        )

    # ── Outlier summaries ─────────────────────────────────────
    for col in numeric_df.columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR    = Q3 - Q1
        n_out  = int(((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum())
        if n_out > 0:
            docs.append(
                f"'{col}' has {n_out} outliers (IQR method). "
                f"Bounds: lower={Q1-1.5*IQR:.2f}, upper={Q3+1.5*IQR:.2f}."
            )

    return docs


def build_rag_index(df: pd.DataFrame):
    """
    Build a TF-IDF index over all documents.
    Uses bigrams + sublinear_tf for richer semantic matching.
    Returns (vectorizer, tfidf_matrix, all_docs).
    """
    dataset_docs = build_dataset_documents(df)
    all_docs     = KNOWLEDGE_BASE + dataset_docs

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),    # unigrams + bigrams
        min_df=1,
        sublinear_tf=True,     # log-scale TF to dampen frequent terms
    )
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    return vectorizer, tfidf_matrix, all_docs


def retrieve_context(query: str, k: int = 6) -> tuple:
    """
    Retrieve top-k docs via TF-IDF cosine similarity.
    Returns (formatted_context_string, list_of_top_docs).
    """
    vectorizer, tfidf_matrix, all_docs = st.session_state.rag_index
    query_vec = vectorizer.transform([query])
    scores    = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx   = np.argsort(scores)[::-1][:k]
    top_docs  = [all_docs[i] for i in top_idx if scores[i] > 0]

    if not top_docs:
        top_docs = [all_docs[-1]]   # fallback: dataset summary

    return "\n".join(f"- {d}" for d in top_docs), top_docs


# ================= LLM PLANNER =================
def get_plan(user_query: str, df: pd.DataFrame) -> dict:
    schema = {
        "columns": list(df.columns),
        "dtypes":  df.dtypes.astype(str).to_dict(),
        "sample":  df.head(2).to_dict(orient="records"),
    }

    context_str, _ = retrieve_context(user_query, k=6)

    valid_analyses = (
        "describe | head | tail | missing | correlation | histogram | "
        "boxplot | scatter | value_counts | variance_max | skew_max | "
        "text | not_useful_column | outliers | datatype_info | unique_values"
    )

    prompt = f"""You are an expert data analyst assistant.

The following context was retrieved from a knowledge base using semantic search.
Use it to answer the user's question accurately.

RETRIEVED CONTEXT:
{context_str}

DATASET SCHEMA:
{json.dumps(schema, indent=2)}

USER QUESTION: "{user_query}"

RULES:
- Use ONLY column names that exist in the schema above.
- Choose the most appropriate analysis type from the list below.
- If the user asks for a chart, plot, or graph → use histogram, boxplot, scatter, or correlation.
- If the user asks about categories or groups → use value_counts.
- Do NOT make up column names.
- Base your explanation on the retrieved context above, not generic knowledge.
- STRICTLY follow numeric values mentioned in the query.
- Example: if user says "first 6 rows", use exactly 6 — DO NOT default to 10.

Respond with ONLY a valid JSON object — no explanation, no markdown, no backticks:

{{
  "analysis": "<one of: {valid_analyses}>",
  "columns": ["col1", "col2"],
  "answer": "<direct short answer if known, else empty string>",
  "explanation": "<2-3 sentences grounded in the retrieved context above>"
}}"""

    content = ""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content.strip()
        content = re.sub(r"```(?:json)?", "", content).strip().rstrip("```").strip()
        return json.loads(content)

    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    except Exception:
        pass

    return {
        "analysis": "text",
        "columns":  [],
        "answer":   "Could not parse AI response. Please rephrase.",
        "explanation": "",
    }


# ================= CHART CONFIG =================
CHART_W, CHART_H = 5.5, 3.5   # compact chart size (inches)

def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ================= EXECUTION =================
def execute_plan(plan: dict, df: pd.DataFrame):
    """Returns (result, answer, explanation)"""
    analysis   = plan.get("analysis", "text")
    cols       = plan.get("columns", [])
    answer     = plan.get("answer", "")
    expl       = plan.get("explanation", "")
    numeric_df = df.select_dtypes(include="number")

    def pick_col(preferred, pool):
        for c in preferred:
            if c in df.columns:
                return c
        return pool[0] if pool else None

    # ── Tabular results ───────────────────────────────────────
    if analysis == "describe":
        return df.describe(), answer or "Statistical summary", expl

    if analysis == "head":
        return df.head(10), answer or "First 10 rows", expl

    if analysis == "tail":
        return df.tail(10), answer or "Last 10 rows", expl

    if analysis == "missing":
        m = df.isna().sum().to_frame("Missing")
        m["% Missing"] = (df.isna().mean() * 100).round(2)
        return m, answer or "Missing value counts", expl

    if analysis == "datatype_info":
        info = df.dtypes.to_frame("Data Type")
        info["Non-Null"] = df.notnull().sum()
        info["Unique"]   = df.nunique()
        return info, answer or "Column data types", expl

    if analysis == "unique_values":
        col = pick_col(cols, list(df.columns))
        if col is None:
            return None, "No column found", ""
        count  = df[col].nunique()
        sample = df[col].dropna().unique()[:10].tolist()
        return None, f"'{col}' has {count} unique values", f"Sample: {sample}"

    if analysis == "not_useful_column":
        reasons = {}
        for c in numeric_df.columns:
            if numeric_df[c].nunique() == 1:
                reasons[c] = "Constant (zero variance)"
        miss_ratio = df.isna().mean()
        for c in miss_ratio[miss_ratio > 0.5].index:
            reasons[c] = f"{miss_ratio[c]*100:.1f}% missing"
        for c in df.columns:
            if df[c].nunique() == len(df) and c not in reasons:
                reasons[c] = "All unique (likely ID)"
        if reasons:
            rdf = pd.DataFrame(list(reasons.items()), columns=["Column", "Reason"])
            return rdf, "Potentially useless columns", expl
        return None, "All columns appear useful", "No constant/high-missing/ID columns detected."

    # ── Charts (all compact size) ─────────────────────────────
    if analysis == "correlation":
        if numeric_df.shape[1] < 2:
            return None, "Need ≥2 numeric columns", ""
        fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
        sns.heatmap(
            numeric_df.corr(), annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.4,
            annot_kws={"size": 7}, ax=ax
        )
        ax.set_title("Correlation Heatmap", fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)
        return fig_to_bytes(fig), answer or "Correlation heatmap", expl

    if analysis == "histogram":
        col = pick_col(cols, list(numeric_df.columns))
        if col is None:
            return None, "No numeric column found", ""
        fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#7c6fcd")
        ax.set_title(f"Distribution: {col}", fontsize=10, fontweight="bold")
        ax.set_xlabel(col, fontsize=8)
        ax.set_ylabel("Frequency", fontsize=8)
        ax.tick_params(labelsize=7)
        return fig_to_bytes(fig), answer or f"Histogram: {col}", expl

    if analysis == "boxplot":
        col = pick_col(cols, list(numeric_df.columns))
        if col is None:
            return None, "No numeric column found", ""
        fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
        sns.boxplot(y=df[col].dropna(), ax=ax, color="#cdb4db", width=0.4)
        ax.set_title(f"Boxplot: {col}", fontsize=10, fontweight="bold")
        ax.set_ylabel(col, fontsize=8)
        ax.tick_params(labelsize=7)
        return fig_to_bytes(fig), answer or f"Boxplot: {col}", expl

    if analysis == "scatter":
        valid = [c for c in cols if c in numeric_df.columns]
        if len(valid) < 2:
            valid = list(numeric_df.columns[:2])
        if len(valid) < 2:
            return None, "Need ≥2 numeric columns for scatter", ""
        fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
        sns.scatterplot(x=df[valid[0]], y=df[valid[1]], ax=ax,
                        alpha=0.5, color="#5a4fcf", s=15)
        ax.set_title(f"{valid[0]} vs {valid[1]}", fontsize=10, fontweight="bold")
        ax.set_xlabel(valid[0], fontsize=8)
        ax.set_ylabel(valid[1], fontsize=8)
        ax.tick_params(labelsize=7)
        return fig_to_bytes(fig), answer or f"Scatter: {valid[0]} vs {valid[1]}", expl

    if analysis == "value_counts":
        col = pick_col(cols, list(df.columns))
        if col is None:
            return None, "No column found", ""
        vc = df[col].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
        vc.plot(kind="bar", ax=ax, color="#8ec5fc", edgecolor="#5a4fcf", width=0.6)
        ax.set_title(f"Value Counts: {col}", fontsize=10, fontweight="bold")
        ax.set_xlabel(col, fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(labelsize=7)
        plt.xticks(rotation=40, ha="right")
        plt.tight_layout()
        return fig_to_bytes(fig), answer or f"Value counts: {col}", expl

    if analysis == "variance_max":
        var_s      = numeric_df.var().sort_values(ascending=False)
        col, val   = var_s.idxmax(), var_s.max()
        fig, ax    = plt.subplots(figsize=(CHART_W, CHART_H))
        var_s.plot(kind="bar", ax=ax, color="#cdb4db", edgecolor="#5a4fcf", width=0.6)
        ax.set_title("Feature Variances", fontsize=10, fontweight="bold")
        ax.set_ylabel("Variance", fontsize=8)
        ax.tick_params(labelsize=7)
        plt.xticks(rotation=40, ha="right")
        plt.tight_layout()
        return (
            fig_to_bytes(fig),
            f"Highest variance: {col} ({val:.2f})",
            expl or f"'{col}' variance={val:.2f} — most spread out.",
        )

    if analysis == "skew_max":
        skew_s    = numeric_df.skew().abs().sort_values(ascending=False)
        col, val  = skew_s.idxmax(), skew_s.max()
        fig, ax   = plt.subplots(figsize=(CHART_W, CHART_H))
        skew_s.plot(kind="bar", ax=ax, color="#e0c3fc", edgecolor="#5a4fcf", width=0.6)
        ax.set_title("Feature Skewness", fontsize=10, fontweight="bold")
        ax.set_ylabel("Skewness", fontsize=8)
        ax.tick_params(labelsize=7)
        plt.xticks(rotation=40, ha="right")
        plt.tight_layout()
        return (
            fig_to_bytes(fig),
            f"Most skewed: {col} (skew={val:.2f})",
            expl or f"'{col}' skewness={val:.2f}. Consider log transformation.",
        )

    if analysis == "outliers":
        col = pick_col(cols, list(numeric_df.columns))
        if col is None:
            return None, "No numeric column found", ""
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR    = Q3 - Q1
        lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        n_out  = int(((df[col] < lo) | (df[col] > hi)).sum())

        fig, axes = plt.subplots(1, 2, figsize=(CHART_W * 1.6, CHART_H))
        sns.boxplot(y=df[col], ax=axes[0], color="#cdb4db", width=0.4)
        axes[0].set_title(f"Boxplot: {col}", fontsize=9, fontweight="bold")
        axes[0].tick_params(labelsize=7)

        sns.histplot(df[col], kde=True, ax=axes[1], color="#7c6fcd", bins=30)
        axes[1].axvline(lo, color="red", linestyle="--", linewidth=1,
                        label=f"Lower {lo:.1f}")
        axes[1].axvline(hi, color="red", linestyle="--", linewidth=1,
                        label=f"Upper {hi:.1f}")
        axes[1].legend(fontsize=7)
        axes[1].set_title(f"Distribution: {col}", fontsize=9, fontweight="bold")
        axes[1].tick_params(labelsize=7)
        plt.tight_layout()
        return (
            fig_to_bytes(fig),
            f"{n_out} outliers in '{col}'",
            expl or f"IQR bounds: lower={lo:.2f}, upper={hi:.2f}.",
        )

    # ── Fallback ──────────────────────────────────────────────
    return None, answer or "Analysis complete", expl or "No further details."


def safe_execute(plan, df):
    try:
        if not isinstance(plan, dict) or "analysis" not in plan:
            return None, "Invalid plan from AI", "Try rephrasing."
        return execute_plan(plan, df)
    except Exception as e:
        return None, "Execution error", str(e)


# ================= DISPLAY =================
def render_message(msg):
    if msg["type"] == "text":
        st.markdown(msg["content"])
    elif msg["type"] == "dataframe":
        st.dataframe(msg["content"], use_container_width=False)
    elif msg["type"] == "image":
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(msg["content"])


def display_result(result, answer, explanation):
    stored = []

    if answer:
        st.markdown(
            f'<div class="answer-box"><b>✅ {answer}</b></div>',
            unsafe_allow_html=True,
        )
        stored.append({"role": "assistant", "type": "text",
                        "content": f"**✅ {answer}**"})

    if isinstance(result, (pd.DataFrame, pd.Series)):
        st.dataframe(result, use_container_width=False)
        stored.append({"role": "assistant", "type": "dataframe", "content": result})

    elif isinstance(result, bytes):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(result)
        stored.append({"role": "assistant", "type": "image", "content": result})

    if explanation:
        points    = [p.strip() for p in re.split(r"[.\n]", explanation) if p.strip()]
        formatted = "\n".join(f"- {p}" for p in points[:4])
        st.markdown(
            f'<div class="explanation-box"><b>🧠 Explanation</b><br>{formatted}</div>',
            unsafe_allow_html=True,
        )
        stored.append({"role": "assistant", "type": "text",
                        "content": f"**🧠 Explanation**\n{formatted}"})

    return stored


# ================= AI SMART DASHBOARD BUTTON (ALWAYS VISIBLE) =================

st.markdown("## 🧠 AI Smart Dashboard")

if "dashboard_on" not in st.session_state:
    st.session_state.dashboard_on = False

if st.button("🚀 Generate Smart Dashboard"):
    st.session_state.dashboard_on = True

# ================= MAIN =================
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Build / refresh RAG index only when dataset changes
    df_hash = str(df.shape) + str(df.columns.tolist()) + str(df.head().to_json())

    if st.session_state.df_hash != df_hash:
        st.session_state.dashboard_on = False   # ✅ FIX: prevent auto dashboard
        with st.spinner("🔍 Building semantic search index..."):
            st.session_state.rag_index = build_rag_index(df)
            st.session_state.df_hash   = df_hash

    # Sidebar
    st.sidebar.write("📏 Rows:", df.shape[0])
    st.sidebar.write("📐 Columns:", df.shape[1])
    st.sidebar.write("❓ Missing:", int(df.isnull().sum().sum()))
    st.sidebar.write("🔢 Numeric:", df.select_dtypes(include="number").shape[1])
    st.sidebar.write("🔤 Categorical:", df.select_dtypes(exclude="number").shape[1])
    if st.session_state.rag_index:
        n_docs = len(st.session_state.rag_index[2])
        st.sidebar.markdown(f"**🗂 RAG index:** {n_docs} documents")

    # Preview
    with st.expander("🔍 Dataset Preview"):
        st.dataframe(df.head(), use_container_width=True)

    st.markdown("---")

    # ---------------- DASHBOARD LOGIC ----------------
    if st.session_state.dashboard_on:

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", int(df.isnull().sum().sum()))

        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if numeric_cols:

            selected_col = st.selectbox(
                "📊 Select Column for Analysis",
                numeric_cols,
                key="selected_column"
            )

            # Histogram
            fig = px.histogram(
                df,
                x=selected_col,
                nbins=30,
                title=f"Distribution of {selected_col}",
                color_discrete_sequence=["#cdb4db"]
            )

            fig.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor="white",
                paper_bgcolor="white",
            )

            # ✅ internal grid (clean)
            fig.update_xaxes(showgrid=True, gridcolor="#e6e6e6")
            fig.update_yaxes(showgrid=True, gridcolor="#e6e6e6")

            # ✅ bin separation
            fig.update_traces(
                marker_line_color="white",
                marker_line_width=1.2,
                opacity=0.9
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"hist_{selected_col}"
            )


            # Boxplot
            fig2 = px.box(
                df,
                y=selected_col,
                title=f"Outlier Detection for {selected_col}",
                color_discrete_sequence=["#ffc8dd"]
            )

            fig2.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor="white",
                paper_bgcolor="white",
            )

            fig2.update_yaxes(showgrid=True, gridcolor="#e6e6e6")

            st.plotly_chart(
                fig2,
                use_container_width=True,
                key=f"box_{selected_col}"
            )

            # AI Insights
            st.markdown("### 🧠 AI Insights")

            query = f"Analyze column {selected_col} and give insights"
            context, docs = retrieve_context(query)

            insight_prompt = f"""
            Column: {selected_col}
            Context:
            {context}

            Give 3 short insights about this column.
            """

            with st.spinner("🤖 Generating insights..."):
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": insight_prompt}]
                )

            insights = response.choices[0].message.content
            st.success(insights)

        else:
            st.warning("No numeric columns available for visualization.")

    # ---------------- CHAT (FIXED) ----------------
    st.markdown("## 💬 Ask Questions")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            render_message(msg)

    user_query = st.chat_input("Ask anything about your dataset...")

    if user_query:
        st.session_state.messages.append(
            {"role": "user", "type": "text", "content": user_query}
        )

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking like a data scientist... 🤖"):
                plan = get_plan(user_query, df)
                result, answer, explanation = safe_execute(plan, df)
                new_msgs = display_result(result, answer, explanation)
                st.session_state.messages.extend(new_msgs)


# ---------------- NO DATA CASE ----------------
else:
    if st.session_state.dashboard_on:
        st.warning("⚠️ Upload a dataset first to use the dashboard.")

    st.info("👈 Upload a CSV file from the sidebar to get started.")

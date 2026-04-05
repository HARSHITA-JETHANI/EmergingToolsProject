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

# ── RAG imports ──────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── AutoML imports ───────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.ensemble        import (RandomForestClassifier, RandomForestRegressor,
                                     GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.linear_model    import LogisticRegression, LinearRegression, Ridge
from sklearn.metrics         import (accuracy_score, f1_score,
                                     mean_squared_error, r2_score, mean_absolute_error)

# ═════════════════════════════════════════════════════════════════════════════
# STATIC KNOWLEDGE BASE
# ═════════════════════════════════════════════════════════════════════════════
KNOWLEDGE_BASE = [
    "Variance measures how spread out a feature is; high variance means values vary significantly.",
    "Skewness indicates asymmetry; positive skew means a long right tail, negative skew means left tail.",
    "Outliers are extreme values that can distort statistical analysis and machine learning models.",
    "Correlation measures linear relationships between variables; values close to 1 or -1 indicate strong relationships.",
    "A heatmap visualizes correlations between numerical variables using color intensity.",
    "Histograms show the frequency distribution of a numerical feature.",
    "Boxplots help detect outliers and understand quartile distribution of data.",
    "Scatter plots show relationships between two numerical variables.",
    "Bar charts compare values across categories and are useful for grouped or aggregated data.",
    "Line charts show trends over time or an ordered period column.",
    "Grouped bar charts compare a numeric metric across two categorical dimensions simultaneously.",
    "Pie charts show the proportion of each category, best for fewer than 6 categories.",
    "Missing values can be handled using imputation with mean, median, or mode, or by removal.",
    "Categorical variables often require encoding such as one-hot encoding or label encoding.",
    "Normalization scales values between 0 and 1, useful for distance-based models and neural networks.",
    "Standardization centers data around mean 0 and standard deviation 1, useful for linear models.",
    "Dropping columns with too many missing values can improve model performance.",
    "Constant columns with zero variance should be removed as they provide no useful information.",
    "ID-like columns where every value is unique are not useful features for modeling.",
    "Feature importance identifies which variables contribute most to predictions.",
    "Highly correlated features can cause multicollinearity and should be handled carefully.",
    "Date or period features can be expanded into year, month, quarter for richer analysis.",
    "Groupby aggregations like mean, sum, count per category reveal useful patterns.",
    "Log transformation reduces right skewness and can improve model performance.",
    "Classification predicts categories. Common algorithms: Logistic Regression, Random Forest, Gradient Boosting.",
    "Regression predicts continuous values. Common algorithms: Linear Regression, Random Forest, Ridge.",
    "Random Forest is an ensemble of decision trees — robust, handles non-linearity, gives feature importance.",
    "Gradient Boosting builds trees sequentially — often achieves higher accuracy than Random Forest.",
    "Logistic Regression is fast and interpretable — good baseline for classification tasks.",
    "Overfitting occurs when a model learns noise instead of underlying patterns.",
    "Train-test split (usually 80/20) evaluates how well a model generalises to unseen data.",
    "Accuracy measures overall correct predictions. F1-score balances precision and recall.",
    "R-squared measures how much variance the regression model explains. Closer to 1 is better.",
    "RMSE penalises large prediction errors more than MAE.",
    "Feature scaling (StandardScaler) is important for linear models and distance-based algorithms.",
    "Class imbalance: one target class has far more samples — use F1 score not just accuracy.",
    "Cross-validation gives a more reliable performance estimate than a single train-test split.",
    "Confusion matrix shows true positives, false positives, true negatives, false negatives.",
    "Feature importance plot shows which columns the model relies on most for predictions.",
    "Natural language filter: use df.query() to filter rows before running any analysis.",
]

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
st.set_page_config(page_title="AI AutoML Assistant", layout="wide")

# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═════════════════════════════════════════════════════════════════════════════
for _k, _v in {
    "messages":      [],
    "rag_index":     None,
    "df_hash":       None,
    "dashboard_on":  False,
    "chat_history":  [],
    "active_filter": None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ═════════════════════════════════════════════════════════════════════════════
# UI / STYLES  — Refined dark-accent editorial theme
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@700&display=swap');

/* ── Global ── */
.stApp {
    background: #f9f8f6;
    font-family: 'DM Sans', sans-serif;
    color: #1a1a2e;
}
section[data-testid="stSidebar"] {
    background: #1a1a2e;
    border-right: 1px solid #2d2d4e;
}
section[data-testid="stSidebar"] * {
    color: #e8e6f0 !important;
}
section[data-testid="stSidebar"] .stButton>button {
    background: #e63946 !important;
    color: white !important;
    border-radius: 8px;
    width: 100%;
}

/* ── Sidebar file uploader — force visibility ── */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: #2d2d4e !important;
    border: 2px dashed #a78bfa !important;
    border-radius: 10px !important;
    padding: 6px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] p,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] span,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] small,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] div {
    color: #000000 !important;
    font-size: 0.82rem !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: #7c3aed !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    width: 100% !important;
    margin-top: 6px !important;
}
}
/* Uploaded file name pill */
section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] {
    background: #3d3d5e !important;
    border-radius: 8px !important;
    padding: 6px 10px !important;
    margin-top: 6px !important;
    border: 1px solid #7c3aed !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderFileName"] {
    color: #e8e6f0 !important;
    font-size: 0.82rem !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderFileSize"] {
    color: #a78bfa !important;
    font-size: 0.75rem !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDeleteBtn"] button {
    background: transparent !important;
    color: #e63946 !important;
    border: none !important;
    padding: 2px 6px !important;
}
/* Upload label */
section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p {
    color: #c4b5fd !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* ── Headings ── */
h1 { font-family: 'Playfair Display', serif !important; color: #1a1a2e !important; font-size: 2.4rem !important; }
h2, h3 { font-family: 'DM Sans', sans-serif !important; color: #1a1a2e !important; font-weight: 600 !important; }

/* ── Sidebar metrics ── */
.sidebar-metric {
    display: flex;
    justify-content: space-between;
    padding: 5px 0;
    border-bottom: 1px solid #2d2d4e;
    font-size: 0.83rem;
}
.sidebar-metric-val {
    font-family: 'DM Mono', monospace;
    color: #a78bfa !important;
    font-weight: 500;
}

/* ── Chat bubbles ── */
.user-bubble {
    background: #1a1a2e;
    color: #f0eefc;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px;
    margin: 6px 0 6px 60px;
    font-size: 0.92rem;
    line-height: 1.55;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.assistant-bubble {
    background: #ffffff;
    color: #1a1a2e;
    border-radius: 18px 18px 18px 4px;
    padding: 14px 18px;
    margin: 6px 60px 6px 0;
    font-size: 0.92rem;
    line-height: 1.6;
    border: 1px solid #e8e3f8;
    box-shadow: 0 2px 10px rgba(100,80,200,0.07);
}

/* ── Answer box ── */
.answer-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #2d2d4e 100%);
    color: #f0eefc;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 1.0rem;
    font-weight: 500;
    letter-spacing: 0.01em;
    box-shadow: 0 4px 16px rgba(26,26,46,0.18);
}
.answer-box .label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #a78bfa;
    margin-bottom: 5px;
    font-family: 'DM Mono', monospace;
}

/* ── Explanation box ── */
.explanation-box {
    background: #ffffff;
    border: 1px solid #e8e3f8;
    border-left: 4px solid #7c3aed;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.88rem;
    line-height: 1.7;
    color: #2d2d4e;
}
.explanation-box .expl-title {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #7c3aed;
    font-family: 'DM Mono', monospace;
    margin-bottom: 8px;
    font-weight: 600;
}
.explanation-box ul {
    margin: 0;
    padding-left: 18px;
}
.explanation-box li {
    margin-bottom: 5px;
}

/* ── Filter banner ── */
.filter-banner {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-left: 4px solid #f59e0b;
    border-radius: 0 10px 10px 0;
    padding: 10px 16px;
    margin: 6px 0;
    font-size: 0.85rem;
    color: #78350f;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── ML box ── */
.ml-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-left: 4px solid #3b82f6;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.88rem;
    color: #1e3a5f;
    line-height: 1.65;
}
.ml-box .ml-title {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #3b82f6;
    font-family: 'DM Mono', monospace;
    margin-bottom: 8px;
    font-weight: 600;
}

/* ── Insight cards ── */
.insight-card {
    background: #ffffff;
    border: 1px solid #e8e3f8;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.88rem;
    color: #1a1a2e;
    line-height: 1.6;
    display: flex;
    gap: 10px;
    align-items: flex-start;
    box-shadow: 0 1px 4px rgba(124,58,237,0.06);
}
.insight-card .icon {
    font-size: 1.1rem;
    flex-shrink: 0;
    margin-top: 1px;
}
.insight-section-title {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #7c3aed;
    font-family: 'DM Mono', monospace;
    margin: 12px 0 6px;
    font-weight: 600;
}

/* ── Download buttons ── */
.stDownloadButton > button {
    background: #f5f3ff !important;
    color: #7c3aed !important;
    border: 1px solid #ddd6fe !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    padding: 6px 14px !important;
    font-family: 'DM Mono', monospace !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
    background: #ede9fe !important;
    border-color: #7c3aed !important;
}

/* ── Primary button ── */
.stButton > button {
    background: #1a1a2e !important;
    color: #f0eefc !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.92rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #2d2d4e !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(26,26,46,0.2) !important;
}

/* ── Chat input ── */
/* ── Chat input ── */
.stChatInput {
    border-radius: 12px !important;
}
.stChatInput > div {
    background: #ffffff !important;
    border-radius: 12px !important;
    border: 2px solid #ddd6fe !important;
    padding: 4px 8px !important;
    overflow: hidden !important;
}
.stChatInput textarea {
    background: #ffffff !important;
    color: #1a1a2e !important;
    border: none !important;
    border-radius: 0px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    box-shadow: none !important;
}
.stChatInput textarea:focus {
    box-shadow: none !important;
    border: none !important;
    outline: none !important;
}
[data-testid="stChatInput"] {
    border-radius: 12px !important;
    overflow: hidden !important;
}
[data-testid="stChatInputContainer"] {
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px !important; overflow: hidden; }

/* ── Expander ── */
.streamlit-expanderHeader {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    color: #1a1a2e !important;
}

/* ── Divider ── */
hr { border-color: #e8e3f8 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #7c3aed !important; }

/* ── Metric ── */
[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    color: #1a1a2e !important;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("📊 Dataset Dashboard")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
st.title("AI AutoML Assistant")
st.markdown("##### Intelligent data analysis, visualization & AutoML — powered by LLaMA 3.3")


# ═════════════════════════════════════════════════════════════════════════════
# RAG PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
def build_dataset_documents(df: pd.DataFrame) -> list:
    docs = []
    numeric_df = df.select_dtypes(include="number")
    cat_df     = df.select_dtypes(exclude="number")

    for col in df.columns:
        pct_miss = df[col].isna().mean() * 100
        n_unique = df[col].nunique()
        base     = f"Column '{col}': {n_unique} unique values, {pct_miss:.1f}% missing."

        if col in numeric_df.columns:
            s = df[col].dropna()
            docs.append(
                f"{base} Numeric. Mean={s.mean():.2f}, median={s.median():.2f}, "
                f"std={s.std():.2f}, min={s.min():.2f}, max={s.max():.2f}, "
                f"skewness={s.skew():.2f}, kurtosis={s.kurtosis():.2f}. "
                f"Use for histogram, boxplot, scatter, distribution, outlier, variance analysis."
            )
        else:
            top = df[col].value_counts().head(5)
            top_str = ", ".join([f"'{v}'={c}" for v, c in top.items()])
            docs.append(
                f"{base} Categorical. Top values: {top_str}. "
                f"Use for bar chart, value_counts, groupby, category comparison."
            )

    docs.append(
        f"Dataset overview: {df.shape[0]} rows, {df.shape[1]} columns. "
        f"Numeric columns: {list(numeric_df.columns)}. "
        f"Categorical columns: {list(cat_df.columns)}. "
        f"Total missing values: {int(df.isna().sum().sum())}."
    )

    for cat_col in cat_df.columns:
        if df[cat_col].nunique() > 30:
            continue
        for num_col in numeric_df.columns:
            try:
                grp_mean = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False)
                top3m    = grp_mean.head(3)
                docs.append(
                    f"Average {num_col} by {cat_col}: "
                    + ", ".join([f"'{k}'={v:.2f}" for k, v in top3m.items()])
                    + f". Use grouped bar chart or groupby to compare {num_col} across {cat_col}."
                )
                grp_sum = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False)
                top3s   = grp_sum.head(3)
                docs.append(
                    f"Total {num_col} by {cat_col}: "
                    + ", ".join([f"'{k}'={v:.2f}" for k, v in top3s.items()])
                    + f". Use for sum/total queries about {num_col} grouped by {cat_col}."
                )
            except Exception:
                pass

    period_cols = [c for c in df.columns
                   if any(k in c.lower() for k in ["period", "year", "date", "time", "month"])]
    for pc in period_cols:
        for num_col in numeric_df.columns:
            try:
                trend  = df.groupby(pc)[num_col].mean().sort_index()
                if len(trend) < 2:
                    continue
                first_v, last_v = trend.iloc[0], trend.iloc[-1]
                direction = "increasing" if last_v > first_v else "decreasing"
                docs.append(
                    f"Trend of {num_col} over {pc}: {direction} from {first_v:.2f} to {last_v:.2f}. "
                    f"Use a line chart to visualise {num_col} over {pc}."
                )
            except Exception:
                pass

    for num_col in numeric_df.columns:
        try:
            top_v = df[num_col].max()
            bot_v = df[num_col].min()
            docs.append(
                f"Highest {num_col} = {top_v:.2f}. Lowest {num_col} = {bot_v:.2f}. "
                f"Relevant for: highest, lowest, maximum, minimum, top, bottom queries."
            )
        except Exception:
            pass

    if numeric_df.shape[1] >= 2:
        corr  = numeric_df.corr()
        pairs = []
        cl    = list(corr.columns)
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                pairs.append((abs(corr.iloc[i, j]), corr.iloc[i, j], cl[i], cl[j]))
        pairs.sort(reverse=True)
        for _, rv, a, b in pairs[:5]:
            direction = "positive" if rv > 0 else "negative"
            strength  = "strong" if abs(rv) > 0.7 else ("moderate" if abs(rv) > 0.4 else "weak")
            docs.append(
                f"'{a}' and '{b}' have a {strength} {direction} correlation r={rv:.2f}. "
                f"Use scatter plot or heatmap to visualise this relationship."
            )

    if not numeric_df.empty:
        var_col  = numeric_df.var().idxmax()
        skew_col = numeric_df.skew().abs().idxmax()
        docs.append(
            f"'{var_col}' has the highest variance ({numeric_df.var().max():.2f}). "
            f"'{skew_col}' is the most skewed (skew={numeric_df.skew().abs().max():.2f})."
        )

    for col in numeric_df.columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR    = Q3 - Q1
        n_out  = int(((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum())
        if n_out > 0:
            docs.append(
                f"'{col}' has {n_out} outliers (IQR). "
                f"Lower={Q1 - 1.5*IQR:.2f}, upper={Q3 + 1.5*IQR:.2f}."
            )

    miss = df.isna().mean() * 100
    high = miss[miss > 20]
    if not high.empty:
        docs.append(
            "Columns with high missing data: "
            + ", ".join([f"'{c}' ({v:.1f}%)" for c, v in high.items()])
            + ". Consider dropping (>50%) or imputing."
        )

    return docs


def build_rag_index(df: pd.DataFrame):
    dataset_docs = build_dataset_documents(df)
    all_docs     = KNOWLEDGE_BASE + dataset_docs
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        max_features=8000,
    )
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    return vectorizer, tfidf_matrix, all_docs


def retrieve_context(query: str, k: int = 8) -> tuple:
    vectorizer, tfidf_matrix, all_docs = st.session_state.rag_index
    query_vec = vectorizer.transform([query])
    scores    = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx   = np.argsort(scores)[::-1][:k]
    top_docs  = [all_docs[i] for i in top_idx if scores[i] > 0.01]
    if not top_docs:
        top_docs = [all_docs[-1]]
    return "\n".join(f"- {d}" for d in top_docs), top_docs


# ═════════════════════════════════════════════════════════════════════════════
# CONVERSATION MEMORY
# Stores last 5 clean Q&A exchanges. Each assistant entry is a compact
# one-line factual summary — NOT the raw explanation dump — so the LLM
# context stays clean across many questions.
# ═════════════════════════════════════════════════════════════════════════════
MEMORY_LIMIT = 10   # 5 exchanges × 2 messages

def update_memory(role: str, content: str):
    """Store a clean, trimmed message in memory."""
    # For assistant messages: strip markdown bold/code artifacts, keep to 120 chars
    if role == "assistant":
        content = re.sub(r"\*\*|`", "", content).strip()
        content = content[:120] + ("…" if len(content) > 120 else "")
    st.session_state.chat_history.append({"role": role, "content": content})
    if len(st.session_state.chat_history) > MEMORY_LIMIT:
        # Always keep the earliest system context + drop oldest pair
        st.session_state.chat_history = st.session_state.chat_history[-MEMORY_LIMIT:]


def get_memory_messages() -> list:
    """Return memory formatted for the LLM messages array."""
    return [{"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_history]


def build_memory_summary(answer: str, explanation: str) -> str:
    """
    Build a clean one-line assistant memory entry.
    Uses the answer headline + first sentence of explanation only.
    """
    clean_answer = re.sub(r"\*\*|`", "", answer).strip()
    if explanation:
        first_sentence = re.split(r"(?<=[.!?])\s+", explanation.strip())[0]
        first_sentence = re.sub(r"\*\*|`", "", first_sentence).strip()
        return f"{clean_answer}. {first_sentence}"
    return clean_answer


# ═════════════════════════════════════════════════════════════════════════════
# NATURAL LANGUAGE FILTER
# ═════════════════════════════════════════════════════════════════════════════
def detect_and_apply_filter(user_query: str, df: pd.DataFrame) -> tuple:
    filter_keywords = ["filter", "only", "where", "show only", "just",
                       "subset", "rows where", "limit to", "select rows"]
    has_intent = any(kw in user_query.lower() for kw in filter_keywords)

    if not has_intent:
        if st.session_state.active_filter:
            try:
                filtered = df.query(st.session_state.active_filter["query"])
                if len(filtered) > 0:
                    return filtered, st.session_state.active_filter["description"]
            except Exception:
                st.session_state.active_filter = None
        return df, None

    schema = {
        "columns": list(df.columns),
        "dtypes":  df.dtypes.astype(str).to_dict(),
        "sample":  df.head(3).to_dict(orient="records"),
    }

    filter_prompt = f"""You are a pandas expert. Convert the user's filter request into a df.query() string.

Dataset schema:
{json.dumps(schema, indent=2)}

User request: "{user_query}"

Rules:
- Use ONLY column names from the schema
- String comparisons must use single quotes: Series_title_2 == 'Mining'
- Numeric comparisons: Data_value > 5000
- If no clear filter exists, return has_filter=false

Respond with ONLY valid JSON:
{{
  "query": "column_name == 'value'",
  "description": "Filtered to rows where ...",
  "has_filter": true
}}"""

    try:
        resp    = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": filter_prompt}],
            temperature=0,
        )
        content = resp.choices[0].message.content.strip()
        content = re.sub(r"```(?:json)?", "", content).strip().rstrip("```").strip()
        parsed  = json.loads(content)

        if parsed.get("has_filter") and parsed.get("query"):
            filtered = df.query(parsed["query"])
            if len(filtered) > 0:
                st.session_state.active_filter = {
                    "query":       parsed["query"],
                    "description": parsed["description"],
                }
                return filtered, parsed["description"]
    except Exception:
        pass

    return df, None


# ═════════════════════════════════════════════════════════════════════════════
# AUTO ML PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def run_automl(df: pd.DataFrame, target_col: str, task: str) -> tuple:
    CHART_W, CHART_H = 5.5, 3.5
    work_df = df.copy().dropna(subset=[target_col])
    _target_col_name = target_col  # preserve original name for report

    drop_cols = []
    for c in work_df.columns:
        if c == target_col:
            continue
        if work_df[c].isna().mean() > 0.5:
            drop_cols.append(c)
        if work_df[c].nunique() == len(work_df):
            drop_cols.append(c)
    work_df.drop(columns=list(set(drop_cols)), inplace=True, errors="ignore")

    for c in work_df.select_dtypes(exclude="number").columns:
        if c == target_col:
            continue
        le = LabelEncoder()
        work_df[c] = le.fit_transform(work_df[c].astype(str))

    if task == "classification":
        le_t = LabelEncoder()
        y = le_t.fit_transform(work_df[target_col].astype(str))
    else:
        y = work_df[target_col].values

    X = work_df.drop(columns=[target_col]).select_dtypes(include="number").fillna(0)

    if X.shape[1] == 0:
        return None, None, "No usable feature columns found after preprocessing."

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler   = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_test  = scaler.transform(X_test)

    if task == "classification":
        models = {
            "Logistic Regression": (LogisticRegression(max_iter=500, random_state=42), True),
            "Random Forest":       (RandomForestClassifier(n_estimators=100, random_state=42), False),
            "Gradient Boosting":   (GradientBoostingClassifier(n_estimators=100, random_state=42), False),
        }
    else:
        models = {
            "Linear Regression": (LinearRegression(), True),
            "Ridge Regression":  (Ridge(alpha=1.0), True),
            "Random Forest":     (RandomForestRegressor(n_estimators=100, random_state=42), False),
            "Gradient Boosting": (GradientBoostingRegressor(n_estimators=100, random_state=42), False),
        }

    results    = []
    best_model = None
    best_score = -np.inf
    best_name  = ""

    for name, (model, use_scaled) in models.items():
        try:
            Xtr = Xs_train if use_scaled else X_train.values
            Xte = Xs_test  if use_scaled else X_test.values
            model.fit(Xtr, y_train)
            preds = model.predict(Xte)

            if task == "classification":
                acc  = accuracy_score(y_test, preds)
                f1   = f1_score(y_test, preds, average="weighted", zero_division=0)
                results.append({"Model": name, "Accuracy": round(acc, 4), "F1 Score": round(f1, 4)})
                score = f1
            else:
                rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
                mae  = float(mean_absolute_error(y_test, preds))
                r2   = float(r2_score(y_test, preds))
                results.append({"Model": name, "RMSE": round(rmse, 4),
                                 "MAE": round(mae, 4), "R²": round(r2, 4)})
                score = r2

            if score > best_score:
                best_score = score
                best_model = model
                best_name  = name
        except Exception as e:
            results.append({"Model": name, "Error": str(e)})

    results_df = pd.DataFrame(results)

    fi_bytes = None
    if best_model and hasattr(best_model, "feature_importances_"):
        fi = pd.Series(best_model.feature_importances_, index=X.columns).sort_values()
        fi = fi.tail(15)
        fig, ax = plt.subplots(figsize=(CHART_W, max(CHART_H, len(fi) * 0.3)))
        fi.plot(kind="barh", ax=ax, color="#7c3aed", edgecolor="#5a189a")
        ax.set_title(f"Feature Importance — {best_name}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Importance", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_facecolor("#f9f8f6")
        fig.patch.set_facecolor("#f9f8f6")
        plt.tight_layout()
        fi_bytes = fig_to_bytes(fig)

    metric_label = (f"F1 = {best_score:.4f}" if task == "classification"
                    else f"R² = {best_score:.4f}")

    # Build full structured report — no truncation
    all_features = list(X.columns)
    feat_str = ", ".join(f"<code>{f}</code>" for f in all_features)

    report = {
        "best_model":   best_name,
        "metric_label": metric_label,
        "task":         task,
        "train_rows":   X_train.shape[0],
        "test_rows":    X_test.shape[0],
        "n_features":   len(all_features),
        "features":     feat_str,
        "best_score":   best_score,
        "target_col":   _target_col_name,
    }

    return results_df, fi_bytes, report


# ═════════════════════════════════════════════════════════════════════════════
# LLM PLANNER  — Structured, grounded response plan
# ═════════════════════════════════════════════════════════════════════════════
def get_plan(user_query: str, df: pd.DataFrame) -> dict:
    schema = {
        "columns":             list(df.columns),
        "dtypes":              df.dtypes.astype(str).to_dict(),
        "shape":               list(df.shape),
        "numeric_columns":     list(df.select_dtypes(include="number").columns),
        "categorical_columns": list(df.select_dtypes(exclude="number").columns),
        "sample":              df.head(2).to_dict(orient="records"),
    }

    context_str, _ = retrieve_context(user_query, k=8)

    valid_analyses = (
        "describe | head | tail | missing | correlation | histogram | "
        "boxplot | scatter | value_counts | variance_max | skew_max | "
        "text | not_useful_column | outliers | datatype_info | unique_values | "
        "insights | categorical_summary | automl"
    )

    system_msg = (
        "You are an expert data analyst assistant. "
        "Your job is to plan the correct analysis type and write a clear, accurate, factual explanation. "
        "IMPORTANT RULES for the explanation field:\n"
        "1. Write 2-4 complete, meaningful sentences.\n"
        "2. Reference ACTUAL column names and REAL statistics from the retrieved context.\n"
        "3. Do NOT say vague things like 'the data shows interesting patterns'.\n"
        "4. Do NOT repeat the question back to the user.\n"
        "5. Include actionable insight: what does this mean? what should the user do?\n"
        "6. If referencing numbers, pull them from the retrieved context — never invent values.\n"
        "7. The 'answer' field should be a SHORT factual headline (one sentence, under 15 words).\n"
        "8. The 'explanation' should expand on the answer with data-backed reasoning.\n"
        "Use the retrieved context and conversation history to answer accurately. "
        "For AutoML queries (train, predict, classify, regress, model), use analysis=automl. "
        "Always reference column names exactly as they appear in the schema."
    )

    user_msg = f"""RETRIEVED CONTEXT (semantic search over the dataset):
{context_str}

DATASET SCHEMA:
{json.dumps(schema, indent=2)}

USER QUESTION: "{user_query}"

RULES:
- Use ONLY column names from the schema.
- Choose the most appropriate analysis type.
- If the user asks for a chart/plot/graph → histogram, boxplot, scatter, or correlation.
- If the user asks about categories/groups → value_counts.
- If the user asks to train a model / predict / classify / regress → automl.
- For automl: set target_col to the column to predict, task to 'classification' or 'regression'.
- Do NOT make up column names.
- Use conversation history for follow-up questions.
- The 'explanation' must be specific and data-backed, NOT generic filler text.

Respond with ONLY valid JSON — no markdown, no backticks:
{{
  "analysis": "<one of: {valid_analyses}>",
  "columns": ["col1", "col2"],
  "answer": "<short direct factual headline — one sentence>",
  "explanation": "<2-4 specific, data-backed sentences that give real insight>",
  "target_col": "<for automl only>",
  "task": "<classification or regression — for automl only>"
}}"""

    messages = [{"role": "system", "content": system_msg}]
    messages += get_memory_messages()
    messages.append({"role": "user", "content": user_msg})

    content = ""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
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
        "analysis":    "text",
        "columns":     [],
        "answer":      "Could not parse AI response. Please rephrase.",
        "explanation": "",
    }


# ═════════════════════════════════════════════════════════════════════════════
# CHART CONFIG
# ═════════════════════════════════════════════════════════════════════════════
CHART_W, CHART_H = 5.5, 3.5
CHART_BG  = "#f9f8f6"
CHART_CLR = ["#7c3aed", "#3b82f6", "#e63946", "#f59e0b", "#10b981"]

def apply_chart_style(ax, title):
    ax.set_facecolor(CHART_BG)
    ax.get_figure().patch.set_facecolor(CHART_BG)
    ax.set_title(title, fontsize=10, fontweight="bold", color="#1a1a2e", pad=10)
    ax.tick_params(labelsize=7, colors="#4a4a6a")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d1d5db")
    ax.spines["bottom"].set_color("#d1d5db")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color("#4a4a6a")


# ═════════════════════════════════════════════════════════════════════════════
# DIRECT INSIGHT GENERATORS
# ═════════════════════════════════════════════════════════════════════════════
def compute_insights(df: pd.DataFrame, n: int = 6) -> list:
    insights   = []
    numeric_df = df.select_dtypes(include="number")

    if not numeric_df.empty:
        skew_col  = numeric_df.skew().abs().idxmax()
        skew_val  = numeric_df.skew().abs().max()
        direction = "positively" if numeric_df[skew_col].skew() > 0 else "negatively"
        insights.append(
            f"**{skew_col}** is {direction} skewed (skew = {skew_val:.2f}). "
            f"The distribution has a long {'right' if direction == 'positively' else 'left'} tail — "
            f"consider a log transformation before modeling."
        )
        var_col = numeric_df.var().idxmax()
        insights.append(
            f"**{var_col}** has the highest variance ({numeric_df.var().max():.2f}), "
            f"meaning its values spread the widest. This column carries the most information "
            f"and should be included in any predictive model."
        )

    miss = df.isna().mean() * 100
    if miss.max() > 0:
        mc = miss.idxmax()
        action = "drop this column" if miss.max() > 50 else "impute using mean or median"
        insights.append(
            f"**{mc}** has {miss.max():.1f}% missing values — the highest in the dataset. "
            f"It's recommended to {action} before training any model."
        )

    useless = [c for c in numeric_df.columns if numeric_df[c].nunique() == 1]
    if useless:
        insights.append(
            f"**{', '.join(useless)}** {'has' if len(useless)==1 else 'have'} zero variance "
            f"(all values are identical). These columns provide no predictive signal and "
            f"should be removed before modeling."
        )

    if not numeric_df.empty:
        out_counts = {}
        for col in numeric_df.columns:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR    = Q3 - Q1
            n_out  = int(((df[col] < Q1-1.5*IQR) | (df[col] > Q3+1.5*IQR)).sum())
            if n_out > 0:
                out_counts[col] = n_out
        if out_counts:
            tc = max(out_counts, key=out_counts.get)
            insights.append(
                f"**{tc}** contains {out_counts[tc]} outliers (IQR method). "
                f"These extreme values may distort model training — consider capping or "
                f"removing them before fitting a regression or distance-based model."
            )

    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr().abs()
        np.fill_diagonal(corr.values, 0)
        if corr.max().max() > 0.5:
            idx = corr.stack().idxmax()
            rv  = numeric_df.corr().loc[idx[0], idx[1]]
            strength = "strong" if abs(rv) > 0.7 else "moderate"
            insights.append(
                f"**{idx[0]}** & **{idx[1]}** have a {strength} {'positive' if rv > 0 else 'negative'} "
                f"correlation (r = {rv:.2f}). Including both as features may introduce multicollinearity "
                f"— consider dropping one or using PCA."
            )

    insights.append(
        f"The dataset has **{df.shape[0]:,} rows** and **{df.shape[1]} columns** — "
        f"{'a reasonable sample size for most ML models' if df.shape[0] > 500 else 'a small dataset; model performance may be limited'}. "
        f"Missing values total: **{int(df.isna().sum().sum())}**."
    )
    return insights[:n]


def compute_categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.select_dtypes(exclude="number").columns:
        n_unique = df[col].nunique()
        pct_miss = round(df[col].isna().mean() * 100, 1)
        top_val  = df[col].value_counts().index[0] if n_unique > 0 else "N/A"
        top_pct  = round(df[col].value_counts().iloc[0] / len(df) * 100, 1) if n_unique > 0 else 0
        note = ("⚠️ ID-like"           if n_unique == len(df) else
                "⚠️ Constant"          if n_unique == 1 else
                "⚠️ High missing"      if pct_miss > 50 else
                "✅ Good for encoding" if n_unique <= 15 else
                "⚠️ High cardinality")
        rows.append({"Column": col, "Unique": n_unique, "Missing %": pct_miss,
                     "Top Value": str(top_val)[:30], "Top %": top_pct, "Note": note})
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# EXECUTE PLAN
# ═════════════════════════════════════════════════════════════════════════════
def execute_plan(plan: dict, df: pd.DataFrame):
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

    if analysis == "insights":
        return ("__insights__", compute_insights(df, n=6)), "Key dataset insights", ""

    if analysis == "categorical_summary":
        sdf = compute_categorical_summary(df)
        return (sdf if not sdf.empty else None), "Categorical column summary", (
            "✅ = suitable for encoding. ⚠️ ID-like = drop before modeling. "
            "⚠️ High missing = impute. ⚠️ High cardinality = consider target encoding."
        )

    if analysis == "automl":
        target_col = plan.get("target_col", "")
        task       = plan.get("task", "regression")
        if not target_col or target_col not in df.columns:
            target_col = df.columns[-1]
        results_df, fi_bytes, report = run_automl(df, target_col, task)
        return ("__automl__", results_df, fi_bytes, report), \
               f"AutoML complete — predicting '{target_col}'", ""

    if analysis == "describe":
        return df.describe().round(4), answer or "Statistical summary of all numeric columns", expl

    if analysis == "head":
        return df.head(10), answer or "First 10 rows", expl

    if analysis == "tail":
        return df.tail(10), answer or "Last 10 rows", expl

    if analysis == "missing":
        m = df.isna().sum().to_frame("Missing Count")
        m["% Missing"] = (df.isna().mean() * 100).round(2)
        m = m[m["Missing Count"] > 0].sort_values("Missing Count", ascending=False)
        if m.empty:
            return None, "No missing values found in this dataset ✅", ""
        return m, answer or f"{int(m['Missing Count'].sum())} total missing values across {len(m)} columns", expl

    if analysis == "datatype_info":
        info = df.dtypes.to_frame("Data Type")
        info["Non-Null Count"] = df.notnull().sum()
        info["Unique Values"]  = df.nunique()
        info["% Missing"]      = (df.isna().mean() * 100).round(2)
        return info, answer or "Column data types and completeness", expl

    if analysis == "unique_values":
        col    = pick_col(cols, list(df.columns))
        count  = df[col].nunique() if col else 0
        sample = df[col].dropna().unique()[:10].tolist() if col else []
        return None, f"'{col}' has {count} unique values", f"Sample values: {sample}"

    if analysis == "not_useful_column":
        reasons = {}
        for c in numeric_df.columns:
            if numeric_df[c].nunique() == 1:
                reasons[c] = "Constant — zero variance"
        for c in df.isna().mean()[df.isna().mean() > 0.5].index:
            reasons[c] = f"{df[c].isna().mean()*100:.1f}% missing"
        for c in df.columns:
            if df[c].nunique() == len(df) and c not in reasons:
                reasons[c] = "All values unique — likely an ID column"
        if reasons:
            return pd.DataFrame(list(reasons.items()), columns=["Column", "Reason"]), \
                   f"Found {len(reasons)} potentially useless columns", expl
        return None, "All columns appear useful — no constants, IDs, or high-missing columns detected", ""

    # ── Charts ────────────────────────────────────────────────
    if analysis == "correlation":
        if numeric_df.shape[1] < 2:
            return None, "Need ≥ 2 numeric columns for a correlation heatmap", ""
        fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
        ax.set_facecolor(CHART_BG)
        fig.patch.set_facecolor(CHART_BG)
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="RdYlBu_r",
                    linewidths=0.5, annot_kws={"size": 7}, ax=ax,
                    cbar_kws={"shrink": 0.8})
        ax.set_title("Correlation Heatmap", fontsize=10, fontweight="bold", color="#1a1a2e", pad=10)
        ax.tick_params(labelsize=7, colors="#4a4a6a")
        plt.tight_layout()
        return fig_to_bytes(fig), answer or "Correlation heatmap", expl

    if analysis == "histogram":
        col = pick_col(cols, list(numeric_df.columns))
        if not col:
            return None, "No numeric column found", ""
        fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color=CHART_CLR[0],
                     edgecolor="white", linewidth=0.5)
        ax.lines[0].set_color(CHART_CLR[1]) if ax.lines else None
        apply_chart_style(ax, f"Distribution of {col}")
        ax.set_xlabel(col, fontsize=8, color="#4a4a6a")
        ax.set_ylabel("Frequency", fontsize=8, color="#4a4a6a")
        plt.tight_layout()
        return fig_to_bytes(fig), answer or f"Distribution of {col}", expl

    if analysis == "boxplot":
        col = pick_col(cols, list(numeric_df.columns))
        if not col:
            return None, "No numeric column found", ""
        fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
        sns.boxplot(y=df[col].dropna(), ax=ax, color=CHART_CLR[0],
                    width=0.35, linewidth=1.2,
                    medianprops=dict(color="#e63946", linewidth=2))
        apply_chart_style(ax, f"Boxplot: {col}")
        ax.set_ylabel(col, fontsize=8, color="#4a4a6a")
        plt.tight_layout()
        return fig_to_bytes(fig), answer or f"Boxplot of {col}", expl

    if analysis == "scatter":
        valid = [c for c in cols if c in numeric_df.columns]
        if len(valid) < 2:
            valid = list(numeric_df.columns[:2])
        if len(valid) < 2:
            return None, "Need ≥ 2 numeric columns for a scatter plot", ""
        fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
        ax.scatter(df[valid[0]], df[valid[1]],
                   alpha=0.4, color=CHART_CLR[0], s=12, edgecolors="none")
        apply_chart_style(ax, f"{valid[0]} vs {valid[1]}")
        ax.set_xlabel(valid[0], fontsize=8, color="#4a4a6a")
        ax.set_ylabel(valid[1], fontsize=8, color="#4a4a6a")
        plt.tight_layout()
        return fig_to_bytes(fig), answer or f"Scatter: {valid[0]} vs {valid[1]}", expl

    if analysis == "value_counts":
        col = pick_col(cols, list(df.columns))
        if not col:
            return None, "No column found", ""
        if df[col].nunique() > 50:
            better = [c for c in df.select_dtypes(exclude="number").columns
                      if df[c].nunique() <= 50]
            if better:
                col = better[0]
        vc  = df[col].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
        bars = ax.bar(range(len(vc)), vc.values, color=CHART_CLR[0],
                      edgecolor="white", linewidth=0.5, width=0.65)
        ax.set_xticks(range(len(vc)))
        ax.set_xticklabels(vc.index, rotation=40, ha="right", fontsize=7)
        apply_chart_style(ax, f"Value Counts: {col}")
        ax.set_ylabel("Count", fontsize=8, color="#4a4a6a")
        plt.tight_layout()
        return fig_to_bytes(fig), answer or f"Top categories in '{col}'", expl

    if analysis == "variance_max":
        var_s    = numeric_df.var().sort_values(ascending=False)
        col, val = var_s.idxmax(), var_s.max()
        fig, ax  = plt.subplots(figsize=(CHART_W, CHART_H))
        colors = [CHART_CLR[0] if c == col else "#d1d5db" for c in var_s.index]
        ax.bar(range(len(var_s)), var_s.values, color=colors, edgecolor="white", width=0.65)
        ax.set_xticks(range(len(var_s)))
        ax.set_xticklabels(var_s.index, rotation=40, ha="right", fontsize=7)
        apply_chart_style(ax, "Feature Variances — Highlighted: Highest")
        ax.set_ylabel("Variance", fontsize=8, color="#4a4a6a")
        plt.tight_layout()
        return fig_to_bytes(fig), f"Highest variance: '{col}' ({val:.2f})", \
               expl or f"'{col}' has the widest spread of values (variance = {val:.2f}). High variance means this column contains the most variability and is likely an important feature for modeling."

    if analysis == "skew_max":
        skew_s   = numeric_df.skew().abs().sort_values(ascending=False)
        col, val = skew_s.idxmax(), skew_s.max()
        direction = "right (positive)" if numeric_df[col].skew() > 0 else "left (negative)"
        fig, ax  = plt.subplots(figsize=(CHART_W, CHART_H))
        colors = [CHART_CLR[2] if c == col else "#d1d5db" for c in skew_s.index]
        ax.bar(range(len(skew_s)), skew_s.values, color=colors, edgecolor="white", width=0.65)
        ax.set_xticks(range(len(skew_s)))
        ax.set_xticklabels(skew_s.index, rotation=40, ha="right", fontsize=7)
        apply_chart_style(ax, "Feature Skewness — Highlighted: Most Skewed")
        ax.set_ylabel("Absolute Skewness", fontsize=8, color="#4a4a6a")
        plt.tight_layout()
        return fig_to_bytes(fig), f"Most skewed: '{col}' (skew = {val:.2f})", \
               expl or f"'{col}' is skewed to the {direction} with a skewness of {val:.2f}. Values above 1.0 are considered highly skewed — a log or square-root transformation is recommended before using this column in linear models."

    if analysis == "outliers":
        col = pick_col(cols, list(numeric_df.columns))
        if not col:
            return None, "No numeric column found", ""
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR    = Q3 - Q1
        lo, hi = Q1-1.5*IQR, Q3+1.5*IQR
        n_out  = int(((df[col] < lo) | (df[col] > hi)).sum())
        fig, axes = plt.subplots(1, 2, figsize=(CHART_W*1.6, CHART_H))
        fig.patch.set_facecolor(CHART_BG)
        sns.boxplot(y=df[col], ax=axes[0], color=CHART_CLR[0],
                    width=0.35, linewidth=1.2,
                    medianprops=dict(color="#e63946", linewidth=2))
        apply_chart_style(axes[0], f"Boxplot: {col}")
        sns.histplot(df[col], kde=True, ax=axes[1], color=CHART_CLR[0],
                     edgecolor="white", linewidth=0.5, bins=30)
        axes[1].axvline(lo, color=CHART_CLR[2], linestyle="--", linewidth=1.2,
                        label=f"Lower fence: {lo:.1f}")
        axes[1].axvline(hi, color=CHART_CLR[2], linestyle="--", linewidth=1.2,
                        label=f"Upper fence: {hi:.1f}")
        axes[1].legend(fontsize=7)
        apply_chart_style(axes[1], f"Distribution: {col}")
        plt.tight_layout()
        return fig_to_bytes(fig), f"{n_out} outliers detected in '{col}'", \
               expl or (f"Using the IQR method: values below {lo:.2f} or above {hi:.2f} are flagged as outliers. "
                        f"{n_out} such values exist in '{col}'. "
                        f"These could be data entry errors or genuine extremes — inspect them before deciding to remove or cap.")

    return None, answer or "Analysis complete", expl or "No further details available."


def safe_execute(plan, df):
    try:
        if not isinstance(plan, dict) or "analysis" not in plan:
            return None, "Invalid plan from AI", "Try rephrasing your question."
        return execute_plan(plan, df)
    except Exception as e:
        return None, "Execution error", str(e)


# ═════════════════════════════════════════════════════════════════════════════
# RENDER & DISPLAY  — polished formatting
# ═════════════════════════════════════════════════════════════════════════════
def format_explanation_html(explanation: str) -> str:
    """
    Converts a raw explanation string into a clean, readable HTML block.
    Splits on sentences and renders as a bullet list when multiple points exist.
    """
    if not explanation:
        return ""

    # Split on sentence boundaries
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", explanation) if s.strip()]

    if len(sentences) == 1:
        # Single sentence — just render as paragraph
        return f'<p style="margin:0">{sentences[0]}</p>'

    # Multiple sentences — render as bullet list
    items = "".join(f"<li>{s}</li>" for s in sentences)
    return f"<ul style='margin:6px 0 0; padding-left:18px'>{items}</ul>"


def _render_automl_report(r: dict) -> str:
    """Renders the AutoML report dict as clean structured HTML."""
    score_color = "#10b981" if r["best_score"] > 0.75 else ("#f59e0b" if r["best_score"] > 0.5 else "#e63946")
    task_label  = "Classification" if r["task"] == "classification" else "Regression"
    metric_name = "F1 Score" if r["task"] == "classification" else "R²"
    return f"""
<div class="ml-box">
  <div class="ml-title">🤖 AutoML Report — {task_label}</div>
  <table style="width:100%;border-collapse:collapse;font-size:0.85rem;margin-bottom:10px">
    <tr>
      <td style="padding:4px 10px 4px 0;color:#64748b;width:40%">🏆 Best Model</td>
      <td style="padding:4px 0;font-weight:600;color:#1e3a5f">{r['best_model']}</td>
    </tr>
    <tr>
      <td style="padding:4px 10px 4px 0;color:#64748b">{metric_name}</td>
      <td style="padding:4px 0;font-weight:700;color:{score_color}">{r['best_score']:.4f}</td>
    </tr>
    <tr>
      <td style="padding:4px 10px 4px 0;color:#64748b">🎯 Target Column</td>
      <td style="padding:4px 0;font-family:'DM Mono',monospace;font-size:0.82rem;color:#1e3a5f">{r['target_col']}</td>
    </tr>
    <tr>
      <td style="padding:4px 10px 4px 0;color:#64748b">📊 Train / Test Rows</td>
      <td style="padding:4px 0;color:#1e3a5f">{r['train_rows']:,} / {r['test_rows']:,}</td>
    </tr>
    <tr>
      <td style="padding:4px 10px 4px 0;color:#64748b">🔢 Features Used</td>
      <td style="padding:4px 0;color:#1e3a5f">{r['n_features']}</td>
    </tr>
  </table>
  <div style="font-size:0.78rem;color:#64748b;margin-bottom:4px;font-family:'DM Mono',monospace;text-transform:uppercase;letter-spacing:0.08em">Feature List</div>
  <div style="font-size:0.80rem;color:#1e3a5f;line-height:1.8;word-break:break-word">{r['features']}</div>
</div>"""


def render_message(msg):
    t = msg["type"]
    if t == "text":
        st.markdown(msg["content"], unsafe_allow_html=True)
    elif t == "dataframe":
        st.dataframe(msg["content"], use_container_width=False)
    elif t == "image":
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(msg["content"])
    elif t == "insights":
        icons = ["📊", "📈", "⚠️", "🔍", "🎯", "💾"]
        for i, item in enumerate(msg["content"]):
            icon = icons[i % len(icons)]
            st.markdown(
                f'<div class="insight-card"><span class="icon">{icon}</span><span>{item}</span></div>',
                unsafe_allow_html=True
            )
    elif t == "automl":
        results_df, fi_bytes, report = msg["content"]
        if results_df is not None:
            st.dataframe(results_df, use_container_width=False)
        if fi_bytes:
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.image(fi_bytes)
        if isinstance(report, dict):
            st.markdown(_render_automl_report(report), unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ml-box"><div class="ml-title">🤖 AutoML Report</div>{report}</div>',
                        unsafe_allow_html=True)


def display_result(result, answer, explanation):
    stored = []

    # Active filter banner
    if st.session_state.active_filter:
        st.markdown(
            f'<div class="filter-banner">🔍 <strong>Active filter:</strong> '
            f'{st.session_state.active_filter["description"]}</div>',
            unsafe_allow_html=True,
        )

    # Answer headline
    if answer:
        st.markdown(
            f'<div class="answer-box">'
            f'<div class="label">Answer</div>'
            f'{answer}'
            f'</div>',
            unsafe_allow_html=True
        )
        stored.append({"role": "assistant", "type": "text",
                        "content": f"**{answer}**"})

    # Insights
    if isinstance(result, tuple) and result[0] == "__insights__":
        icons = ["📊", "📈", "⚠️", "🔍", "🎯", "💾"]
        st.markdown('<div class="insight-section-title">Dataset Insights</div>', unsafe_allow_html=True)
        for i, item in enumerate(result[1]):
            icon = icons[i % len(icons)]
            st.markdown(
                f'<div class="insight-card"><span class="icon">{icon}</span><span>{item}</span></div>',
                unsafe_allow_html=True
            )
        stored.append({"role": "assistant", "type": "insights", "content": result[1]})
        return stored

    # AutoML
    if isinstance(result, tuple) and result[0] == "__automl__":
        _, results_df, fi_bytes, report = result
        if results_df is not None:
            st.dataframe(results_df, use_container_width=False)
            csv_bytes = results_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download AutoML results as CSV",
                                csv_bytes, "automl_results.csv", "text/csv",
                                key="dl_automl_csv")
        if fi_bytes:
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.image(fi_bytes)
            st.download_button("⬇️ Download feature importance chart",
                                fi_bytes, "feature_importance.png", "image/png",
                                key="dl_fi_png")
        st.markdown(_render_automl_report(report), unsafe_allow_html=True)
        stored.append({"role": "assistant", "type": "automl",
                        "content": (results_df, fi_bytes, report)})
        return stored

    # DataFrame
    if isinstance(result, (pd.DataFrame, pd.Series)):
        st.dataframe(result, use_container_width=False)
        csv_bytes = result.to_csv(index=True).encode()
        st.download_button("⬇️ Download table as CSV",
                            csv_bytes, "table.csv", "text/csv",
                            key="dl_table_csv")
        stored.append({"role": "assistant", "type": "dataframe", "content": result})

    # Chart
    elif isinstance(result, bytes):
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(result)
        st.download_button("⬇️ Download chart as PNG",
                            result, "chart.png", "image/png",
                            key="dl_chart_png")
        stored.append({"role": "assistant", "type": "image", "content": result})

    # Explanation — rendered cleanly
    if explanation:
        formatted_html = format_explanation_html(explanation)
        st.markdown(
            f'<div class="explanation-box">'
            f'<div class="expl-title">🧠 Explanation</div>'
            f'{formatted_html}'
            f'</div>',
            unsafe_allow_html=True,
        )
        stored.append({"role": "assistant", "type": "text",
                        "content": f"**🧠 Explanation**\n{explanation}"})

    return stored


# ═════════════════════════════════════════════════════════════════════════════
# SMART DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("## 🧠 AI Smart Dashboard")

if st.button("🚀 Generate Smart Dashboard"):
    st.session_state.dashboard_on = True

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df_hash = str(df.shape) + str(df.columns.tolist()) + str(df.head().to_json())
    if st.session_state.df_hash != df_hash:
        st.session_state.dashboard_on  = False
        st.session_state.active_filter = None
        st.session_state.chat_history  = []
        with st.spinner("🔍 Building semantic search index..."):
            st.session_state.rag_index = build_rag_index(df)
            st.session_state.df_hash   = df_hash

    # ── Sidebar ─────────────────────────────────────────────
    st.sidebar.markdown("### Dataset Summary")
    metrics = [
        ("📏 Rows",         df.shape[0]),
        ("📐 Columns",      df.shape[1]),
        ("❓ Missing",      int(df.isnull().sum().sum())),
        ("🔢 Numeric",      df.select_dtypes(include="number").shape[1]),
        ("🔤 Categorical",  df.select_dtypes(exclude="number").shape[1]),
    ]
    for label, val in metrics:
        st.sidebar.markdown(
            f'<div class="sidebar-metric"><span>{label}</span>'
            f'<span class="sidebar-metric-val">{val:,}</span></div>',
            unsafe_allow_html=True
        )

    if st.session_state.rag_index:
        n_docs = len(st.session_state.rag_index[2])
        st.sidebar.markdown(
            f'<div class="sidebar-metric"><span>🗂 RAG Docs</span>'
            f'<span class="sidebar-metric-val">{n_docs}</span></div>',
            unsafe_allow_html=True
        )

    if st.session_state.active_filter:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**🔍 Filter active**")
        st.sidebar.caption(st.session_state.active_filter["description"])
        if st.sidebar.button("❌ Clear filter"):
            st.session_state.active_filter = None
            st.rerun()

    mem_len = len(st.session_state.chat_history) // 2
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f'<div class="sidebar-metric"><span>💬 Memory</span>'
        f'<span class="sidebar-metric-val">{mem_len} exchanges</span></div>',
        unsafe_allow_html=True
    )

    with st.expander("🔍 Dataset Preview"):
        st.dataframe(df.head(), use_container_width=True)

    st.markdown("---")

    # ── DASHBOARD ──────────────────────────────────────────
    if st.session_state.dashboard_on:
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{df.shape[0]:,}")
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", int(df.isnull().sum().sum()))

        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if numeric_cols:
            selected_col = st.selectbox("📊 Select Column for Analysis",
                                        numeric_cols, key="selected_column")

            fig = px.histogram(df, x=selected_col, nbins=30,
                               title=f"Distribution of {selected_col}",
                               color_discrete_sequence=["#7c3aed"])
            fig.update_layout(height=350, margin=dict(l=20,r=20,t=40,b=20),
                               plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
                               font=dict(family="DM Sans"))
            fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
            fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb")
            fig.update_traces(marker_line_color="white", marker_line_width=1, opacity=0.9)
            st.plotly_chart(fig, use_container_width=True, key=f"hist_{selected_col}")

            fig2 = px.box(df, y=selected_col,
                           title=f"Outlier Detection: {selected_col}",
                           color_discrete_sequence=["#3b82f6"])
            fig2.update_layout(height=350, margin=dict(l=20,r=20,t=40,b=20),
                                plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG,
                                font=dict(family="DM Sans"))
            fig2.update_yaxes(showgrid=True, gridcolor="#e5e7eb")
            st.plotly_chart(fig2, use_container_width=True, key=f"box_{selected_col}")

            st.markdown("### 🧠 AI Insights")
            query   = f"Analyze column {selected_col} and give insights"
            context, docs = retrieve_context(query)
            insight_prompt = f"""You are a data analyst. Column being analyzed: '{selected_col}'.

Retrieved context about this column:
{context}

Write exactly 3 concise, specific, actionable insights about this column.
Each insight should be one sentence. Reference actual statistics from the context.
Do NOT write generic statements like "this data shows patterns".
Format: plain numbered list. No markdown headers."""

            with st.spinner("🤖 Generating insights..."):
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": insight_prompt}],
                    temperature=0,
                )
            insights_text = response.choices[0].message.content
            # Render each line as an insight card
            lines = [l.strip() for l in insights_text.split("\n") if l.strip()]
            for line in lines[:4]:
                # Strip leading numbers like "1. "
                clean = re.sub(r"^\d+[\.\)]\s*", "", line)
                st.markdown(
                    f'<div class="insight-card"><span class="icon">💡</span><span>{clean}</span></div>',
                    unsafe_allow_html=True
                )
        else:
            st.warning("No numeric columns available for visualization.")

    # ── CHAT ──────────────────────────────────────────────
    st.markdown("## 💬 Ask Your Data")
    st.caption("Ask about distributions, correlations, outliers, trends, or train a model — in plain English.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            render_message(msg)

    user_query = st.chat_input("e.g. 'Show distribution of revenue' · 'Which column has most outliers?' · 'Train a classifier on target'")

    if user_query:
        st.session_state.messages.append(
            {"role": "user", "type": "text", "content": user_query}
        )
        update_memory("user", user_query)

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analysing... 🤖"):

                # Step 1: NL filter detection
                df_working, filter_desc = detect_and_apply_filter(user_query, df)

                # Step 2: plan
                plan = get_plan(user_query, df_working)

                # Step 3: execute
                result, answer, explanation = safe_execute(plan, df_working)

                # Step 4: display
                new_msgs = display_result(result, answer, explanation)
                st.session_state.messages.extend(new_msgs)

                # Update memory with a clean compact summary (not raw dump)
                update_memory("assistant", build_memory_summary(answer, explanation))

else:
    if st.session_state.dashboard_on:
        st.warning("⚠️ Upload a dataset first to use the dashboard.")
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #6b7280;">
        <div style="font-size: 3rem; margin-bottom: 16px;">📂</div>
        <div style="font-size: 1.1rem; font-weight: 600; color: #1a1a2e; margin-bottom: 8px;">
            No dataset loaded
        </div>
        <div style="font-size: 0.9rem;">
            Upload a CSV file from the sidebar to get started
        </div>
    </div>
    """, unsafe_allow_html=True)

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
# STATIC KNOWLEDGE BASE  — expanded to 40 docs for richer retrieval
# ═════════════════════════════════════════════════════════════════════════════
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
    "Bar charts compare values across categories and are useful for grouped or aggregated data.",
    "Line charts show trends over time or an ordered period column.",
    "Grouped bar charts compare a numeric metric across two categorical dimensions simultaneously.",
    "Pie charts show the proportion of each category, best for fewer than 6 categories.",
    # Cleaning
    "Missing values can be handled using imputation with mean, median, or mode, or by removal.",
    "Categorical variables often require encoding such as one-hot encoding or label encoding.",
    "Normalization scales values between 0 and 1, useful for distance-based models and neural networks.",
    "Standardization centers data around mean 0 and standard deviation 1, useful for linear models.",
    "Dropping columns with too many missing values can improve model performance.",
    "Constant columns with zero variance should be removed as they provide no useful information.",
    "ID-like columns where every value is unique are not useful features for modeling.",
    # Features
    "Feature importance identifies which variables contribute most to predictions.",
    "Highly correlated features can cause multicollinearity and should be handled carefully.",
    "Date or period features can be expanded into year, month, quarter for richer analysis.",
    "Groupby aggregations like mean, sum, count per category reveal useful patterns.",
    "Log transformation reduces right skewness and can improve model performance.",
    # ML
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
    "chat_history":  [],        # conversation memory — last 6-7 exchanges
    "active_filter": None,      # persisted NL filter  {query, description}
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ═════════════════════════════════════════════════════════════════════════════
# UI / STYLES  (unchanged from original)
# ═════════════════════════════════════════════════════════════════════════════
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
.filter-banner {
    background: #fff8e1;
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 8px 14px;
    margin: 4px 0;
    font-size: 0.88em;
}
.ml-box {
    background: #e8f4fd;
    border-left: 4px solid #2196f3;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 4px 0;
    font-size: 0.9em;
}
.insight-box {
    background: #f0fff4;
    border-left: 4px solid #38a169;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 4px 0;
    font-size: 0.95em;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("📊 Dataset Dashboard")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
st.title("AI AutoML Data Assistant 🤖")
st.markdown("### Your AI-powered Data Scientist")


# ═════════════════════════════════════════════════════════════════════════════
# ╔══════════════════════════════════════════════════════════════════╗
# ║           UPGRADED RAG PIPELINE                                  ║
# ║                                                                  ║
# ║  OLD: TF-IDF on 50 basic column-stat sentences                  ║
# ║  NEW: TF-IDF on 200+ rich documents including:                  ║
# ║    • per-column stats (mean, std, skew, kurtosis, min, max)      ║
# ║    • group-level aggregations (mean & sum per category)          ║
# ║    • time/period trend summaries                                 ║
# ║    • top/bottom value docs                                       ║
# ║    • correlation pair docs                                       ║
# ║    • outlier & missing summaries                                 ║
# ║  Vectorisation: bigrams + sublinear_tf + max_features=8000      ║
# ║  Retrieval:     cosine similarity, top-k=8                       ║
# ╚══════════════════════════════════════════════════════════════════╝

def build_dataset_documents(df: pd.DataFrame) -> list:
    """
    Converts the DataFrame into rich natural-language documents.
    200+ chunks covering column stats, group aggregations, trends,
    correlations, outliers, and missing values.
    """
    docs = []
    numeric_df = df.select_dtypes(include="number")
    cat_df     = df.select_dtypes(exclude="number")

    # ── 1. Per-column documents ───────────────────────────────
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

    # ── 2. Dataset-level summary ──────────────────────────────
    docs.append(
        f"Dataset overview: {df.shape[0]} rows, {df.shape[1]} columns. "
        f"Numeric columns: {list(numeric_df.columns)}. "
        f"Categorical columns: {list(cat_df.columns)}. "
        f"Total missing values: {int(df.isna().sum().sum())}."
    )

    # ── 3. Group-level aggregation docs ──────────────────────
    # Enables queries like "average Data_value per industry"
    for cat_col in cat_df.columns:
        if df[cat_col].nunique() > 30:
            continue   # skip ID-like
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

    # ── 4. Time/period trend docs ─────────────────────────────
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
                    f"Use a line chart to visualise {num_col} over {pc}. "
                    f"Relevant for: trend, over time, across periods, line chart."
                )
            except Exception:
                pass

    # ── 5. Top/bottom value docs ──────────────────────────────
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

    # ── 6. Correlation pair docs ──────────────────────────────
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

    # ── 7. Variance / skew summary ────────────────────────────
    if not numeric_df.empty:
        var_col  = numeric_df.var().idxmax()
        skew_col = numeric_df.skew().abs().idxmax()
        docs.append(
            f"'{var_col}' has the highest variance ({numeric_df.var().max():.2f}) — most spread out feature. "
            f"'{skew_col}' is the most skewed (skew={numeric_df.skew().abs().max():.2f}) — needs log transform."
        )

    # ── 8. Outlier summaries ──────────────────────────────────
    for col in numeric_df.columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR    = Q3 - Q1
        n_out  = int(((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum())
        if n_out > 0:
            docs.append(
                f"'{col}' has {n_out} outliers (IQR). "
                f"Lower={Q1 - 1.5*IQR:.2f}, upper={Q3 + 1.5*IQR:.2f}. "
                f"Relevant for outlier detection, anomaly, extreme values."
            )

    # ── 9. Missing value summary ──────────────────────────────
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
    """
    Builds TF-IDF index over all documents.
    bigrams + sublinear_tf + max_features=8000 for richer matching.
    Returns (vectorizer, tfidf_matrix, all_docs).
    """
    dataset_docs = build_dataset_documents(df)
    all_docs     = KNOWLEDGE_BASE + dataset_docs

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),      # unigrams + bigrams
        min_df=1,
        sublinear_tf=True,       # log-scale TF
        max_features=8000,       # NEW: cap vocab for speed
    )
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    return vectorizer, tfidf_matrix, all_docs


def retrieve_context(query: str, k: int = 8) -> tuple:
    """
    Retrieve top-k docs via TF-IDF cosine similarity.
    k=8 (was 6) for richer context.
    Returns (formatted_context_string, list_of_top_docs).
    """
    vectorizer, tfidf_matrix, all_docs = st.session_state.rag_index
    query_vec = vectorizer.transform([query])
    scores    = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx   = np.argsort(scores)[::-1][:k]
    top_docs  = [all_docs[i] for i in top_idx if scores[i] > 0.01]

    if not top_docs:
        top_docs = [all_docs[-1]]   # fallback: dataset summary

    return "\n".join(f"- {d}" for d in top_docs), top_docs


# ═════════════════════════════════════════════════════════════════════════════
# CONVERSATION MEMORY
# Stores last 6-7 user/assistant exchanges so follow-up questions work.
# e.g. "now filter that for Mining only" resolves correctly.
# ═════════════════════════════════════════════════════════════════════════════
MEMORY_LIMIT = 14   # 7 exchanges × 2 messages each

def update_memory(role: str, content: str):
    st.session_state.chat_history.append({"role": role, "content": content})
    if len(st.session_state.chat_history) > MEMORY_LIMIT:
        st.session_state.chat_history = st.session_state.chat_history[-MEMORY_LIMIT:]


def get_memory_messages() -> list:
    """Returns the last 6-7 exchanges formatted for the LLM messages array."""
    return [{"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_history]


# ═════════════════════════════════════════════════════════════════════════════
# NATURAL LANGUAGE FILTER
# Detects filter intent → asks LLM to generate df.query() string →
# applies it to df before any analysis. Filter persists across questions.
# ═════════════════════════════════════════════════════════════════════════════

def detect_and_apply_filter(user_query: str, df: pd.DataFrame) -> tuple:
    """
    Returns (working_df, filter_description | None).
    If no filter intent, returns original df (or re-applies existing filter).
    """
    filter_keywords = ["filter", "only", "where", "show only", "just",
                       "subset", "rows where", "limit to", "select rows"]

    has_intent = any(kw in user_query.lower() for kw in filter_keywords)

    if not has_intent:
        # Re-apply existing active filter if present
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
# Auto-selects between classification and regression.
# Trains 3-4 models, compares them, shows feature importance chart.
# ═════════════════════════════════════════════════════════════════════════════

def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def run_automl(df: pd.DataFrame, target_col: str, task: str) -> tuple:
    """
    Returns (results_df, feature_importance_bytes, report_text).
    task: 'classification' or 'regression'
    """
    CHART_W, CHART_H = 5.5, 3.5

    work_df = df.copy().dropna(subset=[target_col])

    # Drop high-missing and ID-like columns
    drop_cols = []
    for c in work_df.columns:
        if c == target_col:
            continue
        if work_df[c].isna().mean() > 0.5:
            drop_cols.append(c)
        if work_df[c].nunique() == len(work_df):
            drop_cols.append(c)
    work_df.drop(columns=list(set(drop_cols)), inplace=True, errors="ignore")

    # Encode categorical features
    for c in work_df.select_dtypes(exclude="number").columns:
        if c == target_col:
            continue
        le = LabelEncoder()
        work_df[c] = le.fit_transform(work_df[c].astype(str))

    # Encode target if classification
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

    # Feature importance chart for best tree-based model
    fi_bytes = None
    if best_model and hasattr(best_model, "feature_importances_"):
        fi = pd.Series(best_model.feature_importances_, index=X.columns).sort_values()
        fi = fi.tail(15)
        fig, ax = plt.subplots(figsize=(CHART_W, max(CHART_H, len(fi) * 0.3)))
        fi.plot(kind="barh", ax=ax, color="#7c6fcd", edgecolor="#5a4fcf")
        ax.set_title(f"Feature Importance — {best_name}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Importance", fontsize=8)
        ax.tick_params(labelsize=7)
        plt.tight_layout()
        fi_bytes = fig_to_bytes(fig)

    metric_label = (f"F1={best_score:.4f}" if task == "classification"
                    else f"R²={best_score:.4f}")
    report = (f"Best model: **{best_name}** ({metric_label}). "
              f"Trained on {X_train.shape[0]} rows, tested on {X_test.shape[0]} rows. "
              f"Features used: {list(X.columns[:8])}{'...' if len(X.columns) > 8 else ''}.")

    return results_df, fi_bytes, report


# ═════════════════════════════════════════════════════════════════════════════
# LLM PLANNER  — now includes conversation memory in the messages array
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

Respond with ONLY valid JSON — no markdown, no backticks:
{{
  "analysis": "<one of: {valid_analyses}>",
  "columns": ["col1", "col2"],
  "answer": "<short direct answer or empty string>",
  "explanation": "<2-3 sentences using the retrieved context>",
  "target_col": "<for automl only>",
  "task": "<classification or regression — for automl only>"
}}"""

    # Build messages array with system + memory + current user message
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


# ═════════════════════════════════════════════════════════════════════════════
# DIRECT INSIGHT GENERATORS  (zero LLM hallucination)
# ═════════════════════════════════════════════════════════════════════════════
def compute_insights(df: pd.DataFrame, n: int = 6) -> list:
    insights   = []
    numeric_df = df.select_dtypes(include="number")

    if not numeric_df.empty:
        skew_col  = numeric_df.skew().abs().idxmax()
        skew_val  = numeric_df.skew().abs().max()
        direction = "positively" if numeric_df[skew_col].skew() > 0 else "negatively"
        insights.append(
            f"**{skew_col}** is {direction} skewed (skew={skew_val:.2f}) "
            f"→ not normal, consider log transformation"
        )
        var_col = numeric_df.var().idxmax()
        insights.append(
            f"**{var_col}** has the highest variance ({numeric_df.var().max():.2f}) "
            f"→ most spread out, highest information content"
        )

    miss = df.isna().mean() * 100
    if miss.max() > 0:
        mc = miss.idxmax()
        insights.append(
            f"**{mc}** has {miss.max():.1f}% missing → "
            f"{'drop it' if miss.max() > 50 else 'impute before modeling'}"
        )

    useless = [c for c in numeric_df.columns if numeric_df[c].nunique() == 1]
    if useless:
        insights.append(
            f"**{', '.join(useless)}** {'has' if len(useless)==1 else 'have'} "
            f"zero variance → constant, remove before modeling"
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
                f"**{tc}** has {out_counts[tc]} outliers → may distort model training"
            )

    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr().abs()
        np.fill_diagonal(corr.values, 0)
        if corr.max().max() > 0.5:
            idx = corr.stack().idxmax()
            rv  = numeric_df.corr().loc[idx[0], idx[1]]
            insights.append(
                f"**{idx[0]}** & **{idx[1]}** strongly correlated (r={rv:.2f}) "
                f"→ multicollinearity risk"
            )

    insights.append(
        f"Dataset has **{df.shape[0]:,} rows** & **{df.shape[1]} columns** "
        f"→ {'adequate' if df.shape[0] > 500 else 'small'} sample for modeling"
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

    # ── Insights ──────────────────────────────────────────────
    if analysis == "insights":
        return ("__insights__", compute_insights(df, n=6)), "Key insights", ""

    # ── Categorical summary ───────────────────────────────────
    if analysis == "categorical_summary":
        sdf = compute_categorical_summary(df)
        return (sdf if not sdf.empty else None), "Categorical summary", (
            "✅ = good for encoding. ⚠️ ID-like = drop. ⚠️ High missing = impute."
        )

    # ── AutoML ────────────────────────────────────────────────
    if analysis == "automl":
        target_col = plan.get("target_col", "")
        task       = plan.get("task", "regression")
        if not target_col or target_col not in df.columns:
            target_col = df.columns[-1]
        results_df, fi_bytes, report = run_automl(df, target_col, task)
        return ("__automl__", results_df, fi_bytes, report), \
               f"AutoML: predicting '{target_col}'", ""

    # ── Tabular ───────────────────────────────────────────────
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
        col    = pick_col(cols, list(df.columns))
        count  = df[col].nunique() if col else 0
        sample = df[col].dropna().unique()[:10].tolist() if col else []
        return None, f"'{col}' has {count} unique values", f"Sample: {sample}"

    if analysis == "not_useful_column":
        reasons = {}
        for c in numeric_df.columns:
            if numeric_df[c].nunique() == 1:
                reasons[c] = "Constant (zero variance)"
        for c in df.isna().mean()[df.isna().mean() > 0.5].index:
            reasons[c] = f"{df[c].isna().mean()*100:.1f}% missing"
        for c in df.columns:
            if df[c].nunique() == len(df) and c not in reasons:
                reasons[c] = "All unique (likely ID)"
        if reasons:
            return pd.DataFrame(list(reasons.items()), columns=["Column","Reason"]), \
                   "Potentially useless columns", expl
        return None, "All columns appear useful", "No constant/high-missing/ID columns."

    # ── Charts ────────────────────────────────────────────────
    if analysis == "correlation":
        if numeric_df.shape[1] < 2:
            return None, "Need ≥2 numeric columns", ""
        fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                    linewidths=0.4, annot_kws={"size": 7}, ax=ax)
        ax.set_title("Correlation Heatmap", fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)
        return fig_to_bytes(fig), answer or "Correlation heatmap", expl

    if analysis == "histogram":
        col = pick_col(cols, list(numeric_df.columns))
        if not col:
            return None, "No numeric column found", ""
        fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#7c6fcd")
        ax.set_title(f"Distribution: {col}", fontsize=10, fontweight="bold")
        ax.set_xlabel(col, fontsize=8); ax.set_ylabel("Frequency", fontsize=8)
        ax.tick_params(labelsize=7)
        return fig_to_bytes(fig), answer or f"Histogram: {col}", expl

    if analysis == "boxplot":
        col = pick_col(cols, list(numeric_df.columns))
        if not col:
            return None, "No numeric column found", ""
        fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
        sns.boxplot(y=df[col].dropna(), ax=ax, color="#cdb4db", width=0.4)
        ax.set_title(f"Boxplot: {col}", fontsize=10, fontweight="bold")
        ax.set_ylabel(col, fontsize=8); ax.tick_params(labelsize=7)
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
        ax.set_xlabel(valid[0], fontsize=8); ax.set_ylabel(valid[1], fontsize=8)
        ax.tick_params(labelsize=7)
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
        vc.plot(kind="bar", ax=ax, color="#8ec5fc", edgecolor="#5a4fcf", width=0.6)
        ax.set_title(f"Value Counts: {col}", fontsize=10, fontweight="bold")
        ax.set_xlabel(col, fontsize=8); ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(labelsize=7); plt.xticks(rotation=40, ha="right"); plt.tight_layout()
        return fig_to_bytes(fig), answer or f"Value counts: {col}", expl

    if analysis == "variance_max":
        var_s    = numeric_df.var().sort_values(ascending=False)
        col, val = var_s.idxmax(), var_s.max()
        fig, ax  = plt.subplots(figsize=(CHART_W, CHART_H))
        var_s.plot(kind="bar", ax=ax, color="#cdb4db", edgecolor="#5a4fcf", width=0.6)
        ax.set_title("Feature Variances", fontsize=10, fontweight="bold")
        ax.set_ylabel("Variance", fontsize=8); ax.tick_params(labelsize=7)
        plt.xticks(rotation=40, ha="right"); plt.tight_layout()
        return fig_to_bytes(fig), f"Highest variance: {col} ({val:.2f})", \
               expl or f"'{col}' variance={val:.2f}."

    if analysis == "skew_max":
        skew_s   = numeric_df.skew().abs().sort_values(ascending=False)
        col, val = skew_s.idxmax(), skew_s.max()
        fig, ax  = plt.subplots(figsize=(CHART_W, CHART_H))
        skew_s.plot(kind="bar", ax=ax, color="#e0c3fc", edgecolor="#5a4fcf", width=0.6)
        ax.set_title("Feature Skewness", fontsize=10, fontweight="bold")
        ax.set_ylabel("Skewness", fontsize=8); ax.tick_params(labelsize=7)
        plt.xticks(rotation=40, ha="right"); plt.tight_layout()
        return fig_to_bytes(fig), f"Most skewed: {col} (skew={val:.2f})", \
               expl or f"'{col}' skewness={val:.2f}. Consider log transform."

    if analysis == "outliers":
        col = pick_col(cols, list(numeric_df.columns))
        if not col:
            return None, "No numeric column found", ""
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR    = Q3 - Q1
        lo, hi = Q1-1.5*IQR, Q3+1.5*IQR
        n_out  = int(((df[col] < lo) | (df[col] > hi)).sum())
        fig, axes = plt.subplots(1, 2, figsize=(CHART_W*1.6, CHART_H))
        sns.boxplot(y=df[col], ax=axes[0], color="#cdb4db", width=0.4)
        axes[0].set_title(f"Boxplot: {col}", fontsize=9, fontweight="bold")
        axes[0].tick_params(labelsize=7)
        sns.histplot(df[col], kde=True, ax=axes[1], color="#7c6fcd", bins=30)
        axes[1].axvline(lo, color="red", linestyle="--", linewidth=1, label=f"Lower {lo:.1f}")
        axes[1].axvline(hi, color="red", linestyle="--", linewidth=1, label=f"Upper {hi:.1f}")
        axes[1].legend(fontsize=7)
        axes[1].set_title(f"Distribution: {col}", fontsize=9, fontweight="bold")
        axes[1].tick_params(labelsize=7); plt.tight_layout()
        return fig_to_bytes(fig), f"{n_out} outliers in '{col}'", \
               expl or f"IQR: lower={lo:.2f}, upper={hi:.2f}."

    return None, answer or "Analysis complete", expl or "No further details."


def safe_execute(plan, df):
    try:
        if not isinstance(plan, dict) or "analysis" not in plan:
            return None, "Invalid plan from AI", "Try rephrasing."
        return execute_plan(plan, df)
    except Exception as e:
        return None, "Execution error", str(e)


# ═════════════════════════════════════════════════════════════════════════════
# DISPLAY  — adds export buttons + AutoML rendering + filter banner
# ═════════════════════════════════════════════════════════════════════════════
def render_message(msg):
    t = msg["type"]
    if t == "text":
        st.markdown(msg["content"])
    elif t == "dataframe":
        st.dataframe(msg["content"], use_container_width=False)
    elif t == "image":
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(msg["content"])
    elif t == "insights":
        for item in msg["content"]:
            st.markdown(f'<div class="insight-box">💡 {item}</div>', unsafe_allow_html=True)
    elif t == "automl":
        results_df, fi_bytes, report = msg["content"]
        if results_df is not None:
            st.dataframe(results_df, use_container_width=False)
        if fi_bytes:
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.image(fi_bytes)
        st.markdown(f'<div class="ml-box">🤖 {report}</div>', unsafe_allow_html=True)


def display_result(result, answer, explanation):
    stored = []

    # Filter banner
    if st.session_state.active_filter:
        st.markdown(
            f'<div class="filter-banner">🔍 Active filter: '
            f'{st.session_state.active_filter["description"]}</div>',
            unsafe_allow_html=True,
        )

    if answer:
        st.markdown(f'<div class="answer-box"><b>✅ {answer}</b></div>',
                    unsafe_allow_html=True)
        stored.append({"role": "assistant", "type": "text",
                        "content": f"**✅ {answer}**"})

    # Insights
    if isinstance(result, tuple) and result[0] == "__insights__":
        for item in result[1]:
            st.markdown(f'<div class="insight-box">💡 {item}</div>', unsafe_allow_html=True)
        stored.append({"role": "assistant", "type": "insights", "content": result[1]})
        return stored

    # AutoML
    if isinstance(result, tuple) and result[0] == "__automl__":
        _, results_df, fi_bytes, report = result
        if results_df is not None:
            st.dataframe(results_df, use_container_width=False)
            # Export: results table
            csv_bytes = results_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download AutoML results as CSV",
                                csv_bytes, "automl_results.csv", "text/csv",
                                key="dl_automl_csv")
        if fi_bytes:
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.image(fi_bytes)
            # Export: feature importance chart
            st.download_button("⬇️ Download feature importance chart",
                                fi_bytes, "feature_importance.png", "image/png",
                                key="dl_fi_png")
        st.markdown(f'<div class="ml-box">🤖 {report}</div>', unsafe_allow_html=True)
        stored.append({"role": "assistant", "type": "automl",
                        "content": (results_df, fi_bytes, report)})
        return stored

    # DataFrame  — with CSV export button
    if isinstance(result, (pd.DataFrame, pd.Series)):
        st.dataframe(result, use_container_width=False)
        csv_bytes = result.to_csv(index=True).encode()
        st.download_button("⬇️ Download table as CSV",
                            csv_bytes, "table.csv", "text/csv",
                            key="dl_table_csv")
        stored.append({"role": "assistant", "type": "dataframe", "content": result})

    # Chart (bytes)  — with PNG export button
    elif isinstance(result, bytes):
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(result)
        st.download_button("⬇️ Download chart as PNG",
                            result, "chart.png", "image/png",
                            key="dl_chart_png")
        stored.append({"role": "assistant", "type": "image", "content": result})

    # Explanation
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


# ═════════════════════════════════════════════════════════════════════════════
# AI SMART DASHBOARD BUTTON  (preserved exactly from original)
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

    # Sidebar
    st.sidebar.write("📏 Rows:",      df.shape[0])
    st.sidebar.write("📐 Columns:",   df.shape[1])
    st.sidebar.write("❓ Missing:",   int(df.isnull().sum().sum()))
    st.sidebar.write("🔢 Numeric:",   df.select_dtypes(include="number").shape[1])
    st.sidebar.write("🔤 Categorical:", df.select_dtypes(exclude="number").shape[1])
    if st.session_state.rag_index:
        n_docs = len(st.session_state.rag_index[2])
        st.sidebar.markdown(f"**🗂 RAG index:** {n_docs} documents")
    if st.session_state.active_filter:
        st.sidebar.markdown(f"**🔍 Filter:** {st.session_state.active_filter['description']}")
        if st.sidebar.button("❌ Clear filter"):
            st.session_state.active_filter = None
            st.rerun()
    mem_len = len(st.session_state.chat_history) // 2
    st.sidebar.markdown(f"**💬 Memory:** {mem_len} exchanges stored")

    with st.expander("🔍 Dataset Preview"):
        st.dataframe(df.head(), use_container_width=True)

    st.markdown("---")

    # ── DASHBOARD (preserved exactly) ─────────────────────────
    if st.session_state.dashboard_on:
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", int(df.isnull().sum().sum()))

        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if numeric_cols:
            selected_col = st.selectbox("📊 Select Column for Analysis",
                                        numeric_cols, key="selected_column")

            fig = px.histogram(df, x=selected_col, nbins=30,
                               title=f"Distribution of {selected_col}",
                               color_discrete_sequence=["#cdb4db"])
            fig.update_layout(height=350, margin=dict(l=20,r=20,t=40,b=20),
                               plot_bgcolor="white", paper_bgcolor="white")
            fig.update_xaxes(showgrid=True, gridcolor="#e6e6e6")
            fig.update_yaxes(showgrid=True, gridcolor="#e6e6e6")
            fig.update_traces(marker_line_color="white",
                               marker_line_width=1.2, opacity=0.9)
            st.plotly_chart(fig, use_container_width=True, key=f"hist_{selected_col}")

            fig2 = px.box(df, y=selected_col,
                           title=f"Outlier Detection for {selected_col}",
                           color_discrete_sequence=["#ffc8dd"])
            fig2.update_layout(height=350, margin=dict(l=20,r=20,t=40,b=20),
                                plot_bgcolor="white", paper_bgcolor="white")
            fig2.update_yaxes(showgrid=True, gridcolor="#e6e6e6")
            st.plotly_chart(fig2, use_container_width=True, key=f"box_{selected_col}")

            st.markdown("### 🧠 AI Insights")
            query   = f"Analyze column {selected_col} and give insights"
            context, docs = retrieve_context(query)
            insight_prompt = f"""Column: {selected_col}
Context:
{context}
Give 3 short insights about this column."""
            with st.spinner("🤖 Generating insights..."):
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": insight_prompt}]
                )
            insights = response.choices[0].message.content
            st.success(insights)
        else:
            st.warning("No numeric columns available for visualization.")

    # ── CHAT ──────────────────────────────────────────────────
    st.markdown("## 💬 Ask Questions")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            render_message(msg)

    user_query = st.chat_input("Ask anything — analysis, charts, filters, train a model...")

    if user_query:
        st.session_state.messages.append(
            {"role": "user", "type": "text", "content": user_query}
        )
        update_memory("user", user_query)

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking like a data scientist... 🤖"):

                # Step 1: NL filter detection
                df_working, filter_desc = detect_and_apply_filter(user_query, df)

                # Step 2: plan (with conversation memory)
                plan = get_plan(user_query, df_working)

                # Step 3: execute
                result, answer, explanation = safe_execute(plan, df_working)

                # Step 4: display + export buttons
                new_msgs = display_result(result, answer, explanation)
                st.session_state.messages.extend(new_msgs)

                # Update memory with assistant reply
                update_memory("assistant", (answer + " " + explanation)[:300])

else:
    if st.session_state.dashboard_on:
        st.warning("⚠️ Upload a dataset first to use the dashboard.")
    st.info("👈 Upload a CSV file from the sidebar to get started.")

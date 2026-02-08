import streamlit as st
import pandas as pd
import os
from groq import Groq
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
import io
import contextlib

# ---------- CONFIG ----------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
st.set_page_config(page_title="AI AutoML Assistant", layout="wide")

# ---------- SESSION ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- PASTEL UI ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f5f7ff 0%, #eef2ff 40%, #fdf2ff 100%);
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e0c3fc 0%, #8ec5fc 100%);
    color: white;
}

/* HEADERS */
h1, h2, h3 {
    color: #5a4fcf;
}

/* BUTTONS */
.stButton>button {
    background-color: #cdb4db;
    color: white;
    border-radius: 10px;
    border: none;
}
.stButton>button:hover {
    background-color: #b583d6;
}

/* 🔴 FILE UPLOADER BUTTON — PINK */
[data-testid="stFileUploader"] button {
    background-color: #ff77b7 !important;
    color: white !important;
    border-radius: 12px !important;
    border: none !important;
    font-weight: 600;
}
[data-testid="stFileUploader"] button:hover {
    background-color: #ff5fa2 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.title("📊 Dataset Dashboard")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# ---------- MAIN HEADER ----------
st.title("AI AutoML Data Assistant 🤖")
st.markdown("### Your AI-powered Data Scientist")

# ---------- LLM CODE GENERATOR ----------
def ask_llm_for_code(user_query, df):
    schema = {
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict()
    }

    prompt = f"""
You are a Python data analyst.

User request: "{user_query}"

Dataset schema:
{schema}

IMPORTANT:
- Respond ONLY with valid Python code
- No explanations
- Use dataframe variable df
- Use matplotlib/seaborn with pastel colors
- IF the output is a table, ALWAYS assign it to a variable named result
- IF the output is text, ALWAYS use print()

"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()

# ---------- CLEAN CODE ----------
def clean_code(code):
    lines = [l for l in code.split("\n") if l.strip()]
    cleaned = []

    for line in lines:
        if "```" in line.lower() or line.lower().startswith("here"):
            continue
        cleaned.append(line)

    last_line = cleaned[-1].strip()

    # 🛑 DO NOT rewrite plotting code
    plot_keywords = ("plt.", "sns.", "fig", "ax", "import")

    if (
        "=" not in last_line
        and not last_line.startswith("print")
        and not last_line.startswith(plot_keywords)
    ):
        cleaned[-1] = f"result = {last_line}"

    return "\n".join(cleaned)



# ---------- SMART EXECUTION ----------
def execute_code(code, df):
    local_vars = {"df": df, "plt": plt, "sns": sns, "pd": pd}
    output_buffer = io.StringIO()
    plot_buffer = io.BytesIO()

    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code, {}, local_vars)

        # 📊 Plot handling
        if plt.get_fignums():
            plt.tight_layout()
            plt.savefig(plot_buffer, format="png", bbox_inches="tight")
            plt.close()
            return ("plot", plot_buffer)

        # 🧠 Preferred: explicit result variable
        if "result" in local_vars and isinstance(local_vars["result"], pd.DataFrame):
            return ("dataframe", local_vars["result"])

        # 🧠 Auto-detect any dataframe created
        for name, value in local_vars.items():
            if isinstance(value, pd.DataFrame) and name != "df":
                return ("dataframe", value)


        # 📝 Printed output
        printed_output = output_buffer.getvalue()
        if printed_output.strip():
            return ("text", printed_output)

        return ("text", "Done.")

    except Exception as e:
        return ("error", str(e))
    
#---------------------------------------------------------------
def plot_missing_values(df):
    missing = df.isna().sum()
    missing = missing[missing > 0]  # only columns with missing values

    if missing.empty:
        print("No missing values found.")
        return

    plt.figure(figsize=(10, 4))
    sns.barplot(
        x=missing.index,
        y=missing.values,
        palette="pastel"
    )
    plt.xticks(rotation=45, ha="right")
    plt.title("Missing Values per Column")
    plt.ylabel("Missing Count")
    plt.xlabel("Column")
    plt.tight_layout()



# ---------- MAIN APP ----------
if uploaded_file:
    df = pd.read_csv(
    uploaded_file,
    na_values=["?", "NA", "N/A", "null", "NULL", "None", ""]
)

    st.sidebar.write("Rows:", df.shape[0])
    st.sidebar.write("Columns:", df.shape[1])
    st.sidebar.write("Missing Values:", int(df.isnull().sum().sum()))

    with st.expander("🔍 Dataset Preview"):
        st.dataframe(df.head(), use_container_width=True)

    if st.button("🧠 Explain My Dataset"):
        summary = f"""
Columns: {list(df.columns)}
Data Types: {df.dtypes.to_dict()}
Missing Values: {df.isnull().sum().to_dict()}
Sample Rows:
{df.head().to_string()}
"""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a data scientist."},
                {"role": "user", "content": f"Explain this dataset:\\n{summary}"}
            ]
        )
        explanation = response.choices[0].message.content
        st.session_state.messages.append({
            "role": "assistant",
            "type": "text",
            "content": explanation
        })

    # ---------- DISPLAY CHAT ----------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["type"] == "text":
                st.markdown(msg["content"])
            else:
                st.image(msg["content"])

    # ---------- CHAT INPUT ----------
    user_query = st.chat_input("Ask anything about your dataset...")

    if user_query:
        st.session_state.messages.append({
            "role": "user",
            "type": "text",
            "content": user_query
        })

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            if "missing" in user_query.lower():
                missing_counts = df.isna().sum()

                fig, ax = plt.subplots(figsize=(10, 4))
                missing_counts[missing_counts > 0].plot(
                    kind="bar",
                    ax=ax,
                    color="#f4a3c0"
                )
                ax.set_title("Missing Values per Column")
                ax.set_ylabel("Count")
                plt.xticks(rotation=45)
                st.pyplot(fig)
                plt.close()

                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "text",
                    "content": "Displayed missing values per column."
                })
                st.stop() 


            if result_type == "plot":
                st.image(result)
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "image",
                    "content": result
                })

            elif result_type == "dataframe":
                st.dataframe(result)
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "text",
                    "content": "Displayed table."
                })

            elif result_type == "text":
                st.text(result)
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "text",
                    "content": result
                })

            else:
                st.error(result)
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "text",
                    "content": result
                })

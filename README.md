# 🤖 AI AutoML Data Assistant

An AI-powered chatbot that lets you upload any CSV file and ask questions about it in plain English. No coding required — just type your question and get charts, tables, and data-driven insights instantly.

Built with **LLaMA 3.3 70B**, **RAG (Retrieval Augmented Generation)**, **TF-IDF semantic search**, and **Streamlit**.

---

## 📸 What It Does

| You Ask | You Get |
|---|---|
| "Give me 5 insights from this dataset" | Computed insights — skewness, variance, missing values, outliers |
| "Summarize the categorical columns" | Full table of all categorical columns with usefulness flags |
| "Which feature is most spread out?" | RAG finds the answer semantically — no exact word match needed |
| "Show a histogram of Data_value" | Compact matplotlib chart rendered inline |
| "Which columns should I drop?" | Table of useless columns with reasons |
| "Show the correlation heatmap" | Seaborn heatmap of all numeric columns |
| "Does Data_value have outliers?" | Dual chart — boxplot + histogram with IQR boundary lines |

---

## 🏗️ Architecture

```
User Question
      │
      ▼
┌─────────────────────────────────┐
│       RAG PIPELINE              │
│  1. TF-IDF search over ~50 docs │
│  2. Top 6 relevant docs fetched │
│  3. Docs injected into prompt   │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│    LLaMA 3.3 70B (via Groq)     │
│  Returns JSON: analysis + cols  │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│    execute_plan() — Python      │
│  pandas / matplotlib / seaborn  │
│  All numbers computed from data │
└─────────────────────────────────┘
```

---

## 🧠 How RAG Works Here

The LLM has never seen your CSV. Without RAG it would hallucinate column names and statistics. RAG solves this in 3 steps:

**Step 1 — Build the index** (runs once on upload)
- `build_dataset_documents()` converts every column into a natural language document:
  `"Column 'Data_value' — mean=5444, std=7949, skewness=2.3, has 3200 outliers..."`
- Combined with 22 static data science facts into ~50 total documents
- `build_rag_index()` vectorizes all documents using **TF-IDF with bigrams**

**Step 2 — Retrieve** (runs on every question)
- `retrieve_context()` converts the user's question into a TF-IDF vector
- **Cosine similarity** scores all 50 documents against the query
- Top 6 most relevant documents are selected

**Step 3 — Augment** (injected into LLM prompt)
- The 6 retrieved documents are passed to LLaMA 3.3 70B as context
- LLM answers based on actual retrieved facts — not guesswork

> **Why TF-IDF over keyword search?** A question like *"which feature is most spread out?"* contains no column names. TF-IDF scores by word importance and rarity, so it semantically matches to the variance document even without exact word overlap.

---

## ✨ Features

- **17 analysis types** — describe, head/tail, missing values, correlation heatmap, histogram, boxplot, scatter, value counts, variance, skewness, outliers, data types, unique values, insights, categorical summary, not-useful columns
- **Zero hallucination on insights** — `compute_insights()` and `compute_categorical_summary()` compute all numbers directly from pandas, the LLM never touches the output
- **Smart column validation** — `pick_col()` validates every LLM-suggested column against actual dataframe columns before use
- **Persistent chat history** — all messages, charts, and tables stored in `session_state` and correctly re-rendered
- **Auto RAG rebuild** — index only rebuilds when a new file is uploaded (hash-checked)
- **Compact charts** — all plots render at 5.5×3.5 inches, centred in a 3-column layout

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| UI & chat interface | Streamlit |
| LLM | LLaMA 3.3 70B via Groq API |
| RAG retrieval | TF-IDF + Cosine Similarity (scikit-learn) |
| Data processing | Pandas, NumPy |
| Charts | Matplotlib, Seaborn |
| Environment variables | python-dotenv |

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/ai-automl-assistant.git
cd ai-automl-assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your API key

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at [console.groq.com](https://console.groq.com)

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📦 Requirements

```
streamlit
pandas
numpy
groq
python-dotenv
matplotlib
seaborn
scikit-learn
```

Or install from file:

```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
ai-automl-assistant/
│
├── app.py                  # Main application
├── requirements.txt        # Dependencies
├── .env                    # API key (not committed)
├── .gitignore              # Ignores .env and __pycache__
└── README.md               # This file
```

---

## 🔑 Key Functions

| Function | What it does |
|---|---|
| `build_dataset_documents(df)` | Converts CSV into ~30 natural language documents for RAG |
| `build_rag_index(df)` | Builds TF-IDF index over all documents |
| `retrieve_context(query)` | Cosine similarity search — returns top 6 relevant docs |
| `get_plan(query, df)` | Sends retrieved context + schema to LLaMA, gets JSON plan |
| `compute_insights(df)` | Generates data-specific insights directly from pandas |
| `compute_categorical_summary(df)` | Summarises all categorical columns in one table |
| `execute_plan(plan, df)` | Runs the analysis and generates charts/tables |
| `safe_execute(plan, df)` | Wraps execution in try/except for graceful error handling |

---

## 💬 Example Questions to Ask

```
Give me 5 insights from this dataset
Summarize the categorical columns
Which feature has the highest variance?
Which column is most skewed?
Show me a histogram of [column name]
Show the correlation heatmap
Does [column name] have outliers?
Which columns should I drop before modeling?
Show value counts for [column name]
What are the data types?
How many missing values are there?
Show me a boxplot of [column name]
```

---

## ⚙️ How the LLM is Used

The LLM acts purely as a **planner** — it receives the question, retrieved context, and schema, then returns a structured JSON decision:

```json
{
  "analysis": "histogram",
  "columns": ["Data_value"],
  "answer": "",
  "explanation": "Data_value is right-skewed with high variance..."
}
```

All actual computation — statistics, charts, outlier detection — is done by Python. This means:
- No hallucinated numbers
- Reproducible results
- Fast execution (LLM only generates ~50 tokens)

---

## 🔒 Security

- API key stored in `.env` — never hardcoded
- `.env` is in `.gitignore` — never committed to git
- No user data is stored or sent anywhere except the Groq API for the LLM call

---

## 📝 .gitignore

```
.env
__pycache__/
*.pyc
.DS_Store
*.csv
```

---

## 🙋 FAQ

**Q: Which CSV files does it support?**
Any CSV file. The bot automatically detects numeric and categorical columns.

**Q: Does it work without internet?**
The RAG index (TF-IDF) works fully offline. Only the LLM call requires internet to reach the Groq API.

**Q: Is there a row limit?**
No hard limit. Tested on datasets up to 10,000 rows. Very large files may slow down the RAG index build step.

**Q: Why Groq instead of OpenAI?**
Groq provides ultra-fast inference (under 2 seconds) for LLaMA 3.3 70B and has a generous free tier.

---

## 🚧 Known Limitations

- Supports CSV files only (no Excel, JSON, or databases)
- One analysis per question — complex multi-step queries are not supported
- TF-IDF retrieval is less powerful than neural embeddings (no true semantic understanding)
- No ML model training — analysis only, no predictions

---

## 🔮 Future Improvements

- [ ] Replace TF-IDF with sentence-transformers for true semantic RAG
- [ ] Add Python code execution for custom formulas
- [ ] Support Excel and JSON file formats
- [ ] Add ML model training (classification / regression)
- [ ] Multi-turn conversation memory

---

## 👨‍💻 Author

Built as an academic project demonstrating RAG, LLM integration, and automated data analysis.

---

## 📄 License

MIT License — free to use, modify, and distribute.

# Acko Insurance - AI & Analytics Platform

A Data Science bootcamp capstone project: an AI-powered insurance platform modeled on
Acko, covering policy Q&A via RAG, ML-based premium prediction, and a live analytics
dashboard.

## Scope

This submission covers **3 of the original 5 planned modules**:

| Module | Status |
|---|---|
| 1. RAG Policy Chatbot | ✅ Built |
| 2. Premium Quote Predictor | ✅ Built |
| 3. AI Claims Engine | ⏸ Deferred |
| 4. Management Dashboard | ✅ Built |
| 5. Manager AI Assistant | ⏸ Deferred |

Modules 3 and 5 were deliberately scoped out (see the design decisions below) to allow
depth over breadth on the other three.

## Architecture

- **Module 1 (RAG Chatbot)**: PDF policy documents → chunked (`RecursiveCharacterTextSplitter`)
  → embedded (`sentence-transformers/all-MiniLM-L6-v2`) → stored in ChromaDB → retrieved
  per question (top-3 chunks) → answered by Gemini (`gemini-3.6-flash`), grounded strictly
  in retrieved context.
- **Module 2 (Premium Predictor)**: `RandomForestRegressor` trained on combined car+bike
  quotation data (350,000 rows), target log-transformed for skew, SHAP for explainability.
  R² = 0.9875, MAPE = 7.40% on held-out test data.
- **Module 4 (Dashboard)**: Streamlit + Plotly, reading live from a SQLite database
  (`chat_logs` and `quotations` tables) populated by Modules 1 and 2 as the app is used.

## Tech Stack

Python, Streamlit, LangChain, ChromaDB, Google Gemini API, scikit-learn, SHAP, SQLAlchemy,
Plotly.

## Project Structure
├── app.py # Streamlit app (all 3 module UIs)
├── src/
│ ├── rag/ # Module 1: chatbot.py, ingest.py
│ ├── ml/ # Module 2: train_quote_model.py, predict.py, merge_data.py, analyze_errors.py
│ ├── db/ # Shared: SQLAlchemy models, DB connection, table setup
│ └── dashboard/ # Module 4 (dashboard logic lives in app.py directly)
├── data/
│ ├── policies/ # 3 policy PDFs (included)
│ └── datasets/ # Raw + merged quotation CSVs (NOT included - see Setup)
├── tests/
│ └── test_rag_questions.py # 10-question grounding/timing test suite
├── models/
│ ├── quote_predictor.pkl # Trained model (NOT included - see Setup)
│ └── shap_summary.png # SHAP explainability plot (included)
└── chroma_db/ # Vector store (NOT included - see Setup)


## Why some files aren't in this repo

The raw datasets (~186MB), the trained model (~485MB), and the vector store are
regenerable build artifacts, not source code - committing them would bloat the repo and
they'd go stale anyway. What's committed instead is the code that produces them.

## Setup

1. **Clone and create a virtual environment**
git clone https://github.com/RajeshNarayanamoorthy10/acko-insurance-platform.git
cd acko-insurance-platform
python -m venv venv
venv\Scripts\activate # Windows


2. **Install dependencies**
pip install -r requirements.txt


3. **Set up environment variables**
   Copy `.env.example` to `.env` and add your own Gemini API key (get one free at
   [aistudio.google.com](https://aistudio.google.com)):
copy .env.example .env

4. **Build the vector store (Module 1)**
python src/rag/ingest.py


5. **Create the database (Module 4)**
python -m src.db.create_tables


6. **Get the quotation datasets and train the model (Module 2)**
   Place the 4 raw CSVs (`acko_car_quotation.csv`, `acko_bike_quotation.csv`,
   `acko_car_claims.csv`, `acko_bike_claims.csv`) in `data/datasets/`, then:
python src/ml/merge_data.py
python src/ml/train_quote_model.py


7. **Run the app**
streamlit run app.py


## Key Design Decisions

- **Why Random Forest for Module 2?** Tabular data with mixed numeric/categorical
  features and non-linear interactions (e.g. IDV × vehicle age), no need for the
  interpretability trade-offs of linear models given SHAP already provides
  explainability, and it's robust to the outliers common in skewed premium data.
- **Why log-transform the target?** `annual_premium` ranges from ~₹500 to ~₹16 lakh -
  training in log space prevents a handful of expensive policies from dominating the
  loss function.
- **Why MAPE over RMSE as the primary metric?** RMSE is dominated by absolute rupee
  error on expensive policies; MAPE reports a fair relative error across the whole
  price range, which is what actually matters for a typical customer's quote.
- **Data leakage check**: columns algebraically derived from the target (e.g.
  `gst_amount`, correlation 0.9995 with `annual_premium`) were identified and excluded
  before training.
- **Known limitation - RAG response time**: average response time is ~10.6s against a
  <5s target, after tuning (`TOP_K` 4→3, tightened system prompt). Diagnosed as Gemini
  free-tier API latency itself (not local retrieval, which is sub-second), not
  something further fixable without a paid tier.

## Testing

python tests/test_rag_questions.py # Module 1: grounding + response time across all 3 documents
python src/ml/analyze_errors.py # Module 2: prediction error breakdown by price band

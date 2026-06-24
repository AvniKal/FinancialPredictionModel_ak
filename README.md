# AI Financial Forecasting & Anomaly Explainer

An end-to-end time series analysis, forecasting, and anomaly explanation tool powered by a FastAPI backend, a Streamlit frontend, and LLM reasoning. This application downloads economic time series directly from Federal Reserve Economic Data (FRED) (such as Advance Retail Sales, Industrial Production, and Housing Starts) and provides interactive analysis.

---

## Features

- **Automated Configuration Recommendation**: Analyzes series trend strength, seasonality, and dataset size to recommend optimal forecast horizons, initial training sizes, candidate models, and ARIMA orders.
- **Robust Time Series Forecasting**: Fits models (ARIMA or Naïve Seasonal) on full series data and provides forecasts with 95% confidence intervals.
- **z-Score Residual Anomaly Detection**: Decomposes the time series and performs outlier analysis on the multiplicative residuals, highlighting points of unexpected financial deviation.
- **LLM Anomaly Explainer**: Integrates with the Groq API (using Llama-3.1) to generate structured analyst-grade explanation summaries of flagged anomalies in their historical context.
- **Walk-Forward Validation Model Race**: Runs a backtesting engine to compare forecasts across models, outputting a leaderboard ranking based on MASE, MAPE, and RMSE.

---

## Project Structure

```text
FinancialPredictionModel_ak/
├── api/                                    # FastAPI REST API Backend
│   ├── __init__.py
│   ├── main.py                             # API routes and router logic
│   └── models.py                           # Pydantic request/response schemas
├── src/                                    # Core logic and ML modules
│   ├── __init__.py
│   ├── agents.py                           # Anomaly detection & Groq LLM explainer
│   ├── backtest.py                         # Walk-forward backtesting
│   ├── data_loader.py                      # FRED data downloader via pandas-datareader
│   ├── models.py                           # Model fitting and forecasting functions
│   ├── preprocessing.py                    # Time series cleaning & decomposition
│   └── recommender.py                      # Trend & character-based config recommender
├── app.py                                  # Streamlit UI application frontend
├── requirements.txt                        # Clean, version-pinned direct dependencies
├── .env                                    # API credentials (GROQ_API_KEY)
└── README.md                               # Project documentation
```

---

## Setup & Running

### **1. Set up Environment Variables**
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY="your-groq-api-key-here"
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run the Backend API**
Start the FastAPI development server:
```bash
uvicorn api.main:app --reload --port 8000
```

### **4. Run the Streamlit App**
In another terminal, start the Streamlit frontend:
```bash
streamlit run app.py
```

---

## API Reference

Interactive Swagger docs are available at: http://localhost:8000/docs

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/recommend` | POST | Auto-recommend modeling config |
| `/forecast` | POST | Generate future forecast |
| `/backtest` | POST | Run walk-forward model race |
| `/anomalies` | POST | Detect residual anomalies |
| `/explain` | POST | LLM explanation for anomaly |

### Example — Explain the April 2020 Anomaly
```bash
curl -X POST "http://localhost:8000/explain" \
  -H "Content-Type: application/json" \
  -d '{"series_id":"RSXFS","anomaly_date":"2020-04-01","threshold_std":2.5}'
```

<!--
- Built end-to-end AI forecasting tool with FastAPI REST backend
  (6 endpoints) and Streamlit frontend; decoupled via HTTP API layer
- Implemented walk-forward backtesting engine across 5 models
  (ARIMA, SARIMAX, Prophet, XGBoost, Naïve); ARIMA achieved 42%
  MASE improvement over naïve baseline on 25-year FRED retail dataset
- Developed rule-based recommender that auto-selects forecast horizon,
  training size, and candidate models from series characteristics
  (trend strength, seasonality, structural breaks)
- Integrated Groq Llama-3.1 API for anomaly explanation with structured
  output parsing, exponential backoff retry, and graceful fallback handling
-->

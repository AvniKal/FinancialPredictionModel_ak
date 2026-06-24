# Expenditure & Revenue Forecasting App

This Streamlit application forecasts **expenditure** and **revenue** for upcoming fiscal years using ARIMA time series modeling.  
The app provides interactive visualizations and allows filtering by **G/L Account** or **Cost Centre**.

---

## Features
- Forecast **expenditure** and **revenue** separately.
- Interactive line charts (Plotly).
- Filter forecasts by **G/L Account** or **Cost Centre**.
- Tabbed interface for switching between expenditure and revenue.
- Data-driven insights using ARIMA models.

---

FinancialPredictionModel_ak/<br>
~~~
├── notebooks/                              # Jupyter notebooks for experimentation
│   ├── MAV_Financial Planning Expenditure.ipynb
│   └── MAV_Financial Planning Revenue.ipynb
|-- MAV_Financial Planning Expenditure.csv
|-- MAV_Financial Planning Revenue.csv
├── app.py                                  # Main Streamlit application file
├── requirements.txt                        # Python dependencies
└── README.md                               # Existing project documentation

~~~

---

## Setup & Running

### **1. Clone the Repository**
```bash
git clone https://github.com/AvniKal/ExpenditurePredictionModel_ak.git
cd ExpenditurePredictionModel_ak
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run the Streamlit App**
```bash
streamlit run app.py
```

## API Reference

Start the API server:
    uvicorn api.main:app --reload --port 8000

Interactive docs: http://localhost:8000/docs

| Endpoint | Method | Description |
|----------|--------|-------------|
| /health | GET | Health check |
| /recommend | POST | Auto-recommend modeling config |
| /forecast | POST | Generate future forecast |
| /backtest | POST | Run walk-forward model race |
| /anomalies | POST | Detect residual anomalies |
| /explain | POST | LLM explanation for anomaly |

Example — explain the April 2020 anomaly:
    curl -X POST "http://localhost:8000/explain" \
      -H "Content-Type: application/json" \
      -d '{"series_id":"RSXFS","anomaly_date":"2020-04-01","threshold_std":2.5}'

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




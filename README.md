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
├── notebooks/                              # Jupyter notebooks for experimentation<br>
│   ├── MAV_Financial Planning Expenditure.ipynb<br>
│   └── MAV_Financial Planning Revenue.ipynb<br>
|-- MAV_Financial Planning Expenditure.csv<br>
|-- MAV_Financial Planning Revenue.csv<br>
├── app.py                                  # Main Streamlit application file<br>
├── requirements.txt                        # Python dependencies<br>
└── README.md                               # Existing project documentation<br>

~~~

---

## Installation

### **1. Clone the Repository**
```bash
git clone https://github.com/AvniKal/ExpenditurePredictionModel_ak.git
cd ExpenditurePredictionModel_ak

### **2.Install Dependencies**
pip install -r requirements.txt

### **3.Run the Streamlit App**
streamlit run app.py



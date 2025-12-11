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

## Folder Structure
ExpenditurePredictionModel_ak/
│

├── app.py # Streamlit main app file

├── requirements.txt # Python dependencies

├── README.md # Project documentation
│
├── data/ # CSV datasets
│ ├── MAV_Financial Planning Expenditure.csv
│ └── MAV_Financial Planning Revenue.csv
│
├── notebooks/ # Jupyter or Google Colab notebooks (optional)
│ └── revenue_preduction.ipynb
  └── expenditure_prediction.ipynb


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



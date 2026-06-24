import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import requests
from src.data_loader import fetch_fred_series
from src.preprocessing import clean_series

API_BASE = "http://localhost:8000"

def api_post(endpoint: str, payload: dict) -> dict:
    """Call FastAPI endpoint, raise st.error on failure."""
    try:
        response = requests.post(f"{API_BASE}{endpoint}", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API server. Run: uvicorn api.main:app --reload")
        st.stop()
    except Exception as e:
        st.error(f"API error: {str(e)}")
        st.stop()

st.set_page_config(
    page_title="AI Financial Forecasting Tool",
    page_icon="📈",
    layout="wide"
)

# Cache expensive operations so they don't rerun on every widget interaction
@st.cache_data
def load_data(series_id: str, start: str) -> pd.Series:
    raw_series = fetch_fred_series(series_id, start=start)
    return clean_series(raw_series, freq='MS')

# --- Session State Initializations ---
if "race_results" not in st.session_state:
    st.session_state["race_results"] = None
if "run_race_clicked" not in st.session_state:
    st.session_state["run_race_clicked"] = False

# --- SIDEBAR ---
st.sidebar.title("⚙️ Configuration")

series_display_names = {
    "RSXFS": "RSXFS (Retail Sales)",
    "INDPRO": "INDPRO (Industrial Production)",
    "HOUST": "HOUST (Housing Starts)"
}

series_id = st.sidebar.selectbox(
    "FRED Series",
    options=["RSXFS", "INDPRO", "HOUST"],
    format_func=lambda x: series_display_names.get(x, x)
)

start_date = st.sidebar.text_input("Start Date", value="2000-01-01")

anomaly_threshold = st.sidebar.slider(
    "Anomaly Detection Sensitivity",
    min_value=1.5,
    max_value=4.0,
    value=2.5,
    step=0.1,
    help="Lower = more anomalies flagged, Higher = only severe anomalies"
)

run_race = st.sidebar.button("Run Model Race 🏁")
if run_race:
    st.session_state["run_race_clicked"] = True

st.sidebar.divider()

# Load series to get recommend_config for sidebar info
series = None
config = None

try:
    series = load_data(series_id, start_date)
    response = requests.post(f"{API_BASE}/recommend", params={"series_id": series_id, "start_date": start_date})
    response.raise_for_status()
    config = response.json()
except requests.exceptions.ConnectionError:
    st.sidebar.error("Cannot connect to API server. Run: uvicorn api.main:app --reload")
    st.stop()
except Exception as e:
    st.sidebar.error(f"Error loading data or configuration: {e}")
    st.stop()

# Table 1: Parameter | Value
param_value_df = pd.DataFrame({
    "Parameter": [
        "Forecast Horizon",
        "Initial Train Size",
        "Recommended Models",
        "ARIMA Order"
    ],
    "Value": [
        str(config["forecast_horizon"]),
        str(config["initial_train_size"]),
        ", ".join(config["recommended_models"]),
        str(config["arima_order"])
    ]
})
st.sidebar.markdown("### Recommended Parameters")
st.sidebar.table(param_value_df)

# Table 2: Parameter | Reasoning
param_reasoning_df = pd.DataFrame({
    "Parameter": [
        "Forecast Horizon",
        "Initial Train Size",
        "Recommended Models",
        "ARIMA Order"
    ],
    "Reasoning": [
        config["reasoning"]["forecast_horizon"],
        config["reasoning"]["initial_train_size"],
        config["reasoning"]["recommended_models"],
        config["reasoning"]["arima_order"]
    ]
})
st.sidebar.markdown("### Parameter Reasoning")
st.sidebar.table(param_reasoning_df)

# --- MAIN AREA ---
tab1, tab2, tab3 = st.tabs(["📈 Forecast", "🚨 Anomaly Detection", "🏆 Model Leaderboard"])

# --- TAB 1: FORECAST ---
with tab1:
    st.header("📈 Time Series Forecasting")
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        model_choice = st.selectbox(
            "Model",
            options=["ARIMA (Recommended)", "Naïve Seasonal"]
        )
        
        default_horizon = config.get("forecast_horizon", 12) if 'config' in locals() else 12
        horizon = st.slider(
            "Horizon (Months)",
            min_value=1,
            max_value=24,
            value=int(default_horizon)
        )
        
        generate_forecast = st.button("Generate Forecast 🔮")
        
    with col_right:
        if generate_forecast:
            with st.spinner("Fitting model and generating forecast..."):
                try:
                    model_name = "arima" if model_choice == "ARIMA (Recommended)" else "naive"
                    
                    # Fetch forecast from API
                    res = api_post("/forecast", {
                        "series_id": series_id,
                        "start_date": start_date,
                        "horizon": int(horizon),
                        "model": model_name
                    })
                    
                    # Reconstruct forecast Series
                    forecast_dates = pd.to_datetime(res["forecast_dates"])
                    forecast = pd.Series(res["forecast_values"], index=forecast_dates)
                    
                    # Get RMSE from /backtest endpoint as an estimate of residual std
                    backtest_res = api_post("/backtest", {
                        "series_id": series_id,
                        "start_date": start_date,
                        "initial_train_size": int(config["initial_train_size"]),
                        "step_size": 3,
                        "forecast_horizon": int(horizon),
                        "models_to_run": [model_name]
                    })
                    
                    if backtest_res["leaderboard"] and backtest_res["leaderboard"][0]["status"] == "ok":
                        resid_std = backtest_res["leaderboard"][0]["rmse"]
                    else:
                        resid_std = series.std()
                            
                    # Construct upper and lower bounds
                    upper_bound = forecast + 1.96 * resid_std
                    lower_bound = forecast - 1.96 * resid_std
                    
                    # Plot using Plotly
                    fig = go.Figure()
                    
                    # Historical (blue)
                    fig.add_trace(go.Scatter(
                        x=series.index,
                        y=series.values,
                        name="Historical",
                        line=dict(color="#1f77b4", width=2)
                    ))
                    
                    # Forecast (orange dashed)
                    fig.add_trace(go.Scatter(
                        x=forecast.index,
                        y=forecast.values,
                        name="Forecast",
                        line=dict(color="#ff7f0e", width=2, dash="dash")
                    ))
                    
                    # Confidence interval shaded area
                    fig.add_trace(go.Scatter(
                        x=list(forecast.index) + list(forecast.index)[::-1],
                        y=list(upper_bound) + list(lower_bound)[::-1],
                        fill='toself',
                        fillcolor='rgba(255, 127, 14, 0.15)',
                        line=dict(color='rgba(255,127,14,0)'),
                        hoverinfo="skip",
                        showlegend=True,
                        name="95% Confidence Interval"
                    ))
                    
                    # Vertical line at forecast start date
                    fig.add_vline(
                        x=forecast.index[0],
                        line_width=1.5,
                        line_dash="dot",
                        line_color="gray"
                    )
                    
                    fig.update_layout(
                        title=f"{series_id} Forecast — Next {horizon} Months",
                        xaxis_title="Date",
                        yaxis_title="Value",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Metrics row
                    st.markdown("### Forecast Metrics")
                    m_col1, m_col2, m_col3 = st.columns(3)
                    
                    last_observed = float(series.iloc[-1])
                    forecast_end = float(forecast.iloc[-1])
                    projected_change = (forecast_end - last_observed) / last_observed * 100
                    
                    m_col1.metric("Last Observed Value", f"{last_observed:,.2f}")
                    m_col2.metric("Forecast End Value", f"{forecast_end:,.2f}")
                    m_col3.metric("Projected Change %", f"{projected_change:+.2f}%")
                    
                except Exception as e:
                    st.error(f"Error generating forecast: {e}")

# --- TAB 2: ANOMALY DETECTION ---
with tab2:
    st.header("🚨 Anomaly Detection")
    try:
        # Detect anomalies via API
        res = api_post("/anomalies", {
            "series_id": series_id,
            "start_date": start_date,
            "threshold_std": anomaly_threshold
        })
        
        # Reconstruct df_anomalies DataFrame
        if res["anomalies"]:
            df_anomalies = pd.DataFrame(res["anomalies"])
            df_anomalies["date"] = pd.to_datetime(df_anomalies["date"])
            df_anomalies.set_index("date", inplace=True)
        else:
            df_anomalies = pd.DataFrame(columns=["observed", "residual", "deviation_pct", "direction", "severity"])
            df_anomalies.index.name = "date"
            
        n_anom = len(df_anomalies)
        
        st.write(f"Found **{n_anom}** anomalies in **{series_id}** since **{start_date}**")
        
        # Display DataFrame formatted and colored
        df_display = df_anomalies.reset_index()
        if not df_display.empty:
            df_display["date"] = df_display["date"].dt.strftime("%Y-%m-%d")
            df_display = df_display[["date", "observed", "deviation_pct", "direction", "severity"]]
            
            def color_severity(val):
                if val == "severe":
                    return "background-color: rgba(255, 75, 75, 0.2); color: #FF4B4B; font-weight: bold;"
                elif val == "moderate":
                    return "background-color: rgba(255, 193, 7, 0.2); color: #FFC107; font-weight: bold;"
                return ""
            
            st.dataframe(df_display.style.applymap(color_severity, subset=["severity"]), use_container_width=True)
        else:
            st.info("No anomalies detected with the current sensitivity setting.")
            
        # Plotly anomalies scatter plot
        fig_anom = go.Figure()
        
        # Full series (blue)
        fig_anom.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            name="Observed",
            line=dict(color="#1f77b4", width=2)
        ))
        
        # Anomalies (red dots)
        if not df_anomalies.empty:
            fig_anom.add_trace(go.Scatter(
                x=df_anomalies.index,
                y=df_anomalies["observed"],
                mode="markers",
                name="Anomaly",
                marker=dict(color="#FF4B4B", size=8),
                customdata=df_anomalies["deviation_pct"],
                hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Observed:</b> %{y:,.2f}<br><b>Deviation:</b> %{customdata:+.2f}%<extra></extra>"
            ))
            
        fig_anom.update_layout(
            title=f"{series_id} Anomalies (threshold={anomaly_threshold})",
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode="closest"
        )
        st.plotly_chart(fig_anom, use_container_width=True)
        
        # Anomaly Explainer
        st.markdown("### Anomaly Explainer 🤖")
        if not df_anomalies.empty:
            anomaly_dates = df_anomalies.index.sort_values(ascending=False)
            selected_date = st.selectbox(
                "Select Anomaly Date to Explain",
                options=anomaly_dates,
                format_func=lambda x: x.strftime("%B %Y (%Y-%m-%d)")
            )
            
            if st.button("Explain This Anomaly 🤖"):
                with st.spinner("Asking Groq..."):
                    try:
                        explanation_res = api_post("/explain", {
                            "series_id": series_id,
                            "start_date": start_date,
                            "anomaly_date": selected_date.strftime("%Y-%m-%d"),
                            "threshold_std": anomaly_threshold
                        })
                        
                        # Reconstruct anomaly context for display expander
                        idx = series.index.get_loc(selected_date)
                        preceding = series.iloc[max(0, idx - 12):idx] if idx > 0 else series.iloc[:1]
                        expected_val = float(preceding.mean()) if not preceding.empty else 0.0
                        observed_val = float(series.loc[selected_date])
                        pct_dev = ((observed_val - expected_val) / expected_val * 100) if expected_val != 0 else 0.0
                        
                        context = {
                            "date_str": selected_date.strftime("%B %Y"),
                            "observed": observed_val,
                            "expected": expected_val,
                            "pct_deviation": pct_dev,
                            "window_values": {
                                d.strftime("%B %Y"): float(val)
                                for d, val in series.iloc[max(0, idx - 3):min(len(series), idx + 4)].items()
                            }
                        }
                        
                        with st.expander("📝 Explanation", expanded=True):
                            st.write(explanation_res.get("explanation", "No explanation returned."))
                            
                        with st.expander("📊 Confidence & Factors", expanded=True):
                            conf = explanation_res.get("confidence", "low").lower()
                            if conf == "high":
                                st.success("Confidence: High")
                            elif conf == "medium":
                                st.warning("Confidence: Medium")
                            else:
                                st.error("Confidence: Low")
                            
                            factors = explanation_res.get("factors", [])
                            if factors:
                                st.write("Contributing Factors:")
                                for f in factors:
                                    st.write(f"- {f}")
                            else:
                                st.write("No factors provided.")
                                
                        with st.expander("🔍 Raw Context"):
                            st.json(context)
                    except Exception as exc:
                        st.error(f"Error explaining anomaly: {exc}")
        else:
            st.info("No anomalies detected to explain.")
            
    except Exception as e:
        st.error(f"Error analyzing anomalies: {e}")

# --- TAB 3: MODEL LEADERBOARD ---
with tab3:
    st.header("🏆 Model Leaderboard")
    
    if st.session_state.get("run_race_clicked", False):
        progress_bar = st.progress(0.1)
        st.write("Initializing model race backtesting...")
        
        try:
            # Run race via API
            res = api_post("/backtest", {
                "series_id": series_id,
                "start_date": start_date,
                "initial_train_size": int(config['initial_train_size']),
                "step_size": 3,
                "forecast_horizon": int(config['forecast_horizon'])
            })
            results = res["leaderboard"]
            progress_bar.progress(1.0)
            st.session_state["race_results"] = results
        except Exception as e:
            st.error(f"Error running model race: {e}")
        finally:
            st.session_state["run_race_clicked"] = False
            
    if st.session_state["race_results"] is not None:
        results = st.session_state["race_results"]
        
        # Display Leaderboard
        leaderboard_rows = []
        for rank, r in enumerate(results, 1):
            leaderboard_rows.append({
                "Rank": rank,
                "Model": r["model_name"],
                "MAPE": f"{r['mape']:.2f}%" if r["mape"] is not None else "N/A",
                "RMSE": f"{r['rmse']:.2f}" if r["rmse"] is not None else "N/A",
                "MASE": f"{r['mase']:.4f}" if r["mase"] is not None else "N/A",
                "Fit Time": f"{r['fit_time_seconds']:.2f}s" if r["fit_time_seconds"] is not None else "N/A",
                "Status": r["status"].upper()
            })
        df_leaderboard = pd.DataFrame(leaderboard_rows)
        
        def style_leaderboard(row):
            style_list = [""] * len(row)
            status_val = row["Status"]
            rank_val = row["Rank"]
            if status_val == "FAILED":
                return ["background-color: rgba(255, 75, 75, 0.15); color: #FF4B4B;"] * len(row)
            elif rank_val == 1:
                return ["background-color: rgba(46, 204, 113, 0.15); color: #2ECC71; font-weight: bold;"] * len(row)
            return style_list

        st.dataframe(df_leaderboard.style.apply(style_leaderboard, axis=1), use_container_width=True)
        
        # Plotly bar chart
        bar_colors = []
        for idx, r in enumerate(results):
            if r["status"] == "failed":
                bar_colors.append("#FF4B4B")
            elif idx == 0:
                bar_colors.append("#2ECC71")
            else:
                bar_colors.append("#1f77b4")
                
        plot_names = [r["model_name"] for r in results]
        plot_mases = [r["mase"] if r["mase"] is not None else 0.0 for r in results]
        
        fig_leaderboard = go.Figure(data=[
            go.Bar(
                x=plot_names,
                y=plot_mases,
                marker_color=bar_colors,
                text=[f"{m:.4f}" if m > 0 else "Failed" for m in plot_mases],
                textposition="auto"
            )
        ])
        fig_leaderboard.update_layout(
            title="Model Race Results — MASE Comparison",
            xaxis_title="Model",
            yaxis_title="MASE (Lower is Better)"
        )
        st.plotly_chart(fig_leaderboard, use_container_width=True)
        
        # Three st.metric cards side by side
        ok_results = [r for r in results if r["status"] == "ok"]
        l_col1, l_col2, l_col3 = st.columns(3)
        if ok_results:
            best_mase_model = ok_results[0]
            best_mape_model = min(ok_results, key=lambda x: x["mape"] if x["mape"] is not None else float('inf'))
            fastest_model = min(ok_results, key=lambda x: x["fit_time_seconds"] if x["fit_time_seconds"] is not None else float('inf'))
            
            l_col1.metric("Best MASE", f"{best_mase_model['mase']:.4f}", f"Model: {best_mase_model['model_name']}")
            l_col2.metric("Best MAPE", f"{best_mape_model['mape']:.2f}%", f"Model: {best_mape_model['model_name']}")
            l_col3.metric("Fastest Model", f"{fastest_model['fit_time_seconds']:.2f}s", f"Model: {fastest_model['model_name']}")
        else:
            l_col1.metric("Best MASE", "N/A")
            l_col2.metric("Best MAPE", "N/A")
            l_col3.metric("Fastest Model", "N/A")
            
    else:
        st.info("Click 'Run Model Race' in the sidebar to compare models")

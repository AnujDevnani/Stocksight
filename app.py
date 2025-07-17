import streamlit as st
import pandas as pd
import plotly.express as px
from stock_predictor import *
import yfinance as yf
import os
import subprocess
import time
import io
import sys

def safe_series(preds):
    if preds is None:
        return pd.Series(dtype=float)
    try:
        return pd.Series(preds)
    except Exception:
        return pd.Series([preds])

st.set_page_config(layout="wide")
st.title("💰 Indian Stock Market Predictor (NSE/BSE)")

# 📌 Sidebar Input
st.sidebar.header("Select or Enter Stock")
# List all available stocks by checking models_* directories
available_stocks = [d.replace('models_', '').replace('_', '.') for d in os.listdir('.') if d.startswith('models_')]
stock_name = st.sidebar.text_input("Stock Name (e.g. RELIANCE.NS)", value=available_stocks[0] if available_stocks else "", help="Type or select a stock symbol (e.g. RELIANCE.NS)")

# Removed user data upload feature

# 🔄 Retrain/Clean Models UI
st.sidebar.markdown("---")
st.sidebar.subheader("Model Maintenance")
if stock_name:
    st.sidebar.warning("Retraining may take several minutes and use significant compute. Only retrain if you want to update models with new/latest data.")
    clean_confirm = st.sidebar.checkbox("Are you sure? This will DELETE all existing models for this stock.", key="clean_confirm")
    if clean_confirm:
        clean_action = st.sidebar.selectbox("Select action", ["Clean only"], key="clean_action")
        if st.sidebar.button("Clean", key="clean_btn"):
            with st.spinner("Cleaning all models for this stock..."):
                try:
                    model_dir = f"models_{stock_name.replace('.', '_')}"
                    if os.path.exists(model_dir):
                        import shutil
                        shutil.rmtree(model_dir)
                    st.sidebar.success("Clean complete! All models for this stock have been deleted.")
                except Exception as e:
                    st.sidebar.error(f"Error during clean: {e}")
    # Existing retrain button (optional, can keep or remove)
    if st.sidebar.button("Retrain Models", key="retrain_btn"):
        if st.sidebar.checkbox("Are you sure? This may take a while.", key="retrain_confirm"):
            with st.spinner("Retraining all models for this stock..."):
                try:
                    # Only use yfinance data for retraining
                    cmd = [sys.executable, "train_models.py", "--stock", stock_name]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        st.sidebar.success("Retraining complete! Reloading models...")
                    else:
                        st.sidebar.error(f"Retraining failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
                except Exception as e:
                    st.sidebar.error(f"Error during retraining: {e}")
        else:
            st.sidebar.info("Please confirm retraining by checking the box above.")

# 📖 Help & Info Sidebar
with st.sidebar.expander("ℹ️ Help & Info", expanded=False):
    st.markdown("""
    **How to use this app:**
    1. Select a stock from the dropdown.
    2. Click 'Analyze Stock' to view predictions, news, and insights.
    3. Review the charts, recommendations, and explanations.
    
    **Models Used:**
    - **LSTM:** Deep learning model for sequential price prediction.
    - **Prophet:** Facebook's time series forecasting model.
    - **XGBoost:** Gradient boosting for tabular/engineered features.
    - **Ensemble:** Combines all models for robust prediction.
    
    **Glossary:**
    - **RSI:** Relative Strength Index (momentum indicator)
    - **MACD:** Moving Average Convergence Divergence
    - **EMA:** Exponential Moving Average
    - **Bollinger Bands:** Volatility bands
    - **OBV:** On-Balance Volume
    - **ATR:** Average True Range
    - **SHAP:** Model explainability (feature impact)
    
    **How to interpret predictions:**
    - **Buy:** Models and sentiment predict price increase.
    - **Sell:** Models and sentiment predict price decrease.
    - **Hold:** No strong trend detected.
    - **Confidence:** Lower uncertainty = more reliable prediction.
    """)

if stock_name and st.sidebar.button("Analyze Stock"):
    # Check if models exist for this stock
    model_dir = f"models_{stock_name.replace('.', '_')}"
    if not os.path.exists(model_dir):
        st.warning(f"No trained models found for {stock_name}. Training models now. This may take a few minutes...")
        with st.spinner(f"Training models for {stock_name}..."):
            try:
                # Only use yfinance data for training
                cmd = [sys.executable, "train_models.py", "--stock", stock_name]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    st.success(f"Training complete for {stock_name}! Visualizations will appear below shortly.")
                    time.sleep(2)
                else:
                    st.error(f"Training failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
                    st.stop()
            except Exception as e:
                st.error(f"Error during training: {e}")
                st.stop()
    with st.spinner("Crunching data..."):
        # Always use yfinance data
        data = get_stock_data(stock_name)
        # 📈 Show 4-year historical price chart
        st.subheader(f"📈 {stock_name} Stock Price (Last 4 Years)")
        st.line_chart(data['Close'])
        # Get 30-day predictions for all models and ensemble
        lstm_preds, prophet_preds, xgb_preds, lgbm_preds, catboost_preds, arima_preds, ensemble_preds = get_ensemble_30day_predictions(stock_name, data)
        days = list(range(1, 31))
        # Update the predictions DataFrame to include all models
        pred_df = pd.DataFrame({
            'Day': days,
            'LSTM': lstm_preds,
            'Prophet': prophet_preds,
            'XGBoost': xgb_preds,
            'LightGBM': safe_series(lgbm_preds),
            'CatBoost': safe_series(catboost_preds),
            'ARIMA': safe_series(arima_preds),
            'Ensemble': ensemble_preds
        })
        # Show warnings if any model is not available
        if lgbm_preds is None:
            st.warning('LightGBM model is not available or failed to train.')
        if catboost_preds is None:
            st.warning('CatBoost model is not available or failed to train.')
        if arima_preds is None:
            st.warning('ARIMA model is not available or failed to train.')

        # 📈 Show as line chart
        st.subheader("🔮 Next 30 Days Prediction (All Models)")
        chart_fig = px.line(pred_df, x='Day', y=pred_df.columns[1:], title="30-Day Predictions")
        st.plotly_chart(chart_fig, use_container_width=True)
        # 📋 Show as table
        st.subheader("📋 30-Day Prediction Table (All Models)")
        st.dataframe(pred_df, use_container_width=True)
        # ⬇️ Export buttons
        csv = pred_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions (CSV)", csv, "predictions.csv", "text/csv")
        img_bytes = io.BytesIO()
        chart_fig.write_image(img_bytes, format='png')
        st.download_button("Download Chart (PNG)", img_bytes.getvalue(), "predictions.png", "image/png")

        # 📊 Model Confidence/Uncertainty
        st.subheader("📊 Model Confidence (Uncertainty)")
        def safe_std(preds):
            if preds is None or all(v == 'N/A' for v in preds):
                return 0
            return float(pd.Series(preds).std())

        conf_df = pd.DataFrame({
            'Model': ['LSTM', 'Prophet', 'XGBoost', 'LightGBM', 'CatBoost', 'ARIMA', 'Ensemble'],
            'StdDev': [
                float(pd.Series(lstm_preds).std()),
                float(pd.Series(prophet_preds).std()),
                float(pd.Series(xgb_preds).std()),
                safe_std(lgbm_preds),
                safe_std(catboost_preds),
                safe_std(arima_preds),
                float(pd.Series(ensemble_preds).std())
            ]
        })
        st.bar_chart(conf_df.set_index('Model'))
        st.caption("Lower standard deviation = higher confidence in prediction.")

        # 📝 Simple Explanation of Each Model
        with st.expander("What do these models mean?", expanded=False):
            st.markdown("""
            **LSTM:** Deep learning model that learns from past price sequences and patterns.
            
            **Prophet:** Facebook's time series model, good for trends and seasonality.
            
            **XGBoost:** Powerful machine learning model using engineered features (technical, fundamental, sentiment, etc.).
            
            **LightGBM:** Fast, efficient gradient boosting model, similar to XGBoost but with different optimizations.
            
            **CatBoost:** Gradient boosting model that handles categorical features well, often robust to overfitting.
            
            **ARIMA:** Classic statistical model for time series, good for capturing autocorrelation and trends.
            
            **Ensemble:** Combines all the above models for a more robust, balanced prediction.
            """)

        # Removed backtest and all user_data references

        # 💡 Recommendation
        st.subheader("💡 Trading Recommendation")
        sentiment, headline_sentiments = analyze_sentiment(stock_name.split('.')[0])
        recommendation = get_recommendation(lstm_preds, prophet_preds, sentiment)
        st.success(recommendation)

        # Explain recommendation in simple terms
        explanation = ""
        if "BUY" in recommendation:
            explanation = (
                "Buy because the ensemble of advanced models predicts a price increase, and recent news sentiment is favorable. "
                "All models (LSTM, Prophet, XGBoost) forecast growth, and news headlines are mostly positive."
            )
        elif "SELL" in recommendation:
            explanation = (
                "Sell because the ensemble of advanced models predicts a price decrease, and recent news sentiment is unfavorable. "
                "All models forecast a decline, and news headlines are mostly negative."
            )
        else:
            explanation = (
                "Hold because the models and news sentiment do not show a strong trend. "
                "It's best to wait for clearer signals."
            )
        st.info(explanation)

        # Why this prediction? (Expanded Explainability)
        with st.expander("Why this prediction? (Model Explainability)"):
            # XGBoost SHAP
            pred, shap_feats = get_xgboost_shap_for_last_day(data, stock_name)
            st.markdown("**XGBoost (Top Influencing Features):**")
            for feat, val in shap_feats:
                st.write(f"{feat}: {val:.2f}")
            # LSTM Explainability (recent days influence)
            st.markdown("**LSTM (Recent Days Influence):**")
            try:
                lstm_attn = get_lstm_attention_for_last_day(data, stock_name)  # implement this if available
                st.bar_chart(pd.Series(lstm_attn, name="Attention Weight").tail(30))
            except Exception:
                st.info("LSTM attention/feature importances not available. Typically, the last 7-14 days have the most influence on LSTM predictions.")
            # Prophet Explainability (trend/seasonality)
            st.markdown("**Prophet (Trend/Seasonality Impact):**")
            try:
                prophet_trend, prophet_seasonality = get_prophet_components_for_last_day(data, stock_name)  # implement if available
                st.write(f"Trend: {prophet_trend:.2f}, Seasonality: {prophet_seasonality:.2f}")
            except Exception:
                st.info("Prophet trend/seasonality decomposition not available. Prophet typically splits prediction into trend and seasonality components.")
            # Summary
            st.markdown(
                f"**Summary:** The forecast was most influenced by: {shap_feats[0][0]} (XGBoost), recent days (LSTM), and trend/seasonality (Prophet)."
            )

        # 📰 News Sentiment Headlines Used
        st.subheader("📰 News Headlines Used for Sentiment Analysis")
        sentiment, headline_sentiments = analyze_sentiment(stock_name.split('.')[0])
        for headline, label, score in headline_sentiments:
            st.markdown(f"**{headline}**\n- Sentiment: {label} ({score:.2f})")

        # ℹ️ Key Metrics
        st.subheader("ℹ️ Key Metrics")
        last_date = data.index[-1] if hasattr(data.index, 'freq') or hasattr(data.index, 'dtype') else data.index.values[-1]
        st.metric("Current Price", f"₹{data['Close'].iloc[-1]:.2f}")
        st.write(f"**Last data date used:** {last_date.strftime('%Y-%m-%d') if hasattr(last_date, 'strftime') else str(last_date)}")
        lstm_avg = np.mean(lstm_preds)
        if np.isnan(lstm_avg):
            st.metric("30-Day Avg Prediction (LSTM)", "N/A")
        else:
            st.metric("30-Day Avg Prediction (LSTM)", f"₹{lstm_avg:.2f}")
        st.metric("30-Day Avg Prediction (Prophet)", f"₹{np.mean(prophet_preds):.2f}")

        # Show company info if available
        stock = yf.Ticker(stock_name)
        info = stock.info
        with st.expander("Show Company Info"):
            st.write({k: info[k] for k in ["longName", "sector", "industry", "marketCap", "trailingPE", "dividendYield", "website"] if k in info})
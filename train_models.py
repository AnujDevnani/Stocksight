import os
import joblib
import numpy as np
import pandas as pd
from stock_predictor import get_stock_data, add_technical_indicators, get_fundamental_features, prepare_ml_features
from stock_predictor import predict_with_lstm, predict_with_prophet
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from stock_predictor import train_multistep_lstm, predict_multistep_lstm, train_multistep_xgboost, predict_multistep_xgboost, advanced_feature_engineering
from stock_predictor import train_multistep_lightgbm, train_multistep_catboost, train_arima
import argparse
import pickle
import keras
import tensorflow as tf
print('Keras version:', keras.__version__)
print('TensorFlow version:', tf.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('--stock', type=str, required=True, help='Stock symbol, e.g. TATAMOTORS.NS')
parser.add_argument('--data', type=str, default=None, help='Optional path to CSV data file')
args = parser.parse_args()

STOCK = args.stock
MODEL_DIR = f"models_{STOCK.replace('.', '_')}"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_lstm(data):
    # Use new multi-step LSTM
    model, scaler = train_multistep_lstm(data, epochs=100)
    # Save in new Keras v3 format if available
    try:
        model.save(os.path.join(MODEL_DIR, 'lstm_multistep_model.keras'), save_format='keras_v3')
    except TypeError:
        # Fallback for older Keras versions
        model.save(os.path.join(MODEL_DIR, 'lstm_multistep_model.keras'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'lstm_multistep_scaler.pkl'))

def train_prophet(data):
    from prophet import Prophet
    df = data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    model = Prophet()
    model.fit(df)
    with open(os.path.join(MODEL_DIR, 'prophet_model.pkl'), 'wb') as f:
        pickle.dump(model, f)

def train_xgboost(data):
    # Use new multi-output XGBoost
    model = train_multistep_xgboost(data, STOCK)
    joblib.dump(model, os.path.join(MODEL_DIR, 'xgb_multistep_model.pkl'))

def train_stacking_model(data):
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LinearRegression
    import joblib
    # Prepare features
    df = advanced_feature_engineering(data, STOCK)
    X = df.drop(['Close'], axis=1, errors='ignore')
    print("Stacking model features:", list(X.columns))
    y = df['Close']
    n = len(df)
    n_outputs = 30  # number of days predicted by XGBoost and others
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    lstm_preds = np.zeros(n)
    prophet_preds = np.zeros(n)
    xgb_preds = np.zeros((n, n_outputs))
    lgbm_preds = np.zeros((n, n_outputs))
    catboost_preds = np.zeros((n, n_outputs))
    arima_preds = np.zeros((n, n_outputs))
    # Load base models
    from tensorflow.keras.models import load_model
    lstm_model = load_model(os.path.join(MODEL_DIR, 'lstm_multistep_model.keras'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'lstm_multistep_scaler.pkl'))
    with open(os.path.join(MODEL_DIR, 'prophet_model.pkl'), 'rb') as f:
        prophet_model = pickle.load(f)
    xgb_model = joblib.load(os.path.join(MODEL_DIR, 'xgb_multistep_model.pkl'))
    # Optional models
    try:
        lgbm_model = joblib.load(os.path.join(MODEL_DIR, 'lgbm_multistep_model.pkl'))
    except Exception:
        lgbm_model = None
    try:
        catboost_model = joblib.load(os.path.join(MODEL_DIR, 'catboost_multistep_model.pkl'))
    except Exception:
        catboost_model = None
    try:
        with open(os.path.join(MODEL_DIR, 'arima_model.pkl'), 'rb') as f:
            arima_model = pickle.load(f)
    except Exception:
        arima_model = None
    # Generate out-of-fold predictions
    for train_idx, test_idx in kf.split(X):
        # LSTM
        close_scaled = scaler.transform(y.values.reshape(-1,1))
        for i in test_idx:
            if i < 60:
                continue
            seq = close_scaled[i-60:i].reshape(1, 60, 1)
            lstm_preds[i] = lstm_model.predict(seq, verbose=0)[0][0]
        # Prophet
        df_reset = df.reset_index()
        if 'Date' in df_reset.columns:
            df_prophet = df_reset[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        else:
            df_prophet = df_reset[['Close']]
            df_prophet['ds'] = pd.date_range(end=pd.Timestamp.today(), periods=len(df_prophet))
            df_prophet = df_prophet[['ds', 'Close']].rename(columns={'Close': 'y'})
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds']).dt.tz_localize(None)
        prophet_forecast = prophet_model.predict(df_prophet[['ds']])
        prophet_preds[test_idx] = prophet_forecast['yhat'].values[test_idx]
        # XGBoost
        xgb_preds[test_idx, :] = xgb_model.predict(X.iloc[test_idx])
        # LightGBM
        if lgbm_model is not None:
            lgbm_preds[test_idx, :] = lgbm_model.predict(X.iloc[test_idx])
        # CatBoost
        if catboost_model is not None:
            catboost_preds[test_idx, :] = catboost_model.predict(X.iloc[test_idx])
        # ARIMA (use zeros if not available)
        # (ARIMA is not a multioutput regressor, so we skip out-of-fold for ARIMA here)
    # For stacking, use the first day ahead from each model
    stack_X = np.vstack([
        lstm_preds,
        prophet_preds,
        xgb_preds[:, 0],
        lgbm_preds[:, 0],
        catboost_preds[:, 0],
        arima_preds[:, 0]  # will be zeros if not available
    ]).T
    stack_X = np.nan_to_num(stack_X)
    meta_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
    meta_model.fit(stack_X, y)
    joblib.dump(meta_model, os.path.join(MODEL_DIR, 'stacking_model.pkl'))

if __name__ == '__main__':
    if args.data:
        data = pd.read_csv(args.data, parse_dates=[0])
    else:
        data = get_stock_data(STOCK)
    data = advanced_feature_engineering(data, STOCK)
    print('Training LSTM...')
    train_lstm(data)
    print('Training Prophet...')
    train_prophet(data)
    print('Training XGBoost...')
    train_xgboost(data)
    print('Training LightGBM...')
    lgbm_model = train_multistep_lightgbm(data, STOCK)
    if lgbm_model is not None:
        joblib.dump(lgbm_model, os.path.join(MODEL_DIR, 'lgbm_multistep_model.pkl'))
    else:
        print('LightGBM model not trained (package missing or error).')
    print('Training CatBoost...')
    catboost_model = train_multistep_catboost(data, STOCK)
    if catboost_model is not None:
        joblib.dump(catboost_model, os.path.join(MODEL_DIR, 'catboost_multistep_model.pkl'))
    else:
        print('CatBoost model not trained (package missing or error).')
    print('Training ARIMA...')
    arima_model = train_arima(data)
    if arima_model is not None:
        with open(os.path.join(MODEL_DIR, 'arima_model.pkl'), 'wb') as f:
            pickle.dump(arima_model, f)
    else:
        print('ARIMA model not trained (error or insufficient data).')
    print('Training Stacking Meta-Model...')
    train_stacking_model(data)
    print('All models trained and saved!') 
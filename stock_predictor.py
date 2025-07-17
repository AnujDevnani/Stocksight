import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from newsapi import NewsApiClient
from transformers import pipeline
import ta
import xgboost as xgb
import shap
import joblib
import pickle
from tensorflow.keras.models import load_model
import requests
from tensorflow.keras import Input
# Remove global MODEL_DIR
# All functions that load models should take model_dir as an argument

# 🔍 Fetch Stock Data (NSE: Use ".NS", e.g., "RELIANCE.NS")
def get_stock_data(stock_name, years=4):
    stock = yf.Ticker(stock_name)
    data = stock.history(period=f"{years}y")
    return data

# 📈 LSTM Model (Deep Learning for Prediction)
def predict_with_lstm(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
    
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32)
    
    # Predict next 30 days
    last_60_days = scaled_data[-60:]
    future_preds = []
    for _ in range(30):
        x_input = last_60_days.reshape(1, 60, 1)
        pred = model.predict(x_input)
        future_preds.append(pred[0])
        last_60_days = np.append(last_60_days[1:], pred)
    
    future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1,1))
    return future_preds.flatten()

# 🔮 Prophet Model (Facebook's Time Series Forecasting)
def predict_with_prophet(data):
    df = data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    # Remove timezone info from 'ds' column
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast.tail(30)['yhat'].values

# 📰 News Sentiment Analysis (BERT)
def analyze_sentiment(stock_name):
    newsapi = NewsApiClient(api_key='9bad87c13e404eb3a2d871124acd652e')  # Get free API key from newsapi.org
    news = newsapi.get_everything(q=stock_name, language='en', sort_by='publishedAt')
    
    sentiment_pipeline = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
    headlines = [article['title'] for article in news['articles']]
    sentiments = [sentiment_pipeline(headline)[0] for headline in headlines]
    
    # Prepare list of (headline, label, score)
    headline_sentiments = [
        (headline, s['label'], s['score']) for headline, s in zip(headlines, sentiments)
    ]
    
    avg_sentiment = np.mean([
        s['score'] if s['label'] == 'POS' else -s['score'] if s['label'] == 'NEG' else 0
        for s in sentiments
    ])
    return avg_sentiment, headline_sentiments

# 💡 Generate Buy/Sell Recommendation
def get_recommendation(lstm_pred, prophet_pred, sentiment):
    avg_pred = (np.mean(lstm_pred) + np.mean(prophet_pred)) / 2
    if avg_pred > 0 and sentiment > 0.5:
        return "STRONG BUY 🚀"
    elif avg_pred > 0 and sentiment > 0:
        return "BUY 📈"
    elif avg_pred < 0 and sentiment < 0:
        return "SELL 📉"
    else:
        return "HOLD ⏳"

# Calculate technical indicators

def add_technical_indicators(data):
    df = data.copy()
    import ta
    # Existing indicators
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['EMA20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
    df['EMA50'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()
    bb = ta.volatility.BollingerBands(df['Close'], window=20)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    # More indicators
    df['Stoch_K'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
    df['Stoch_D'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch_signal()
    df['WilliamsR'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()
    df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
    df['SAR'] = ta.trend.PSARIndicator(df['High'], df['Low'], df['Close']).psar()
    df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume']).chaikin_money_flow()
    df['A/D'] = ta.volume.AccDistIndexIndicator(df['High'], df['Low'], df['Close'], df['Volume']).acc_dist_index()
    df['Donchian_High'] = ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close'], window=20).donchian_channel_hband()
    df['Donchian_Low'] = ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close'], window=20).donchian_channel_lband()
    df['StdDev'] = df['Close'].rolling(window=20).std()
    # Lagged returns
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_5d'] = df['Close'].pct_change(5)
    df['Return_21d'] = df['Close'].pct_change(21)
    # Rolling min/max
    df['RollMin_5'] = df['Close'].rolling(window=5).min()
    df['RollMax_5'] = df['Close'].rolling(window=5).max()
    df['RollMin_21'] = df['Close'].rolling(window=21).min()
    df['RollMax_21'] = df['Close'].rolling(window=21).max()
    # Rolling std (volatility)
    df['RollStd_5'] = df['Close'].rolling(window=5).std()
    df['RollStd_21'] = df['Close'].rolling(window=21).std()
    # Realized volatility (21d)
    df['RealizedVol_21'] = df['Return_1d'].rolling(window=21).std() * (252**0.5)
    # Price action patterns (basic candlestick)
    df['Hammer'] = ((df['High']-df['Low'] > 3*(df['Open']-df['Close'])) & ((df['Close']-df['Low'])/(.001+df['High']-df['Low']) > 0.6) & ((df['Open']-df['Low'])/(.001+df['High']-df['Low']) > 0.6)).astype(int)
    df['Doji'] = (abs(df['Close']-df['Open']) <= (df['High']-df['Low'])*0.1).astype(int)
    df['Engulfing'] = ((df['Close'].shift(1) < df['Open'].shift(1)) & (df['Close'] > df['Open']) & (df['Close'] > df['Open'].shift(1)) & (df['Open'] < df['Close'].shift(1))).astype(int)
    # Gap up/down
    df['GapUp'] = (df['Open'] > df['Close'].shift(1)).astype(int)
    df['GapDown'] = (df['Open'] < df['Close'].shift(1)).astype(int)
    return df

def get_macro_features():
    # Example: Use FRED API for US macro data (can be adapted for India if available)
    # FRED series: FEDFUNDS (interest rate), CPIAUCSL (CPI), GDP (GDP)
    # For India, you can use RBI or World Bank APIs if available
    try:
        # US Federal Funds Rate (as a proxy, replace with Indian repo rate if available)
        r = requests.get('https://api.stlouisfed.org/fred/series/observations', params={
            'series_id': 'FEDFUNDS', 'api_key': 'YOUR_FRED_API_KEY', 'file_type': 'json', 'sort_order': 'desc', 'limit': 1
        })
        interest_rate = float(r.json()['observations'][0]['value']) if r.ok else None
    except Exception:
        interest_rate = None
    try:
        r = requests.get('https://api.stlouisfed.org/fred/series/observations', params={
            'series_id': 'CPIAUCSL', 'api_key': 'YOUR_FRED_API_KEY', 'file_type': 'json', 'sort_order': 'desc', 'limit': 1
        })
        inflation = float(r.json()['observations'][0]['value']) if r.ok else None
    except Exception:
        inflation = None
    try:
        r = requests.get('https://api.stlouisfed.org/fred/series/observations', params={
            'series_id': 'GDP', 'api_key': 'YOUR_FRED_API_KEY', 'file_type': 'json', 'sort_order': 'desc', 'limit': 1
        })
        gdp = float(r.json()['observations'][0]['value']) if r.ok else None
    except Exception:
        gdp = None
    return {'interest_rate': interest_rate, 'inflation': inflation, 'gdp': gdp}

def get_twitter_sentiment(stock_name):
    try:
        import snscrape.modules.twitter as sntwitter
        from transformers import pipeline
        sentiment_pipeline = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
        tweets = []
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{stock_name} since:2023-01-01').get_items()):
            if i > 20:
                break
            tweets.append(tweet.content)
        if not tweets:
            return 0.0
        sentiments = [sentiment_pipeline(t)[0] for t in tweets]
        avg_sentiment = np.mean([
            s['score'] if s['label'] == 'POS' else -s['score'] if s['label'] == 'NEG' else 0
            for s in sentiments
        ])
        return avg_sentiment
    except Exception:
        return 0.0

# Update get_fundamental_features to include sentiment

def get_fundamental_features(stock_name):
    import yfinance as yf
    stock = yf.Ticker(stock_name)
    info = stock.info
    features = {}
    # List of all required features
    required_fields = [
        'interest_rate', 'debtToEquity', 'gdp', 'grossMargins', 'currentRatio', 'revenueGrowth',
        'returnOnEquity', 'earningsQuarterlyGrowth', 'priceToBook', 'marketCap', 'trailingPE',
        'pegRatio', 'operatingMargins', 'dividendYield', 'twitter_sentiment', 'quickRatio',
        'priceToSalesTrailing12Months', 'profitMargins', 'earningsGrowth', 'forwardPE',
        'inflation', 'returnOnAssets'
    ]
    # Map yfinance fields to our required fields where needed
    yfinance_map = {
        'debtToEquity': 'debtToEquity',
        'grossMargins': 'grossMargins',
        'currentRatio': 'currentRatio',
        'revenueGrowth': 'revenueGrowth',
        'returnOnEquity': 'returnOnEquity',
        'earningsQuarterlyGrowth': 'earningsQuarterlyGrowth',
        'priceToBook': 'priceToBook',
        'marketCap': 'marketCap',
        'trailingPE': 'trailingPE',
        'pegRatio': 'pegRatio',
        'operatingMargins': 'operatingMargins',
        'dividendYield': 'dividendYield',
        'quickRatio': 'quickRatio',
        'priceToSalesTrailing12Months': 'priceToSalesTrailing12Months',
        'profitMargins': 'profitMargins',
        'earningsGrowth': 'earningsGrowth',
        'forwardPE': 'forwardPE',
        'returnOnAssets': 'returnOnAssets',
    }
    # Add yfinance fields
    for req, yf_key in yfinance_map.items():
        features[req] = info.get(yf_key, None)
    # Add macro features
    macro = get_macro_features()
    features['interest_rate'] = macro.get('interest_rate', None)
    features['gdp'] = macro.get('gdp', None)
    features['inflation'] = macro.get('inflation', None)
    # Add Twitter sentiment
    features['twitter_sentiment'] = get_twitter_sentiment(stock_name)
    # Ensure all required fields are present
    for field in required_fields:
        if field not in features:
            features[field] = None
    return features

# Get sector/industry average returns (if available)
def get_sector_industry_comparison(stock_name):
    import yfinance as yf
    stock = yf.Ticker(stock_name)
    info = stock.info
    sector = info.get('sector', None)
    industry = info.get('industry', None)
    # This is a placeholder: in practice, you would fetch a list of tickers in the same sector/industry and average their returns
    # For now, just return sector and industry names
    return sector, industry

def prepare_ml_features(data, stock_name):
    # Add technical indicators
    df = add_technical_indicators(data)
    # Add fundamental features
    fund = get_fundamental_features(stock_name)
    for k, v in fund.items():
        df[k] = v
    # Convert all columns to numeric, coerce errors to NaN
    import pandas as pd
    df = df.apply(pd.to_numeric, errors='coerce')
    # Fill missing values with forward and backward fill
    df = df.fillna(method='bfill').fillna(method='ffill')
    return df

def predict_with_xgboost(data, stock_name):
    df = prepare_ml_features(data, stock_name)
    if df.shape[0] < 2:
        # Not enough data to train and predict
        return float('nan'), []
    # Use last 60 days for training, predict next day
    X = df.drop(['Close'], axis=1, errors='ignore')
    y = df['Close'] if 'Close' in df else None
    # Train/test split: last row for prediction
    X_train, X_pred = X.iloc[:-1], X.iloc[[-1]]
    y_train = y.iloc[:-1] if y is not None else None
    model = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_pred)[0]
    # SHAP explainability
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_pred)
    # Get top 5 features
    feature_importance = sorted(zip(X_pred.columns, shap_values.values[0]), key=lambda x: abs(x[1]), reverse=True)[:5]
    return pred, feature_importance

def get_ensemble_prediction_and_explanation(stock_name, data):
    # LSTM prediction (next day)
    lstm_pred = predict_with_lstm(data)[0]
    # Prophet prediction (next day)
    prophet_pred = predict_with_prophet(data)[0]
    # XGBoost prediction and SHAP
    xgb_pred, shap_feats = predict_with_xgboost(data, stock_name)
    # Ensemble: average
    ensemble_pred = (lstm_pred + prophet_pred + xgb_pred) / 3
    return ensemble_pred, xgb_pred, shap_feats

# Load and predict next 30 days with LSTM

def predict_lstm_30days(data, model_dir):
    scaler = joblib.load(f'{model_dir}/lstm_scaler.pkl')
    model = load_model(f'{model_dir}/lstm_model.keras')
    # Use only the 'Close' column for scaling and prediction
    close_prices = data['Close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(close_prices)
    last_60_days = scaled_data[-60:].copy()
    future_preds = []
    for i in range(30):
        x_input = last_60_days.reshape(1, 60, 1)
        pred_scaled = model.predict(x_input, verbose=0)
        # Inverse transform to get the predicted price
        pred = scaler.inverse_transform(pred_scaled)[0][0]
        future_preds.append(pred)
        # Update the sequence: scale the new prediction and append
        pred_scaled_for_seq = scaler.transform(np.array([[pred]]))
        last_60_days = np.append(last_60_days[1:], pred_scaled_for_seq).reshape(60, 1)
        # Debug output
        # print(f"LSTM Day {i+1}: Predicted={pred}")
    return np.array(future_preds)

# Fix Prophet 30-day prediction to ensure continuous future dates

def predict_prophet_30days(data, model_dir):
    with open(f'{model_dir}/prophet_model.pkl', 'rb') as f:
        model = pickle.load(f)
    df = data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    # Ensure daily frequency and no missing dates
    df = df.set_index('ds').asfreq('D').reset_index()
    df['y'] = df['y'].interpolate()  # Fill missing values if any
    # model.fit(df)  # <-- Removed to avoid Prophet refit error
    future = model.make_future_dataframe(periods=30, freq='D')
    forecast = model.predict(future)
    preds = forecast.tail(30)['yhat'].values
    # Debug output
    # print(f"Prophet 30-day predictions: {preds}")
    return preds

# Load and predict next 30 days with XGBoost (rolling window)

def predict_xgboost_30days(data, stock_name, model_dir):
    model = joblib.load(f'{model_dir}/xgb_model.pkl')
    # Start with the original data
    df_orig = data.copy()
    preds = []
    for _ in range(30):
        # Prepare features for the current data
        df_feat = prepare_ml_features(df_orig, stock_name)
        X_pred = df_feat.drop(['Close'], axis=1, errors='ignore').iloc[[-1]]
        pred = model.predict(X_pred)[0]
        preds.append(pred)
        # Append the new prediction as the next day's close
        new_row = df_orig.iloc[[-1]].copy()
        new_row['Close'] = pred
        # Increment the date for the new row
        if 'Date' in new_row.columns:
            new_row['Date'] = new_row['Date'] + pd.Timedelta(days=1)
        df_orig = pd.concat([df_orig, new_row], ignore_index=True)
    return np.array(preds)

# Get ensemble of all 30-day predictions

def get_stacking_30day_predictions(stock_name, data, model_dir):
    import joblib
    import os
    import numpy as np
    # LSTM
    lstm_model = load_lstm_model_safe(f'{model_dir}/lstm_multistep_model.keras')
    if lstm_model is not None:
        lstm_scaler = joblib.load(f'{model_dir}/lstm_multistep_scaler.pkl')
        lstm_preds = predict_multistep_lstm(data, lstm_model, lstm_scaler)
    else:
        lstm_preds = [np.nan] * 30  # Use np.nan instead of 'N/A'
    # Prophet
    prophet_preds = predict_prophet_30days(data, model_dir)
    # XGBoost
    xgb_model = joblib.load(f'{model_dir}/xgb_multistep_model.pkl')
    xgb_preds = predict_multistep_xgboost(data, stock_name, xgb_model)
    # LightGBM
    try:
        lgbm_model = joblib.load(f'{model_dir}/lgbm_multistep_model.pkl')
    except Exception:
        lgbm_model = None
    lgbm_preds = predict_multistep_lightgbm(data, stock_name, lgbm_model)
    # CatBoost
    try:
        catboost_model = joblib.load(f'{model_dir}/catboost_multistep_model.pkl')
    except Exception:
        catboost_model = None
    catboost_preds = predict_multistep_catboost(data, stock_name, catboost_model)
    # ARIMA
    try:
        import pickle
        with open(f'{model_dir}/arima_model.pkl', 'rb') as f:
            arima_model = pickle.load(f)
    except Exception:
        arima_model = None
    arima_preds = predict_arima(data, arima_model) if arima_model is not None else np.zeros(30)
    # Stacking model
    stacking_model_path = f'{model_dir}/stacking_model.pkl'
    if not os.path.exists(stacking_model_path):
        raise FileNotFoundError(f"Stacking model not found for {stock_name}. Please retrain the models for this stock.")
    stacking_model = joblib.load(stacking_model_path)
    stack_X = np.vstack([lstm_preds, prophet_preds, xgb_preds, lgbm_preds, catboost_preds, arima_preds]).T
    # Ensure all values are float, replace non-numeric with np.nan, then nan_to_num
    def to_float(x):
        try:
            return float(x)
        except Exception:
            return np.nan
    stack_X = np.vectorize(to_float)(stack_X)
    stack_X = np.nan_to_num(stack_X)
    stacking_preds = stacking_model.predict(stack_X)
    return lstm_preds, prophet_preds, xgb_preds, lgbm_preds, catboost_preds, arima_preds, stacking_preds

# Update get_ensemble_30day_predictions to use new stacking logic

def get_ensemble_30day_predictions(stock_name, data):
    model_dir = f"models_{stock_name.replace('.', '_')}"
    return get_stacking_30day_predictions(stock_name, data, model_dir)

def get_xgboost_shap_for_last_day(data, stock_name):
    model_dir = f"models_{stock_name.replace('.', '_')}"
    import joblib
    import shap
    import os
    model_path = f'{model_dir}/xgb_multistep_model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'XGBoost model not found at {model_path}. Please retrain the models.')
    model = joblib.load(model_path)
    # Use advanced_feature_engineering to match training
    df = advanced_feature_engineering(data, stock_name)
    X_pred = df.drop(['Close'], axis=1, errors='ignore').iloc[[-1]]
    # Debug: print feature names
    try:
        model_features = model.estimators_[0].feature_names_in_
    except Exception:
        model_features = []
    print('Model features:', model_features)
    print('Prediction features:', list(X_pred.columns))
    # Try prediction, catch feature mismatch
    try:
        pred = model.predict(X_pred)[0]
    except Exception as e:
        print('Feature mismatch error:', e)
        raise
    # SHAP explainability: use the first base regressor
    base_model = model.estimators_[0]
    explainer = shap.Explainer(base_model, df.drop(['Close'], axis=1, errors='ignore'))
    shap_values = explainer(X_pred)
    feature_importance = sorted(zip(X_pred.columns, shap_values.values[0]), key=lambda x: abs(x[1]), reverse=True)[:5]
    return pred, feature_importance

def add_market_index_features(data, index_symbol='^NSEI'):
    # Add NIFTY index returns as a feature
    try:
        index = yf.Ticker(index_symbol)
        index_data = index.history(start=data.index.min(), end=data.index.max())
        index_data = index_data['Close'].pct_change().rename('NIFTY_Return')
        data = data.copy()
        data['NIFTY_Return'] = index_data.reindex(data.index, method='ffill')
    except Exception:
        data['NIFTY_Return'] = 0.0
    return data

def add_peer_features(data, peer_symbols=['TATAMOTORS.NS', 'RELIANCE.NS']):
    # Add peer stock returns as features
    for peer in peer_symbols:
        if peer not in data.columns:
            try:
                peer_data = yf.Ticker(peer).history(start=data.index.min(), end=data.index.max())
                data[f'{peer}_Return'] = peer_data['Close'].pct_change().reindex(data.index, method='ffill')
            except Exception:
                data[f'{peer}_Return'] = 0.0
    return data

def add_calendar_features(data):
    data = data.copy()
    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month
    data['is_month_start'] = data.index.is_month_start.astype(int)
    data['is_month_end'] = data.index.is_month_end.astype(int)
    return data

def clean_data(data):
    # Remove/cap outliers and smooth noisy features
    data = data.copy()
    for col in data.select_dtypes(include=[float, int]).columns:
        data[col] = data[col].clip(lower=data[col].quantile(0.01), upper=data[col].quantile(0.99))
        data[col] = data[col].rolling(window=3, min_periods=1).mean()
    return data

def add_rolling_sentiment(data, stock_name):
    # Add rolling sentiment as a feature (7-day rolling mean)
    avg_sentiment, headline_sentiments = analyze_sentiment(stock_name.split('.')[0])
    sentiment_series = pd.Series(
        [s[2] if s[1]=='POS' else -s[2] if s[1]=='NEG' else 0 for s in headline_sentiments],
        index=pd.date_range(end=pd.Timestamp.today(), periods=len(headline_sentiments))
    )
    # Ensure both indices are timezone-naive
    sentiment_series.index = sentiment_series.index.tz_localize(None)
    data = data.copy()
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    data['rolling_sentiment'] = sentiment_series.rolling(window=7, min_periods=1).mean().reindex(data.index, method='ffill').fillna(0)
    return data

def add_advanced_candlestick_patterns(data):
    df = data.copy()
    # Morning Star
    df['Morning_Star'] = (
        (df['Close'].shift(2) < df['Open'].shift(2)) &
        (abs(df['Close'].shift(1) - df['Open'].shift(1)) / (df['High'].shift(1) - df['Low'].shift(1) + 1e-9) < 0.1) &
        (df['Close'] > df['Open']) &
        (df['Close'] > df['Close'].shift(2))
    ).astype(int)
    # Evening Star
    df['Evening_Star'] = (
        (df['Close'].shift(2) > df['Open'].shift(2)) &
        (abs(df['Close'].shift(1) - df['Open'].shift(1)) / (df['High'].shift(1) - df['Low'].shift(1) + 1e-9) < 0.1) &
        (df['Close'] < df['Open']) &
        (df['Close'] < df['Close'].shift(2))
    ).astype(int)
    # Three White Soldiers
    df['Three_White_Soldiers'] = (
        (df['Close'].shift(2) < df['Open'].shift(2)) &
        (df['Close'].shift(1) > df['Open'].shift(1)) &
        (df['Close'] > df['Open']) &
        (df['Close'].shift(1) > df['Close'].shift(2)) &
        (df['Close'] > df['Close'].shift(1))
    ).astype(int)
    # Three Black Crows
    df['Three_Black_Crows'] = (
        (df['Close'].shift(2) > df['Open'].shift(2)) &
        (df['Close'].shift(1) < df['Open'].shift(1)) &
        (df['Close'] < df['Open']) &
        (df['Close'].shift(1) < df['Close'].shift(2)) &
        (df['Close'] < df['Close'].shift(1))
    ).astype(int)
    return df

def add_ichimoku_cloud(data):
    df = data.copy()
    high_9 = df['High'].rolling(window=9).max()
    low_9 = df['Low'].rolling(window=9).min()
    df['tenkan_sen'] = (high_9 + low_9) / 2
    high_26 = df['High'].rolling(window=26).max()
    low_26 = df['Low'].rolling(window=26).min()
    df['kijun_sen'] = (high_26 + low_26) / 2
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    high_52 = df['High'].rolling(window=52).max()
    low_52 = df['Low'].rolling(window=52).min()
    df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
    df['chikou_span'] = df['Close'].shift(-26)
    return df

def add_keltner_channels(data):
    df = data.copy()
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    ema = typical_price.ewm(span=20, adjust=False).mean()
    atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    df['KC_Middle'] = ema
    df['KC_Upper'] = ema + 2 * atr
    df['KC_Lower'] = ema - 2 * atr
    return df

def add_fractals(data):
    df = data.copy()
    df['Fractal_Up'] = ((df['High'].shift(2) < df['High'].shift(0)) &
                        (df['High'].shift(1) < df['High'].shift(0)) &
                        (df['High'].shift(-1) < df['High'].shift(0)) &
                        (df['High'].shift(-2) < df['High'].shift(0))).astype(int)
    df['Fractal_Down'] = ((df['Low'].shift(2) > df['Low'].shift(0)) &
                          (df['Low'].shift(1) > df['Low'].shift(0)) &
                          (df['Low'].shift(-1) > df['Low'].shift(0)) &
                          (df['Low'].shift(-2) > df['Low'].shift(0))).astype(int)
    return df

def add_advanced_fundamental_features(data, stock_name):
    import yfinance as yf
    df = data.copy()
    stock = yf.Ticker(stock_name)
    # YoY revenue and EPS growth
    try:
        fin = stock.financials.T
        fin = fin.sort_index()
        df['YoY_Revenue_Growth'] = fin['Total Revenue'].pct_change().reindex(df.index, method='ffill')
        df['YoY_EPS_Growth'] = fin['Basic EPS'].pct_change().reindex(df.index, method='ffill')
    except Exception:
        df['YoY_Revenue_Growth'] = 0.0
        df['YoY_EPS_Growth'] = 0.0
    # Analyst target price and consensus
    try:
        info = stock.info
        df['Analyst_Target_Price'] = info.get('targetMeanPrice', 0.0)
        df['Analyst_Consensus'] = info.get('recommendationKey', 'none')
        # Convert consensus to ordinal
        consensus_map = {'strong_buy': 2, 'buy': 1, 'hold': 0, 'sell': -1, 'strong_sell': -2, 'none': 0}
        df['Analyst_Consensus'] = df['Analyst_Consensus'].map(lambda x: consensus_map.get(str(x).lower(), 0))
    except Exception:
        df['Analyst_Target_Price'] = 0.0
        df['Analyst_Consensus'] = 0
    # Insider trading activity (net shares bought/sold in last year)
    try:
        cal = stock.get_calendar()
        if 'Insider Transactions' in cal:
            df['Insider_Activity'] = cal['Insider Transactions'].sum()
        else:
            df['Insider_Activity'] = 0.0
    except Exception:
        df['Insider_Activity'] = 0.0
    return df

def add_google_trends_feature(data, stock_name):
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl='en-US', tz=330)
        kw_list = [stock_name.split('.')[0]]
        pytrends.build_payload(kw_list, timeframe='today 5-y')
        trends = pytrends.interest_over_time()
        if not trends.empty:
            trends = trends[kw_list[0]].reindex(data.index, method='ffill')
            data = data.copy()
            data['GoogleTrends'] = trends
        else:
            data['GoogleTrends'] = 0.0
    except Exception:
        data['GoogleTrends'] = 0.0
    return data

def add_reddit_sentiment_feature(data, stock_name):
    try:
        import snscrape.modules.reddit as snreddit
        from transformers import pipeline
        sentiment_pipeline = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
        posts = []
        for i, post in enumerate(snreddit.RedditSearchScraper(f'{stock_name}').get_items()):
            if i > 20:
                break
            posts.append(post.content)
        if not posts:
            data['RedditSentiment'] = 0.0
        else:
            sentiments = [sentiment_pipeline(t)[0] for t in posts]
            avg_sentiment = np.mean([
                s['score'] if s['label'] == 'POS' else -s['score'] if s['label'] == 'NEG' else 0
                for s in sentiments
            ])
            data['RedditSentiment'] = avg_sentiment
    except Exception:
        data['RedditSentiment'] = 0.0
    return data

def add_news_volume_feature(data, stock_name):
    try:
        newsapi = NewsApiClient(api_key='9bad87c13e404eb3a2d871124acd652e')
        news = newsapi.get_everything(q=stock_name, language='en', sort_by='publishedAt', page_size=100)
        articles = news['articles']
        # Count articles per day
        news_dates = pd.to_datetime([a['publishedAt'][:10] for a in articles])
        news_counts = pd.Series(news_dates).value_counts().sort_index()
        news_counts = news_counts.reindex(data.index, fill_value=0)
        data = data.copy()
        data['NewsVolume'] = news_counts
    except Exception:
        data['NewsVolume'] = 0
    return data

def add_advanced_macro_features(data):
    df = data.copy()
    # Unemployment rate, PMI, exchange rate, global indices, oil/gold price
    try:
        # Example: Use FRED API for US/Global data (can be adapted for India)
        import requests
        # Unemployment (USUNRATE)
        r = requests.get('https://api.stlouisfed.org/fred/series/observations', params={
            'series_id': 'UNRATE', 'api_key': 'YOUR_FRED_API_KEY', 'file_type': 'json', 'sort_order': 'desc', 'limit': 1
        })
        df['Unemployment'] = float(r.json()['observations'][0]['value']) if r.ok else 0.0
        # PMI (NAPM)
        r = requests.get('https://api.stlouisfed.org/fred/series/observations', params={
            'series_id': 'NAPM', 'api_key': 'YOUR_FRED_API_KEY', 'file_type': 'json', 'sort_order': 'desc', 'limit': 1
        })
        df['PMI'] = float(r.json()['observations'][0]['value']) if r.ok else 0.0
        # Exchange rate (DEXINUS)
        r = requests.get('https://api.stlouisfed.org/fred/series/observations', params={
            'series_id': 'DEXINUS', 'api_key': 'YOUR_FRED_API_KEY', 'file_type': 'json', 'sort_order': 'desc', 'limit': 1
        })
        df['USDINR'] = float(r.json()['observations'][0]['value']) if r.ok else 0.0
        # S&P 500 (SP500)
        r = requests.get('https://api.stlouisfed.org/fred/series/observations', params={
            'series_id': 'SP500', 'api_key': 'YOUR_FRED_API_KEY', 'file_type': 'json', 'sort_order': 'desc', 'limit': 1
        })
        df['SP500'] = float(r.json()['observations'][0]['value']) if r.ok else 0.0
        # Oil price (DCOILWTICO)
        r = requests.get('https://api.stlouisfed.org/fred/series/observations', params={
            'series_id': 'DCOILWTICO', 'api_key': 'YOUR_FRED_API_KEY', 'file_type': 'json', 'sort_order': 'desc', 'limit': 1
        })
        df['OilPrice'] = float(r.json()['observations'][0]['value']) if r.ok else 0.0
        # Gold price (GOLDAMGBD228NLBM)
        r = requests.get('https://api.stlouisfed.org/fred/series/observations', params={
            'series_id': 'GOLDAMGBD228NLBM', 'api_key': 'YOUR_FRED_API_KEY', 'file_type': 'json', 'sort_order': 'desc', 'limit': 1
        })
        df['GoldPrice'] = float(r.json()['observations'][0]['value']) if r.ok else 0.0
    except Exception:
        df['Unemployment'] = 0.0
        df['PMI'] = 0.0
        df['USDINR'] = 0.0
        df['SP500'] = 0.0
        df['OilPrice'] = 0.0
        df['GoldPrice'] = 0.0
    return df

# Update advanced_feature_engineering to include all new features
def advanced_feature_engineering(data, stock_name):
    data = add_technical_indicators(data)
    data = add_market_index_features(data)
    data = add_peer_features(data)
    data = add_sector_etf_and_vix_features(data, stock_name)
    data = add_calendar_features(data)
    data = add_event_calendar_features(data, stock_name)
    data = add_rolling_sentiment(data, stock_name)
    data = add_google_trends_feature(data, stock_name)
    data = add_reddit_sentiment_feature(data, stock_name)
    data = add_news_volume_feature(data, stock_name)
    data = add_advanced_candlestick_patterns(data)
    data = add_ichimoku_cloud(data)
    data = add_keltner_channels(data)
    data = add_fractals(data)
    data = add_advanced_fundamental_features(data, stock_name)
    data = add_advanced_macro_features(data)
    data = clean_data(data)
    return data

# Multi-step LSTM (sequence-to-sequence) for 30-day prediction
from tensorflow.keras.layers import RepeatVector, TimeDistributed

def train_multistep_lstm(data, epochs=100):
    scaler = MinMaxScaler()
    close_scaled = scaler.fit_transform(data['Close'].values.reshape(-1,1))
    X, y = [], []
    n_steps_in, n_steps_out = 60, 30
    for i in range(n_steps_in, len(close_scaled)-n_steps_out):
        X.append(close_scaled[i-n_steps_in:i, 0])
        y.append(close_scaled[i:i+n_steps_out, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = y.reshape((y.shape[0], y.shape[1], 1))
    model = Sequential([
        Input(shape=(n_steps_in, 1)),
        LSTM(50, activation='relu'),
        RepeatVector(n_steps_out),
        LSTM(50, activation='relu', return_sequences=True),
        TimeDistributed(Dense(1))
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.2)
    return model, scaler

def predict_multistep_lstm(data, model, scaler):
    close_scaled = scaler.transform(data['Close'].values.reshape(-1,1))
    last_60_days = close_scaled[-60:].reshape(1, 60, 1)
    preds_scaled = model.predict(last_60_days)
    preds = scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()
    return preds

# Multi-output XGBoost for 30-day prediction
from sklearn.multioutput import MultiOutputRegressor

def train_multistep_xgboost(data, stock_name):
    df = advanced_feature_engineering(data, stock_name)
    X = df.drop(['Close'], axis=1, errors='ignore')
    y = []
    for i in range(30, len(df)):
        y.append(df['Close'].values[i-30:i])
    y = np.array(y)
    X = X.iloc[30:]
    model = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42))
    model.fit(X, y)
    return model

def predict_multistep_xgboost(data, stock_name, model):
    df = advanced_feature_engineering(data, stock_name)
    X_pred = df.drop(['Close'], axis=1, errors='ignore').iloc[[-1]]
    preds = model.predict(X_pred).flatten()
    return preds

# LightGBM, CatBoost, ARIMA imports
from sklearn.multioutput import MultiOutputRegressor
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None
from statsmodels.tsa.arima.model import ARIMA

def train_multistep_lightgbm(data, stock_name):
    df = advanced_feature_engineering(data, stock_name)
    X = df.drop(['Close'], axis=1, errors='ignore')
    y = []
    for i in range(30, len(df)):
        y.append(df['Close'].values[i-30:i])
    y = np.array(y)
    X = X.iloc[30:]
    if lgb is not None:
        model = MultiOutputRegressor(lgb.LGBMRegressor(n_estimators=100, random_state=42))
        model.fit(X, y)
        return model
    return None

def predict_multistep_lightgbm(data, stock_name, model):
    if model is None:
        print('LightGBM model is not trained or not loaded.')
        return None
    df = advanced_feature_engineering(data, stock_name)
    X_pred = df.drop(['Close'], axis=1, errors='ignore').iloc[[-1]]
    preds = model.predict(X_pred).flatten()
    return preds

def train_multistep_catboost(data, stock_name):
    df = advanced_feature_engineering(data, stock_name)
    X = df.drop(['Close'], axis=1, errors='ignore')
    y = []
    for i in range(30, len(df)):
        y.append(df['Close'].values[i-30:i])
    y = np.array(y)
    X = X.iloc[30:]
    if CatBoostRegressor is not None:
        model = MultiOutputRegressor(CatBoostRegressor(verbose=0, iterations=100, random_state=42))
        model.fit(X, y)
        return model
    return None

def predict_multistep_catboost(data, stock_name, model):
    if model is None:
        print('CatBoost model is not trained or not loaded.')
        return None
    df = advanced_feature_engineering(data, stock_name)
    X_pred = df.drop(['Close'], axis=1, errors='ignore').iloc[[-1]]
    preds = model.predict(X_pred).flatten()
    return preds

def train_arima(data):
    # Fit ARIMA on closing prices, forecast 30 days
    close = data['Close'].dropna()
    model = ARIMA(close, order=(5,1,0))
    model_fit = model.fit()
    return model_fit

def predict_arima(data, model_fit):
    if model_fit is None:
        print('ARIMA model is not trained or not loaded.')
        return None
    preds = model_fit.forecast(steps=30)
    return preds.values if hasattr(preds, 'values') else np.array(preds)

def add_sector_etf_and_vix_features(data, stock_name=None):
    """
    Adds sector ETF (NIFTYBEES.NS) and India VIX (^INDIAVIX) returns as features to the dataframe.
    """
    import yfinance as yf
    df = data.copy()
    # Fetch NIFTY ETF (broad market proxy)
    try:
        nifty = yf.Ticker('NIFTYBEES.NS').history(period='max')
        nifty = nifty['Close'].rename('NIFTY_Close')
        nifty = nifty.reindex(df.index, method='ffill')
        df['NIFTY_Return'] = nifty.pct_change()
    except Exception:
        df['NIFTY_Return'] = 0.0
    # Fetch India VIX
    try:
        vix = yf.Ticker('^INDIAVIX').history(period='max')
        vix = vix['Close'].rename('VIX_Close')
        vix = vix.reindex(df.index, method='ffill')
        df['VIX'] = vix
        df['VIX_Change'] = vix.pct_change()
    except Exception:
        df['VIX'] = 0.0
        df['VIX_Change'] = 0.0
    return df

def add_event_calendar_features(data, stock_name=None):
    """
    Adds event-based calendar features (earnings, holiday, budget_day, expiry_day) as binary columns. Placeholder: all zeros.
    """
    df = data.copy()
    df['Earnings'] = 0  # Placeholder: set to 1 on earnings dates if available
    df['Holiday'] = 0   # Placeholder: set to 1 on market holidays if available
    df['Budget_Day'] = 0  # Placeholder: set to 1 on Union Budget day
    df['Expiry_Day'] = 0  # Placeholder: set to 1 on F&O expiry day
    return df

def load_lstm_model_safe(model_path):
    from tensorflow.keras.models import load_model
    try:
        return load_model(model_path)
    except (OSError, ValueError) as e:
        print(f"LSTM model load failed: {e}")
        return None
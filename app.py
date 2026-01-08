"""
Bank Balance Forecasting API
XGBoost-based forecaster for bank transaction time series
Deployable on Render
"""

import os
import io
import json
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables to store trained model and data
trained_model = None
training_data = None
feature_names = []


class BankTransactionForecaster:
    """
    XGBoost-based forecaster for bank transaction time series
    """
    
    def __init__(self, horizon=7, lag_features=14):
        self.horizon = horizon
        self.lag_features = lag_features
        self.model = None
        self.feature_names = []
        
    def create_time_features(self, df):
        """Create time-based features"""
        df = df.copy()
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['week_of_year'] = df.index.isocalendar().week.astype(int)
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['date_offset'] = (df.index.month * 100 + df.index.day - 320) % 1300
        df['season'] = pd.cut(df['date_offset'], [0, 300, 602, 900, 1300], 
                              labels=[0, 1, 2, 3])
        return df
    
    def create_lag_features(self, df, target_col='balance', n_lags=None):
        """Create lag features"""
        if n_lags is None:
            n_lags = self.lag_features
        df = df.copy()
        for lag in range(1, n_lags + 1):
            df[f'lag_{lag}'] = df[target_col].shift(lag)
        return df
    
    def create_rolling_features(self, df, target_col='balance', windows=[3, 7, 14, 30]):
        """Create rolling window statistics"""
        df = df.copy()
        for window in windows:
            df[f'rolling_mean_{window}'] = df[target_col].shift(1).rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df[target_col].shift(1).rolling(window=window).std()
            df[f'rolling_min_{window}'] = df[target_col].shift(1).rolling(window=window).min()
            df[f'rolling_max_{window}'] = df[target_col].shift(1).rolling(window=window).max()
        return df
    
    def create_transaction_features(self, df):
        """Create features from transaction patterns"""
        df = df.copy()
        df['daily_credits'] = df['credit']
        df['daily_debits'] = df['debit']
        df['daily_net_flow'] = df['daily_credits'] - df['daily_debits']
        
        for lag in [1, 3, 7]:
            df[f'net_flow_lag_{lag}'] = df['daily_net_flow'].shift(lag)
            df[f'credits_lag_{lag}'] = df['daily_credits'].shift(lag)
            df[f'debits_lag_{lag}'] = df['daily_debits'].shift(lag)
        return df
    
    def prepare_data(self, df, target_col='balance'):
        """Main data preparation pipeline"""
        df = df.copy()
        df = self.create_time_features(df)
        df = self.create_transaction_features(df)
        df = self.create_lag_features(df, target_col)
        df = self.create_rolling_features(df, target_col)
        df_clean = df.dropna()
        return df_clean
    
    def fit(self, train_df, test_df=None, target_col='balance', xgb_params=None):
        """Train XGBoost model"""
        if xgb_params is None:
            xgb_params = {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        
        feature_cols = [col for col in train_df.columns 
                       if col not in [target_col, 'description', 'name', 'category']]
        self.feature_names = feature_cols
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        
        eval_set = [(X_train, y_train)]
        if test_df is not None:
            X_test = test_df[feature_cols]
            y_test = test_df[target_col]
            eval_set.append((X_test, y_test))
        
        self.model = xgb.XGBRegressor(enable_categorical=True, **xgb_params)
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        return self
    
    def predict(self, test_df):
        """Make predictions"""
        X_test = test_df[self.feature_names]
        return self.model.predict(X_test)
    
    def forecast_future(self, df, days=7, target_col='balance'):
        """
        Forecast future days iteratively
        """
        df = df.copy()
        forecasts = []
        
        for day in range(days):
            # Get the last row prepared for prediction
            prepared = self.prepare_data(df, target_col)
            if len(prepared) == 0:
                break
                
            last_row = prepared.iloc[[-1]]
            prediction = self.model.predict(last_row[self.feature_names])[0]
            
            # Create next day entry
            next_date = df.index[-1] + timedelta(days=1)
            forecasts.append({
                'date': next_date,
                'predicted_balance': float(prediction)
            })
            
            # Add to dataframe for next iteration (with estimated 0 transactions)
            new_row = pd.DataFrame({
                'balance': [prediction],
                'credit': [0],
                'debit': [0]
            }, index=[next_date])
            df = pd.concat([df, new_row])
        
        return forecasts
    
    def evaluate(self, test_df, target_col='balance'):
        """Evaluate model"""
        y_true = test_df[target_col].values
        y_pred = self.predict(test_df)
        
        metrics = {
            'MAE': float(mean_absolute_error(y_true, y_pred)),
            'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'MAPE': float(mean_absolute_percentage_error(y_true, y_pred) * 100)
        }
        return metrics, y_pred


def parse_transaction_data(data_string):
    """
    Parse transaction data from tab-separated string format
    Expected columns: Date, Description, Name, Category, Credit, Debit, Balance, Currency
    """
    lines = data_string.strip().split('\n')
    
    # Check if first line is header
    if 'Date' in lines[0] and 'Balance' in lines[0]:
        lines = lines[1:]  # Skip header
    
    records = []
    for line in lines:
        parts = line.split('\t')
        if len(parts) >= 7:
            try:
                # Parse date
                date = pd.to_datetime(parts[0].strip())
                
                # Parse credit (remove commas and handle empty)
                credit_str = parts[4].strip().replace(',', '').replace('$', '')
                credit = float(credit_str) if credit_str else 0
                
                # Parse debit (remove commas, parentheses, and handle empty)
                debit_str = parts[5].strip().replace(',', '').replace('$', '').replace('(', '').replace(')', '')
                debit = float(debit_str) if debit_str else 0
                
                # Parse balance (remove commas)
                balance_str = parts[6].strip().replace(',', '').replace('$', '')
                balance = float(balance_str) if balance_str else 0
                
                records.append({
                    'date': date,
                    'description': parts[1].strip() if len(parts) > 1 else '',
                    'name': parts[2].strip() if len(parts) > 2 else '',
                    'category': parts[3].strip() if len(parts) > 3 else '',
                    'credit': credit,
                    'debit': debit,
                    'balance': balance
                })
            except (ValueError, IndexError) as e:
                continue  # Skip malformed rows
    
    return pd.DataFrame(records)


def preprocess_data(df):
    """
    Preprocess transaction data to daily aggregates
    """
    df = df.copy()
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()
    
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Fill NaN credits/debits with 0
    df['credit'] = df['credit'].fillna(0)
    df['debit'] = df['debit'].fillna(0)
    
    # Aggregate to daily level (take end-of-day balance)
    daily_df = df.groupby('date').agg({
        'balance': 'last',
        'credit': 'sum',
        'debit': 'sum'
    }).reset_index()
    
    daily_df = daily_df.set_index('date')
    daily_df = daily_df.sort_index()
    
    return daily_df


# ==================== API ENDPOINTS ====================

@app.route('/', methods=['GET'])
def home():
    """Health check and API info"""
    return jsonify({
        'status': 'ok',
        'message': 'Bank Balance Forecasting API',
        'version': '1.0.0',
        'endpoints': {
            '/': 'API info and health check',
            '/train': 'POST - Train model with transaction data',
            '/predict': 'POST - Get predictions for test data',
            '/forecast': 'POST - Forecast future days',
            '/train-and-forecast': 'POST - Train and immediately forecast'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for Render"""
    return jsonify({'status': 'healthy'})


@app.route('/train', methods=['POST'])
def train():
    """
    Train the model with transaction data
    
    Expected JSON body:
    {
        "data": "tab-separated transaction data string" OR
        "transactions": [{"date": "...", "credit": 0, "debit": 0, "balance": 0}, ...]
    }
    """
    global trained_model, training_data, feature_names
    
    try:
        content = request.json
        
        if 'data' in content:
            # Parse tab-separated string
            df = parse_transaction_data(content['data'])
        elif 'transactions' in content:
            # Parse JSON array
            df = pd.DataFrame(content['transactions'])
        else:
            return jsonify({'error': 'No data provided. Use "data" or "transactions" key.'}), 400
        
        if len(df) < 30:
            return jsonify({'error': 'Insufficient data. Need at least 30 days of transactions.'}), 400
        
        # Preprocess
        daily_data = preprocess_data(df)
        training_data = daily_data.copy()
        
        # Initialize and train forecaster
        forecaster = BankTransactionForecaster(lag_features=14)
        prepared_data = forecaster.prepare_data(daily_data, target_col='balance')
        
        # Split for training
        split_idx = int(len(prepared_data) * 0.8)
        train_df = prepared_data.iloc[:split_idx]
        test_df = prepared_data.iloc[split_idx:]
        
        # Train
        forecaster.fit(train_df, test_df, target_col='balance')
        trained_model = forecaster
        feature_names = forecaster.feature_names
        
        # Evaluate
        metrics, _ = forecaster.evaluate(test_df, target_col='balance')
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'data_points': len(daily_data),
            'training_samples': len(train_df),
            'test_samples': len(test_df),
            'metrics': metrics,
            'date_range': {
                'start': str(daily_data.index.min().date()),
                'end': str(daily_data.index.max().date())
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make predictions on provided data
    
    Expected JSON body:
    {
        "data": "tab-separated transaction data string" OR
        "transactions": [{"date": "...", "credit": 0, "debit": 0, "balance": 0}, ...]
    }
    """
    global trained_model
    
    if trained_model is None:
        return jsonify({'error': 'Model not trained. Call /train first.'}), 400
    
    try:
        content = request.json
        
        if 'data' in content:
            df = parse_transaction_data(content['data'])
        elif 'transactions' in content:
            df = pd.DataFrame(content['transactions'])
        else:
            return jsonify({'error': 'No data provided.'}), 400
        
        daily_data = preprocess_data(df)
        prepared_data = trained_model.prepare_data(daily_data, target_col='balance')
        
        predictions = trained_model.predict(prepared_data)
        
        results = []
        for i, (date, pred) in enumerate(zip(prepared_data.index, predictions)):
            results.append({
                'date': str(date.date()),
                'actual_balance': float(prepared_data['balance'].iloc[i]),
                'predicted_balance': float(pred)
            })
        
        return jsonify({
            'success': True,
            'predictions': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/forecast', methods=['POST'])
def forecast():
    """
    Forecast future days
    
    Expected JSON body:
    {
        "days": 7  (optional, default 7)
    }
    """
    global trained_model, training_data
    
    if trained_model is None or training_data is None:
        return jsonify({'error': 'Model not trained. Call /train first.'}), 400
    
    try:
        content = request.json or {}
        days = content.get('days', 7)
        
        if days < 1 or days > 365:
            return jsonify({'error': 'Days must be between 1 and 365.'}), 400
        
        forecasts = trained_model.forecast_future(training_data, days=days, target_col='balance')
        
        # Format dates as strings
        for f in forecasts:
            f['date'] = str(f['date'].date())
        
        return jsonify({
            'success': True,
            'forecast_days': days,
            'last_known_date': str(training_data.index[-1].date()),
            'last_known_balance': float(training_data['balance'].iloc[-1]),
            'forecasts': forecasts
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/train-and-forecast', methods=['POST'])
def train_and_forecast():
    """
    Train model and immediately forecast future days
    
    Expected JSON body:
    {
        "data": "tab-separated transaction data string" OR
        "transactions": [{"date": "...", "credit": 0, "debit": 0, "balance": 0}, ...],
        "forecast_days": 7  (optional, default 7)
    }
    """
    global trained_model, training_data, feature_names
    
    try:
        content = request.json
        forecast_days = content.get('forecast_days', 7)
        
        if 'data' in content:
            df = parse_transaction_data(content['data'])
        elif 'transactions' in content:
            df = pd.DataFrame(content['transactions'])
        else:
            return jsonify({'error': 'No data provided.'}), 400
        
        if len(df) < 30:
            return jsonify({'error': 'Insufficient data. Need at least 30 days of transactions.'}), 400
        
        # Preprocess
        daily_data = preprocess_data(df)
        training_data = daily_data.copy()
        
        # Initialize and train forecaster
        forecaster = BankTransactionForecaster(lag_features=14)
        prepared_data = forecaster.prepare_data(daily_data, target_col='balance')
        
        # Split for training
        split_idx = int(len(prepared_data) * 0.8)
        train_df = prepared_data.iloc[:split_idx]
        test_df = prepared_data.iloc[split_idx:]
        
        # Train
        forecaster.fit(train_df, test_df, target_col='balance')
        trained_model = forecaster
        feature_names = forecaster.feature_names
        
        # Evaluate
        metrics, _ = forecaster.evaluate(test_df, target_col='balance')
        
        # Forecast
        forecasts = forecaster.forecast_future(daily_data, days=forecast_days, target_col='balance')
        
        # Format dates as strings
        for f in forecasts:
            f['date'] = str(f['date'].date())
        
        return jsonify({
            'success': True,
            'message': 'Model trained and forecasted successfully',
            'training': {
                'data_points': len(daily_data),
                'training_samples': len(train_df),
                'test_samples': len(test_df),
                'metrics': metrics,
                'date_range': {
                    'start': str(daily_data.index.min().date()),
                    'end': str(daily_data.index.max().date())
                }
            },
            'forecast': {
                'days': forecast_days,
                'last_known_date': str(daily_data.index[-1].date()),
                'last_known_balance': float(daily_data['balance'].iloc[-1]),
                'predictions': forecasts
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)

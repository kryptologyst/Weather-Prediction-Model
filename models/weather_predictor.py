"""
Enhanced Weather Prediction Model
Improved version with multiple algorithms and better feature engineering.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class WeatherPredictor:
    def __init__(self):
        self.models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.model_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
        os.makedirs(self.model_dir, exist_ok=True)
    
    def create_features(self, df):
        """Create advanced features for weather prediction"""
        df = df.copy()
        df = df.sort_values('date')
        
        # Lag features for temperature
        for lag in range(1, 8):  # 1-7 days
            df[f'temp_lag_{lag}'] = df['temperature'].shift(lag)
        
        # Rolling statistics
        for window in [3, 7, 14]:
            df[f'temp_rolling_mean_{window}'] = df['temperature'].rolling(window=window).mean()
            df[f'temp_rolling_std_{window}'] = df['temperature'].rolling(window=window).std()
            df[f'humidity_rolling_mean_{window}'] = df['humidity'].rolling(window=window).mean()
            df[f'pressure_rolling_mean_{window}'] = df['pressure'].rolling(window=window).mean()
        
        # Seasonal features
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['season'] = df['month'].map({12: 0, 1: 0, 2: 0,  # Winter
                                       3: 1, 4: 1, 5: 1,   # Spring
                                       6: 2, 7: 2, 8: 2,   # Summer
                                       9: 3, 10: 3, 11: 3}) # Fall
        
        # Cyclical encoding for seasonal patterns
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Weather condition encoding
        condition_mapping = {
            'Clear': 0, 'Cloudy': 1, 'Light Rain': 2, 
            'Heavy Rain': 3, 'Hot': 4, 'Cold': 5
        }
        df['condition_encoded'] = df['condition'].map(condition_mapping).fillna(0)
        
        # Interaction features
        df['temp_humidity_interaction'] = df['temp_lag_1'] * df['humidity']
        df['pressure_wind_interaction'] = df['pressure'] * df['wind_speed']
        
        return df
    
    def prepare_data(self, df, target_col='temperature'):
        """Prepare data for training"""
        df_features = self.create_features(df)
        
        # Select feature columns (exclude date, city, target, and other non-numeric)
        exclude_cols = ['date', 'city', target_col, 'condition']
        feature_cols = [col for col in df_features.columns if col not in exclude_cols]
        
        # Remove rows with NaN values
        df_clean = df_features.dropna()
        
        if len(df_clean) == 0:
            raise ValueError("No valid data after feature engineering")
        
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        self.feature_columns = feature_cols
        
        return X, y, df_clean
    
    def train_models(self, X, y):
        """Train multiple models and select the best one"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for name, model in self.models.items():
            try:
                # Train model
                if name in ['svr']:  # Models that need scaled data
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Evaluate
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                # Cross-validation score
                if name in ['svr']:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                
                results[name] = {
                    'model': model,
                    'mse': mse,
                    'r2': r2,
                    'mae': mae,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred,
                    'test_actual': y_test
                }
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        # Select best model based on R² score
        if results:
            best_name = max(results.keys(), key=lambda k: results[k]['r2'])
            self.best_model = results[best_name]['model']
            self.best_model_name = best_name
        
        return results
    
    def predict(self, X):
        """Make predictions using the best model"""
        if self.best_model is None:
            raise ValueError("No trained model available")
        
        if self.best_model_name in ['svr']:
            X_scaled = self.scaler.transform(X)
            return self.best_model.predict(X_scaled)
        else:
            return self.best_model.predict(X)
    
    def predict_next_days(self, recent_data, days=7):
        """Predict weather for the next N days"""
        predictions = []
        current_data = recent_data.copy()
        
        for day in range(days):
            # Prepare features for the next day
            features_df = self.create_features(current_data)
            
            if len(features_df) == 0:
                break
                
            # Get the latest row features
            latest_features = features_df[self.feature_columns].iloc[-1:].fillna(0)
            
            # Make prediction
            pred_temp = self.predict(latest_features)[0]
            
            # Create next day data point
            next_date = current_data['date'].max() + pd.Timedelta(days=1)
            next_row = {
                'date': next_date,
                'temperature': pred_temp,
                'humidity': current_data['humidity'].iloc[-1],  # Use last known
                'pressure': current_data['pressure'].iloc[-1],
                'wind_speed': current_data['wind_speed'].iloc[-1],
                'precipitation': 0,  # Assume no precipitation
                'condition': 'Clear',
                'city': current_data['city'].iloc[-1] if 'city' in current_data.columns else 'Unknown'
            }
            
            predictions.append({
                'date': next_date,
                'predicted_temperature': round(pred_temp, 1),
                'day': day + 1
            })
            
            # Add prediction to current data for next iteration
            current_data = pd.concat([current_data, pd.DataFrame([next_row])], ignore_index=True)
        
        return predictions
    
    def save_model(self, filename=None):
        """Save the trained model"""
        if filename is None:
            filename = f'weather_model_{self.best_model_name}.joblib'
        
        filepath = os.path.join(self.model_dir, filename)
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, filepath)
        return filepath
    
    def load_model(self, filename):
        """Load a saved model"""
        filepath = os.path.join(self.model_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
    
    def get_feature_importance(self):
        """Get feature importance for tree-based models"""
        if self.best_model_name in ['random_forest', 'gradient_boost']:
            importance = self.best_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importance
            }).sort_values('importance', ascending=False)
            return feature_importance
        else:
            return None

if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.weather_database import WeatherDatabase
    
    # Load data
    db = WeatherDatabase()
    data = db.load_data(city='New York')
    
    # Initialize and train predictor
    predictor = WeatherPredictor()
    X, y, df_clean = predictor.prepare_data(data)
    
    print("Training weather prediction models...")
    results = predictor.train_models(X, y)
    
    # Print results
    print(f"\nBest model: {predictor.best_model_name}")
    for name, result in results.items():
        print(f"{name}: R² = {result['r2']:.3f}, MSE = {result['mse']:.3f}, MAE = {result['mae']:.3f}")
    
    # Save model
    model_path = predictor.save_model()
    print(f"Model saved to: {model_path}")
    
    # Make predictions for next 7 days
    recent_data = data.tail(30)
    predictions = predictor.predict_next_days(recent_data, days=7)
    
    print("\n7-Day Weather Forecast:")
    for pred in predictions:
        print(f"Day {pred['day']} ({pred['date'].strftime('%Y-%m-%d')}): {pred['predicted_temperature']}°C")

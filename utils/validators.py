"""
Input validation and error handling utilities
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class DataValidator:
    """Validates weather data and model inputs"""
    
    @staticmethod
    def validate_city(city):
        """Validate city name"""
        valid_cities = ['New York', 'Los Angeles', 'Chicago', 'Miami', 'Seattle']
        
        if not isinstance(city, str):
            raise ValidationError("City must be a string")
        
        if city not in valid_cities:
            raise ValidationError(f"City '{city}' not supported. Available cities: {valid_cities}")
        
        return city
    
    @staticmethod
    def validate_weather_data(df):
        """Validate weather DataFrame structure and content"""
        required_columns = ['date', 'temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation']
        
        if not isinstance(df, pd.DataFrame):
            raise ValidationError("Data must be a pandas DataFrame")
        
        if df.empty:
            raise ValidationError("DataFrame cannot be empty")
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValidationError(f"Missing required columns: {missing_cols}")
        
        # Validate data types and ranges
        try:
            df['date'] = pd.to_datetime(df['date'])
        except:
            raise ValidationError("Invalid date format")
        
        # Temperature validation (-50°C to 60°C)
        if not df['temperature'].between(-50, 60).all():
            raise ValidationError("Temperature values out of valid range (-50°C to 60°C)")
        
        # Humidity validation (0% to 100%)
        if not df['humidity'].between(0, 100).all():
            raise ValidationError("Humidity values out of valid range (0% to 100%)")
        
        # Pressure validation (900 to 1100 hPa)
        if not df['pressure'].between(900, 1100).all():
            raise ValidationError("Pressure values out of valid range (900 to 1100 hPa)")
        
        # Wind speed validation (0 to 200 km/h)
        if not df['wind_speed'].between(0, 200).all():
            raise ValidationError("Wind speed values out of valid range (0 to 200 km/h)")
        
        # Precipitation validation (0 to 500 mm)
        if not df['precipitation'].between(0, 500).all():
            raise ValidationError("Precipitation values out of valid range (0 to 500 mm)")
        
        return True
    
    @staticmethod
    def validate_prediction_days(days):
        """Validate number of prediction days"""
        if not isinstance(days, int):
            try:
                days = int(days)
            except (ValueError, TypeError):
                raise ValidationError("Days must be an integer")
        
        if days < 1 or days > 30:
            raise ValidationError("Days must be between 1 and 30")
        
        return days
    
    @staticmethod
    def validate_date_range(start_date, end_date=None):
        """Validate date range"""
        try:
            start = pd.to_datetime(start_date)
        except:
            raise ValidationError("Invalid start date format")
        
        if end_date:
            try:
                end = pd.to_datetime(end_date)
            except:
                raise ValidationError("Invalid end date format")
            
            if start >= end:
                raise ValidationError("Start date must be before end date")
            
            if (end - start).days > 365 * 5:  # Max 5 years
                raise ValidationError("Date range cannot exceed 5 years")
        
        return True

class ModelValidator:
    """Validates machine learning model inputs and outputs"""
    
    @staticmethod
    def validate_features(X):
        """Validate feature matrix"""
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise ValidationError("Features must be a DataFrame or numpy array")
        
        if len(X) == 0:
            raise ValidationError("Feature matrix cannot be empty")
        
        # Check for infinite or NaN values
        if isinstance(X, pd.DataFrame):
            if X.isnull().any().any():
                raise ValidationError("Feature matrix contains NaN values")
            if np.isinf(X.select_dtypes(include=[np.number])).any().any():
                raise ValidationError("Feature matrix contains infinite values")
        else:
            if np.isnan(X).any():
                raise ValidationError("Feature matrix contains NaN values")
            if np.isinf(X).any():
                raise ValidationError("Feature matrix contains infinite values")
        
        return True
    
    @staticmethod
    def validate_target(y):
        """Validate target variable"""
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise ValidationError("Target must be a Series or numpy array")
        
        if len(y) == 0:
            raise ValidationError("Target array cannot be empty")
        
        # Check for infinite or NaN values
        if isinstance(y, pd.Series):
            if y.isnull().any():
                raise ValidationError("Target contains NaN values")
        else:
            if np.isnan(y).any():
                raise ValidationError("Target contains NaN values")
        
        if np.isinf(y).any():
            raise ValidationError("Target contains infinite values")
        
        return True
    
    @staticmethod
    def validate_model_results(results):
        """Validate model training results"""
        if not isinstance(results, dict):
            raise ValidationError("Results must be a dictionary")
        
        if not results:
            raise ValidationError("No model results available")
        
        required_metrics = ['mse', 'r2', 'mae']
        for model_name, result in results.items():
            for metric in required_metrics:
                if metric not in result:
                    raise ValidationError(f"Missing metric '{metric}' for model '{model_name}'")
                
                if not isinstance(result[metric], (int, float)):
                    raise ValidationError(f"Invalid metric value for '{metric}' in model '{model_name}'")
        
        return True

def safe_execute(func, *args, **kwargs):
    """Safely execute a function with error handling"""
    try:
        return func(*args, **kwargs), None
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return None, str(e)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None, f"An unexpected error occurred: {str(e)}"

def log_api_request(endpoint, params=None):
    """Log API requests for debugging"""
    logger.info(f"API Request: {endpoint}")
    if params:
        logger.info(f"Parameters: {params}")

def handle_missing_data(df, strategy='interpolate'):
    """Handle missing data in weather DataFrame"""
    if df.isnull().sum().sum() == 0:
        return df
    
    logger.warning(f"Found {df.isnull().sum().sum()} missing values")
    
    if strategy == 'interpolate':
        # Interpolate numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
        
        # Forward fill remaining missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
    
    elif strategy == 'drop':
        df = df.dropna()
    
    elif strategy == 'mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    return df

if __name__ == "__main__":
    # Test validators
    print("Testing validators...")
    
    # Test city validation
    try:
        DataValidator.validate_city("New York")
        print("✓ City validation passed")
    except ValidationError as e:
        print(f"✗ City validation failed: {e}")
    
    # Test invalid city
    try:
        DataValidator.validate_city("Invalid City")
        print("✗ Invalid city validation should have failed")
    except ValidationError:
        print("✓ Invalid city correctly rejected")
    
    # Test prediction days validation
    try:
        DataValidator.validate_prediction_days(7)
        print("✓ Prediction days validation passed")
    except ValidationError as e:
        print(f"✗ Prediction days validation failed: {e}")
    
    print("Validator tests completed")

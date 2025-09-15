"""
Flask Web Application for Weather Prediction
Modern UI with interactive charts and real-time predictions.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import json
import plotly
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.weather_database import WeatherDatabase
from models.weather_predictor import WeatherPredictor
from utils.validators import DataValidator, ModelValidator, ValidationError, safe_execute, log_api_request

app = Flask(__name__)
CORS(app)

# Initialize components
db = WeatherDatabase()
predictor = WeatherPredictor()

# Global variables to store trained model
model_trained = False
available_cities = list(db.cities.keys())

def train_model_if_needed():
    """Train the model if not already trained"""
    global model_trained
    if not model_trained:
        try:
            # Load data for training (using New York as default)
            data = db.load_data(city='New York')
            X, y, df_clean = predictor.prepare_data(data)
            predictor.train_models(X, y)
            model_trained = True
            return True
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    return True

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', cities=available_cities)

@app.route('/api/cities')
def get_cities():
    """Get available cities"""
    return jsonify(available_cities)

@app.route('/api/weather/<city>')
def get_weather_data(city):
    """Get historical weather data for a city"""
    log_api_request(f'/api/weather/{city}')
    
    def _get_weather_data():
        # Validate city
        DataValidator.validate_city(city)
        
        # Load and validate data
        data = db.load_data(city=city)
        DataValidator.validate_weather_data(data)
        
        # Get recent 90 days
        recent_data = data.tail(90)
        
        # Convert to JSON-serializable format
        weather_data = {
            'dates': recent_data['date'].dt.strftime('%Y-%m-%d').tolist(),
            'temperatures': recent_data['temperature'].tolist(),
            'humidity': recent_data['humidity'].tolist(),
            'pressure': recent_data['pressure'].tolist(),
            'wind_speed': recent_data['wind_speed'].tolist(),
            'precipitation': recent_data['precipitation'].tolist(),
            'conditions': recent_data['condition'].tolist()
        }
        
        return weather_data
    
    result, error = safe_execute(_get_weather_data)
    if error:
        return jsonify({'error': error}), 400
    return jsonify(result)

@app.route('/api/stats/<city>')
def get_weather_stats(city):
    """Get weather statistics for a city"""
    try:
        stats = db.get_weather_stats(city)
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/<city>')
def predict_weather(city):
    """Predict weather for the next 7 days"""
    log_api_request(f'/api/predict/{city}')
    
    def _predict_weather():
        # Validate city
        DataValidator.validate_city(city)
        
        # Train model if needed
        if not train_model_if_needed():
            raise ValidationError('Failed to train prediction model')
        
        # Get recent data for the city
        data = db.load_data(city=city)
        DataValidator.validate_weather_data(data)
        recent_data = data.tail(30)
        
        # Make predictions
        predictions = predictor.predict_next_days(recent_data, days=7)
        
        return predictions
    
    result, error = safe_execute(_predict_weather)
    if error:
        return jsonify({'error': error}), 400
    return jsonify(result)

@app.route('/api/model-performance')
def get_model_performance():
    """Get model performance metrics"""
    try:
        if not train_model_if_needed():
            return jsonify({'error': 'Failed to train prediction model'}), 500
        
        # Get training data and evaluate
        data = db.load_data(city='New York')
        X, y, df_clean = predictor.prepare_data(data)
        results = predictor.train_models(X, y)
        
        # Format results for frontend
        performance = {}
        for name, result in results.items():
            performance[name] = {
                'r2_score': round(result['r2'], 3),
                'mse': round(result['mse'], 3),
                'mae': round(result['mae'], 3),
                'cv_mean': round(result['cv_mean'], 3),
                'cv_std': round(result['cv_std'], 3)
            }
        
        return jsonify({
            'best_model': predictor.best_model_name,
            'performance': performance
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chart/<city>/<chart_type>')
def get_chart_data(city, chart_type):
    """Generate chart data for different visualizations"""
    try:
        data = db.load_data(city=city)
        recent_data = data.tail(90)
        
        if chart_type == 'temperature_trend':
            fig = px.line(
                recent_data, 
                x='date', 
                y='temperature',
                title=f'Temperature Trend - {city}',
                labels={'temperature': 'Temperature (°C)', 'date': 'Date'}
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            
        elif chart_type == 'weather_correlation':
            fig = px.scatter_matrix(
                recent_data[['temperature', 'humidity', 'pressure', 'wind_speed']],
                title=f'Weather Parameters Correlation - {city}'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            
        elif chart_type == 'seasonal_pattern':
            monthly_avg = recent_data.groupby(recent_data['date'].dt.month)['temperature'].mean()
            fig = px.bar(
                x=monthly_avg.index,
                y=monthly_avg.values,
                title=f'Monthly Average Temperature - {city}',
                labels={'x': 'Month', 'y': 'Average Temperature (°C)'}
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            
        else:
            return jsonify({'error': 'Invalid chart type'}), 400
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify({'chart': graphJSON})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Generate database if it doesn't exist
    try:
        db.load_data()
        print("Weather database loaded successfully")
    except:
        print("Generating weather database...")
        db.save_data()
        print("Weather database created")
    
    print("Starting Weather Prediction App...")
    print("Available cities:", available_cities)
    app.run(debug=True, host='0.0.0.0', port=5000)

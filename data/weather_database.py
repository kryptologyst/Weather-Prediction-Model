"""
Mock Weather Database
Generates realistic weather data for training and testing the prediction model.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os

class WeatherDatabase:
    def __init__(self):
        self.data_file = os.path.join(os.path.dirname(__file__), 'weather_data.csv')
        self.cities = {
            'New York': {'lat': 40.7128, 'lon': -74.0060, 'base_temp': 12},
            'Los Angeles': {'lat': 34.0522, 'lon': -118.2437, 'base_temp': 18},
            'Chicago': {'lat': 41.8781, 'lon': -87.6298, 'base_temp': 9},
            'Miami': {'lat': 25.7617, 'lon': -80.1918, 'base_temp': 24},
            'Seattle': {'lat': 47.6062, 'lon': -122.3321, 'base_temp': 11}
        }
        
    def generate_weather_data(self, city='New York', days=365*2, start_date='2022-01-01'):
        """Generate realistic weather data for a specific city"""
        np.random.seed(42)
        
        city_info = self.cities.get(city, self.cities['New York'])
        base_temp = city_info['base_temp']
        
        # Generate date range
        start = pd.to_datetime(start_date)
        dates = pd.date_range(start=start, periods=days, freq='D')
        
        # Generate temperature with seasonal patterns
        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        seasonal_temp = base_temp + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # Add random variations and trends
        noise = np.random.normal(0, 3, days)
        trend = np.linspace(0, 2, days)  # Slight warming trend
        temperature = seasonal_temp + noise + trend
        
        # Generate humidity (inversely correlated with temperature)
        humidity = 70 - 0.5 * (temperature - base_temp) + np.random.normal(0, 10, days)
        humidity = np.clip(humidity, 20, 95)
        
        # Generate pressure
        pressure = 1013 + np.random.normal(0, 15, days)
        
        # Generate wind speed
        wind_speed = np.abs(np.random.normal(10, 5, days))
        
        # Generate precipitation (higher chance in winter)
        precip_prob = 0.3 + 0.2 * np.sin(2 * np.pi * (day_of_year + 180) / 365)
        precipitation = np.where(
            np.random.random(days) < precip_prob,
            np.random.exponential(5, days),
            0
        )
        
        # Generate weather conditions
        conditions = []
        for i in range(days):
            if precipitation[i] > 10:
                conditions.append('Heavy Rain')
            elif precipitation[i] > 2:
                conditions.append('Light Rain')
            elif humidity[i] > 85:
                conditions.append('Cloudy')
            elif temperature[i] > base_temp + 20:
                conditions.append('Hot')
            elif temperature[i] < base_temp - 10:
                conditions.append('Cold')
            else:
                conditions.append('Clear')
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'city': city,
            'temperature': np.round(temperature, 1),
            'humidity': np.round(humidity, 1),
            'pressure': np.round(pressure, 1),
            'wind_speed': np.round(wind_speed, 1),
            'precipitation': np.round(precipitation, 2),
            'condition': conditions
        })
        
        return df
    
    def save_data(self, df=None):
        """Save weather data to CSV file"""
        if df is None:
            # Generate data for all cities
            all_data = []
            for city in self.cities.keys():
                city_data = self.generate_weather_data(city=city)
                all_data.append(city_data)
            df = pd.concat(all_data, ignore_index=True)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        df.to_csv(self.data_file, index=False)
        return df
    
    def load_data(self, city=None):
        """Load weather data from CSV file"""
        if not os.path.exists(self.data_file):
            df = self.save_data()
        else:
            df = pd.read_csv(self.data_file)
            df['date'] = pd.to_datetime(df['date'])
        
        if city:
            df = df[df['city'] == city]
        
        return df
    
    def get_recent_data(self, city='New York', days=30):
        """Get recent weather data for a city"""
        df = self.load_data(city=city)
        return df.tail(days)
    
    def get_weather_stats(self, city='New York'):
        """Get weather statistics for a city"""
        df = self.load_data(city=city)
        
        stats = {
            'city': city,
            'total_records': len(df),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d')
            },
            'temperature': {
                'mean': round(df['temperature'].mean(), 1),
                'min': round(df['temperature'].min(), 1),
                'max': round(df['temperature'].max(), 1),
                'std': round(df['temperature'].std(), 1)
            },
            'humidity': {
                'mean': round(df['humidity'].mean(), 1),
                'min': round(df['humidity'].min(), 1),
                'max': round(df['humidity'].max(), 1)
            },
            'conditions': df['condition'].value_counts().to_dict()
        }
        
        return stats

if __name__ == "__main__":
    # Initialize database and generate sample data
    db = WeatherDatabase()
    
    print("Generating weather database...")
    data = db.save_data()
    print(f"Generated {len(data)} weather records for {len(db.cities)} cities")
    
    # Show sample statistics
    for city in db.cities.keys():
        stats = db.get_weather_stats(city)
        print(f"\n{city} Weather Stats:")
        print(f"  Records: {stats['total_records']}")
        print(f"  Avg Temperature: {stats['temperature']['mean']}°C")
        print(f"  Temperature Range: {stats['temperature']['min']}°C to {stats['temperature']['max']}°C")

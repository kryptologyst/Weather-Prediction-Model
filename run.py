#!/usr/bin/env python3
"""
Quick start script for the Weather Prediction Model
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import flask
        import pandas
        import numpy
        import sklearn
        import plotly
        print("✓ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def setup_database():
    """Initialize the weather database"""
    try:
        from data.weather_database import WeatherDatabase
        db = WeatherDatabase()
        
        # Check if data exists
        if not os.path.exists(db.data_file):
            print("Generating weather database...")
            db.save_data()
            print("✓ Weather database created")
        else:
            print("✓ Weather database already exists")
        return True
    except Exception as e:
        print(f"✗ Error setting up database: {e}")
        return False

def main():
    """Main entry point"""
    print("🌤️  Weather Prediction Model Setup")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Setup database
    if not setup_database():
        return 1
    
    print("\n🚀 Starting Weather Prediction App...")
    print("Open your browser to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("-" * 40)
    
    # Start the Flask app
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
        return 0
    except Exception as e:
        print(f"✗ Error starting server: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

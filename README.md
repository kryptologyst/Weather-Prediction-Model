# Weather Prediction Model 🌤️

A comprehensive machine learning project for weather forecasting with an interactive web interface. This project demonstrates advanced ML techniques, feature engineering, and modern web development practices.

## Features

- **Multiple ML Models**: Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, and SVR
- **Advanced Feature Engineering**: Lag features, rolling statistics, seasonal patterns, and interaction terms
- **Interactive Web UI**: Modern, responsive interface with real-time charts and predictions
- **Mock Weather Database**: Realistic synthetic weather data for 5 major cities
- **7-Day Forecasting**: Predict weather conditions for the next week
- **Model Performance Comparison**: Automatic model selection based on performance metrics
- **Real-time Visualizations**: Interactive charts using Plotly.js

  
<img width="1033" height="865" alt="Screenshot 2025-09-11 at 9 22 41 PM" src="https://github.com/user-attachments/assets/c5809bec-c047-4a49-9d97-3ea01e507149" />

## 📁 Project Structure

```
0055_Weather_prediction_model/
├── app.py                      # Flask web application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── 0055.py                     # Original simple model
├── data/
│   ├── weather_database.py     # Mock weather data generator
│   └── weather_data.csv        # Generated weather dataset
├── models/
│   ├── weather_predictor.py    # Enhanced ML prediction model
│   └── saved_models/           # Trained model storage
├── templates/
│   └── index.html              # Web interface template
└── static/                     # Static assets (if needed)
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd 0055_Weather_prediction_model
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate weather database** (optional - auto-generated on first run):
   ```bash
   python data/weather_database.py
   ```

## Usage

### Web Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Features available**:
   - Select different cities from the dropdown
   - View real-time weather statistics
   - Explore interactive temperature trends
   - Generate 7-day weather forecasts
   - Compare model performance metrics
   - Analyze weather parameter correlations

### Command Line Usage

```python
from data.weather_database import WeatherDatabase
from models.weather_predictor import WeatherPredictor

# Initialize components
db = WeatherDatabase()
predictor = WeatherPredictor()

# Load data and train model
data = db.load_data(city='New York')
X, y, df_clean = predictor.prepare_data(data)
results = predictor.train_models(X, y)

# Make predictions
recent_data = data.tail(30)
predictions = predictor.predict_next_days(recent_data, days=7)
print(predictions)
```

## Machine Learning Models

The project implements and compares multiple ML algorithms:

| Model | Description | Use Case |
|-------|-------------|----------|
| **Linear Regression** | Simple baseline model | Quick predictions, interpretability |
| **Ridge Regression** | L2 regularized linear model | Handles multicollinearity |
| **Lasso Regression** | L1 regularized with feature selection | Sparse feature sets |
| **Random Forest** | Ensemble of decision trees | Non-linear patterns, feature importance |
| **Gradient Boosting** | Sequential ensemble method | High accuracy, complex patterns |
| **Support Vector Regression** | Kernel-based regression | Non-linear relationships |

## Features Engineering

The model uses sophisticated feature engineering:

- **Lag Features**: Previous 1-7 days temperature values
- **Rolling Statistics**: Moving averages and standard deviations (3, 7, 14 days)
- **Seasonal Patterns**: Day of year, month, season encoding
- **Cyclical Encoding**: Sine/cosine transformations for seasonal cycles
- **Weather Interactions**: Temperature-humidity and pressure-wind interactions
- **Condition Encoding**: Categorical weather condition mapping

## Available Cities

The mock database includes realistic weather patterns for:

- **New York** (Continental climate)
- **Los Angeles** (Mediterranean climate)
- **Chicago** (Continental climate)
- **Miami** (Tropical climate)
- **Seattle** (Oceanic climate)

## Performance Metrics

Models are evaluated using:

- **R² Score**: Coefficient of determination
- **Mean Squared Error (MSE)**: Average squared prediction errors
- **Mean Absolute Error (MAE)**: Average absolute prediction errors
- **Cross-Validation**: 5-fold CV for robust performance estimation

## Web Interface Features

- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Updates**: Dynamic city switching and data loading
- **Interactive Charts**: Plotly.js powered visualizations
- **Modern UI**: Glassmorphism design with gradient backgrounds
- **Performance Dashboard**: Model comparison and metrics display

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/api/cities` | GET | List available cities |
| `/api/weather/<city>` | GET | Historical weather data |
| `/api/stats/<city>` | GET | Weather statistics |
| `/api/predict/<city>` | GET | 7-day forecast |
| `/api/model-performance` | GET | ML model metrics |
| `/api/chart/<city>/<type>` | GET | Chart data |

## Testing

Run the original simple model:
```bash
python 0055.py
```

Test the enhanced predictor:
```bash
python models/weather_predictor.py
```

Generate and explore the database:
```bash
python data/weather_database.py
```

## Technical Details

### Data Generation
- Realistic seasonal temperature patterns using sine waves
- Correlated humidity, pressure, and wind speed
- Weather condition classification based on parameters
- Multi-year historical data simulation

### Model Training
- Automatic feature scaling for appropriate models
- Cross-validation for robust performance estimation
- Automatic best model selection
- Model persistence using joblib

### Web Framework
- Flask backend with CORS support
- RESTful API design
- Plotly.js for interactive visualizations
- Tailwind CSS for modern styling

## Future Enhancements

- [ ] Real weather API integration
- [ ] Deep learning models (LSTM, GRU)
- [ ] Ensemble model combinations
- [ ] Weather alerts and notifications
- [ ] Historical accuracy tracking
- [ ] Mobile app development
- [ ] Docker containerization

## License

This project is open source and available under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or suggestions, please open an issue in the GitHub repository.

---

**Built with ❤️ using Python, Flask, Scikit-learn, and Plotly**
# Weather-Prediction-Model

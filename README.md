# 📈 Time Series Forecasting for Retail Warehouse Inventory

A comprehensive demand forecasting solution built with Prophet for retail warehouse inventory management. This project provides both a Streamlit web application and a FastAPI REST API for predicting demand patterns, inventory management, and business intelligence.

## 🚀 Features

### Web Application (Streamlit)
- **Demand Prediction**: Forecast units sold for next week/month
- **Trend Analysis**: Detect rising or falling demand patterns
- **Inventory Management**: Get stock recommendations for optimal inventory levels
- **Date-specific Forecasting**: Predict sales for any specific future date
- **Comparative Analysis**: Compare forecasts between different products
- **Restocking Alerts**: Identify stores that need immediate restocking
- **Interactive Visualizations**: Real-time charts and graphs for better insights

### REST API (FastAPI)
- **RESTful Endpoints**: Complete API for integration with external systems
- **Automated Documentation**: Interactive API documentation with Swagger UI
- **Scalable Architecture**: Built for production deployment
- **JSON Responses**: Structured data output for easy integration

## 📁 Project Structure

```
Time_Series_Forecasting/
├── data/                                    # Dataset storage
│   └── retail_warehouse_inventory_dataset.csv
├── models/                                  # Trained model storage
│   └── prophet_models/                      # Prophet model files
├── notebooks/                               # Jupyter notebooks for analysis
│   └── EDA_and_Feature_Engineering.ipynb
├── src/                                     # Source code modules
│   ├── __init__.py
│   ├── data_loader.py                       # Data loading utilities
│   ├── preprocessing.py                     # Data preprocessing functions
│   ├── forecasting.py                       # Prophet model implementation
│   ├── train.py                            # Model training pipeline
│   ├── evaluate.py                         # Model evaluation metrics
│   └── utils.py                            # Helper utilities
├── app.py                                  # Streamlit web application
├── server.py                               # FastAPI REST server
├── main.py                                 # Training pipeline entry point
└── requirements.txt                        # Python dependencies
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/GeekSomesh/Time-Series-Forecasting.git
   cd Time_Series_Forecasting
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset**:
   - Place your retail warehouse inventory dataset in the `data/` directory
   - Ensure the CSV file is named `retail_warehouse_inventory_dataset.csv`
   - The dataset should contain columns: Date, Store ID, Product ID, and sales data

4. **Train the models**:
   ```bash
   python main.py
   ```
   This will:
   - Create the necessary model directories
   - Train Prophet models for each store-product combination
   - Evaluate model performance
   - Save trained models in `models/prophet_models/`

## 🖥️ Usage

### Web Application (Streamlit)

Launch the interactive web application:
```bash
streamlit run app.py
```

The application provides 7 main features:
1. **Predict Units Sold**: Forecast demand for next 7 or 30 days
2. **Detect Demand Trends**: Identify rising or falling demand patterns
3. **Inventory Recommendations**: Get optimal stock levels for next month
4. **Date-specific Forecasting**: Predict sales for any future date
5. **Trend Analysis**: Compare demand trends across all products
6. **Product Comparison**: Compare forecasts between two products
7. **Restocking Alerts**: Identify stores needing immediate attention

### REST API Server (FastAPI)

Start the API server:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

#### API Endpoints:

- **POST** `/predict-units/` - Predict units sold for specified period
- **POST** `/detect-trend/` - Detect rising/falling demand trends
- **POST** `/recommend-stock/` - Get inventory stock recommendations
- **POST** `/forecast-date/` - Forecast for specific future date
- **GET** `/trend-analysis/` - Analyze trends across all store-product pairs
- **POST** `/compare-products/` - Compare forecasts between two products
- **GET** `/restocking-needed/` - Find stores needing restocking

#### Example API Usage:

```bash
# Predict units for next 30 days
curl -X POST "http://localhost:8000/predict-units/" \
     -H "Content-Type: application/json" \
     -d '{"store_id": "S001", "product_id": "P001", "period": 30}'

# Check trend analysis
curl -X GET "http://localhost:8000/trend-analysis/"

# Get restocking recommendations
curl -X GET "http://localhost:8000/restocking-needed/?threshold=50"
```

Access interactive API documentation at: `http://localhost:8000/docs`

## 📊 Dataset Requirements

Your dataset should contain the following columns:
- **Date**: Date of the sales record (YYYY-MM-DD format)
- **Store ID**: Unique identifier for each store
- **Product ID**: Unique identifier for each product
- **Units Sold**: Number of units sold (target variable)

Example dataset structure:
```csv
Date,Store ID,Product ID,Units Sold
2023-01-01,S001,P001,150
2023-01-01,S001,P002,200
2023-01-02,S001,P001,145
...
```

## 🤖 Machine Learning Approach

This project uses **Facebook Prophet** for time series forecasting:

### Why Prophet?
- **Handles seasonality**: Automatically detects daily, weekly, and yearly patterns
- **Robust to missing data**: Works well with incomplete datasets
- **Holiday effects**: Can incorporate holiday impacts on demand
- **Trend changepoints**: Automatically detects trend changes
- **Uncertainty intervals**: Provides confidence intervals for predictions

### Model Training
- Individual models are trained for each store-product combination
- Models capture unique demand patterns for specific store-product pairs
- Automatic hyperparameter tuning for optimal performance
- Cross-validation for model evaluation

## 📈 Business Value

### For Inventory Managers:
- **Reduce stockouts**: Predict when inventory will run low
- **Optimize storage costs**: Avoid overstocking with accurate forecasts
- **Seasonal planning**: Prepare for seasonal demand changes
- **Performance tracking**: Monitor demand trends across locations

### For Business Analysts:
- **Data-driven decisions**: Replace gut-feeling with statistical forecasts
- **Comparative analysis**: Compare performance across products and stores
- **Trend identification**: Spot emerging patterns early
- **ROI optimization**: Focus resources on high-demand products

### For Operations Teams:
- **Automated alerts**: Get notified when stores need restocking
- **Resource allocation**: Plan staff and logistics based on predicted demand
- **Risk mitigation**: Prepare for demand fluctuations
- **Performance metrics**: Track forecast accuracy over time

## 🔧 Configuration

### Model Parameters
You can customize Prophet model parameters in `src/forecasting.py`:
- **Seasonality**: Weekly, monthly, yearly patterns
- **Growth**: Linear or logistic growth curves
- **Changepoints**: Trend change detection sensitivity
- **Holidays**: Regional holiday effects

### API Configuration
Modify server settings in `server.py`:
- **CORS settings**: Cross-origin resource sharing
- **Rate limiting**: API request limits
- **Authentication**: Security middleware
- **Caching**: Response caching for better performance

## 🧪 Testing and Validation

### Model Evaluation Metrics:
- **MAE (Mean Absolute Error)**: Average prediction error
- **MAPE (Mean Absolute Percentage Error)**: Percentage-based accuracy
- **RMSE (Root Mean Square Error)**: Penalty for large errors
- **Cross-validation**: Time series cross-validation scores

### Run Evaluation:
```bash
# Evaluate all trained models
python -c "from src.evaluate import evaluate_models; evaluate_models('data/retail_warehouse_inventory_dataset.csv', 'models/prophet_models')"
```

## 🚀 Deployment

### Streamlit Cloud
Deploy the web app to Streamlit Cloud for easy sharing:
1. Push your code to GitHub
2. Connect your repository to Streamlit Cloud
3. Deploy with one click

### Docker Deployment
Create a Dockerfile for containerized deployment:
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud Platforms
- **Heroku**: Easy deployment with Git integration
- **AWS**: EC2, ECS, or Lambda deployment options
- **Google Cloud**: App Engine or Cloud Run
- **Azure**: Container Instances or App Service

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines:
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation for API changes

## 📋 Dependencies

### Core Libraries:
- **streamlit**: Web application framework
- **fastapi**: REST API framework
- **prophet**: Time series forecasting
- **pandas**: Data manipulation and analysis
- **matplotlib**: Data visualization
- **numpy**: Numerical computations
- **joblib**: Model serialization
- **pydantic**: Data validation for APIs

See `requirements.txt` for complete dependency list.

## 🐛 Troubleshooting

### Common Issues:

1. **Module Import Errors**:
   ```bash
   # Add src to Python path
   export PYTHONPATH="${PYTHONPATH}:./src"
   ```

2. **Prophet Installation Issues**:
   ```bash
   # Install Prophet dependencies
   conda install -c conda-forge prophet
   ```

3. **Memory Issues with Large Datasets**:
   - Process data in chunks
   - Use data sampling for testing
   - Increase system RAM allocation

4. **Model Loading Errors**:
   - Ensure models are trained before prediction
   - Check file paths and permissions
   - Verify model file integrity

## 📊 Performance Optimization

### For Large Datasets:
- **Parallel Processing**: Train models in parallel for different store-product pairs
- **Data Sampling**: Use representative samples for faster training
- **Feature Selection**: Remove irrelevant columns before processing
- **Batch Prediction**: Process multiple forecasts simultaneously

### API Optimization:
- **Caching**: Cache frequently requested forecasts
- **Connection Pooling**: Optimize database connections
- **Async Processing**: Use async/await for I/O operations
- **Load Balancing**: Distribute requests across multiple instances

## 📞 Support

For questions, issues, or feature requests:
- **GitHub Issues**: Open an issue on the repository
- **Documentation**: Check this README and code comments
- **Community**: Join discussions in the Issues section

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Facebook Prophet**: For the powerful time series forecasting library
- **Streamlit**: For the amazing web application framework
- **FastAPI**: For the high-performance API framework
- **Open Source Community**: For the incredible tools and libraries

---

**Made with ❤️ for better inventory management and demand forecasting**

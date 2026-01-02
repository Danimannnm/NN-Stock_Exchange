# NN-Stock Exchange

A comprehensive machine learning pipeline for stock price prediction using neural networks. This project implements three different deep learning architectures (LSTM, CNN, Transformer) to forecast next-day closing prices for major tech stocks.

## Project Overview

This system fetches historical stock data, engineers technical indicators, trains multiple neural network models, and evaluates their performance through backtesting. The project includes an interactive Streamlit dashboard for visualizing predictions and model comparisons.

## Features

- Automated data ingestion from Polygon.io API
- Comprehensive technical indicator generation (MA, EMA, RSI, MACD, Bollinger Bands)
- Three neural network architectures for time series forecasting
- Automated train/validation/test splitting
- Model backtesting with performance metrics
- Interactive web dashboard for analysis and visualization

## Supported Stocks

- AAPL (Apple)
- MSFT (Microsoft)
- GOOGL (Google)
- AMZN (Amazon)
- TSLA (Tesla)

## Architecture

### Models

**LSTM (Long Short-Term Memory)**
- 50 LSTM units
- Captures temporal dependencies in sequential data
- Best for learning long-term patterns

**1D CNN (Convolutional Neural Network)**
- 32 filters with kernel size 3
- Global average pooling
- Efficient at detecting local patterns in time series

**Transformer**
- Multi-head self-attention mechanism (2 heads, 64 dimensions)
- Feed-forward network (128 dimensions)
- State-of-the-art architecture for sequence modeling

### Pipeline Stages

1. **Ingestion**: Fetch historical daily stock data from Polygon.io
2. **Preprocessing**: Clean timestamps and handle missing values
3. **Feature Engineering**: Add technical indicators and lagged features
4. **Filtering**: Remove initial NaN values from rolling calculations
5. **Feature Processing**: Drop non-numeric columns for model training
6. **Dataset Splitting**: Split data chronologically (70% train, 15% validation, 15% test)
7. **Training**: Train all three models with early stopping and checkpointing
8. **Backtesting**: Evaluate models on unseen test data
9. **Reporting**: Visualize results in interactive dashboard

## Project Structure

```
NN-Stock_Exchange/
├── configs/
│   ├── api_keys.yaml          # API credentials
│   └── settings.yaml          # Pipeline configuration
├── data/
│   ├── raw/                   # Raw stock data
│   ├── processed/             # Cleaned and engineered features
│   ├── features/              # Feature-enriched datasets
│   ├── splits/                # Train/val/test splits
│   └── backtest/              # Backtest results and predictions
├── models/
│   ├── *.keras                # Trained model files
│   └── scalers/               # MinMaxScaler objects for each stock
├── src/
│   ├── ingestion/             # Data fetching from Polygon.io
│   ├── preprocessing/         # Data cleaning and feature engineering
│   ├── dataset/               # Dataset splitting
│   ├── modeling/              # Model architectures and training
│   ├── backtest/              # Backtesting logic
│   └── report/                # Streamlit dashboard
├── requirements.txt           # Python dependencies
└── run_pipeline.py           # Main orchestration script
```

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd NN-Stock_Exchange
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API credentials:

Create `configs/api_keys.yaml`:
```yaml
polygon:
  api_key: YOUR_POLYGON_API_KEY
```

Get your free API key from [Polygon.io](https://polygon.io/)

4. Adjust settings (optional):

Edit `configs/settings.yaml` to customize:
- Stock symbols
- Date range
- Data interval (day, hour, minute)

## Usage

### Run Complete Pipeline

Execute all stages from data ingestion to backtesting:

```bash
python run_pipeline.py all
```

### Run Individual Stages

Execute specific pipeline stages:

```bash
# Data ingestion only
python run_pipeline.py ingest

# Preprocessing
python run_pipeline.py prep1

# Feature engineering
python run_pipeline.py feat2

# Filter NaN values
python run_pipeline.py filt3

# Final feature processing
python run_pipeline.py feat4

# Split datasets
python run_pipeline.py split

# Train models
python run_pipeline.py train

# Run backtesting
python run_pipeline.py backtest
```

### Launch Dashboard

Start the interactive Streamlit dashboard:

```bash
streamlit run src/report/app.py
```

The dashboard provides:
- Stock price predictions vs actual values
- Model comparison across architectures
- Error analysis (MAE, MSE, RMSE)
- Returns simulation with trading strategies
- Project workflow visualization
- Technical glossary

## Technical Details

### Data Processing

**Technical Indicators Generated:**
- 10-day Simple Moving Average (SMA)
- 10-day Exponential Moving Average (EMA)
- 14-day Relative Strength Index (RSI)
- MACD and Signal Line
- Bollinger Bands (20-day)
- 1-day lagged returns
- 10-day rolling standard deviation and mean

**Scaling:**
- MinMaxScaler applied to all features
- Fitted on training data only
- Same scaler used for validation and test sets

**Sequence Generation:**
- Window size: 10 days
- Features: All technical indicators
- Target: Next-day closing price

### Model Training

**Configuration:**
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam
- Batch size: 32
- Max epochs: 100
- Early stopping: Patience of 5 epochs on validation loss
- Best model checkpointing: Save best weights based on validation loss

**Hardware Requirements:**
- CPU: Any modern processor
- RAM: 8GB+ recommended
- GPU: Optional (TensorFlow will auto-detect CUDA)

## Performance Metrics

Each model is evaluated using:
- **MAE** (Mean Absolute Error): Average prediction error
- **MSE** (Mean Squared Error): Penalizes larger errors
- **RMSE** (Root Mean Squared Error): Error in original price units
- **Prediction vs Actual Plots**: Visual comparison of forecasts

Results are saved in `data/backtest/` for each symbol-model combination.

## Configuration

### Date Range
Default: 2015-01-01 to 2025-03-31

Modify in `configs/settings.yaml`:
```yaml
start_date: 2015-01-01
end_date: 2025-03-31
```

### Train/Val/Test Split
Default: 70% / 15% / 15%

Modify in `src/dataset/split_dataset.py`:
```python
train_frac=0.7, val_frac=0.15
```

### Model Hyperparameters

Adjust in respective model files:
- `src/modeling/models_lstm.py`
- `src/modeling/models_cnn.py`
- `src/modeling/models_transformer.py`

## Dependencies

Core libraries:
- **TensorFlow 2.19.0**: Deep learning framework
- **pandas 2.2.3**: Data manipulation
- **numpy 1.26.4**: Numerical operations
- **scikit-learn 1.6.1**: Preprocessing and metrics
- **pandas-ta**: Technical analysis indicators
- **Streamlit 1.45.0**: Web dashboard
- **Plotly 6.0.1**: Interactive visualizations
- **polygon-api-client**: Stock data API
- **PyYAML**: Configuration management

## Troubleshooting

### Rate Limit Errors
The ingestion script includes automatic retry logic with exponential backoff. If you encounter persistent rate limits, consider:
- Using a paid Polygon.io plan
- Reducing the date range
- Adding longer sleep intervals

### Memory Issues
For limited RAM:
- Process fewer stocks at once
- Reduce window size in sequence generation
- Lower batch size during training

### Model Not Converging
- Increase training epochs
- Adjust learning rate in optimizer
- Check for data quality issues
- Verify feature scaling is working

## Future Enhancements

- Add more model architectures (GRU, Attention-based models)
- Implement ensemble methods
- Real-time prediction updates
- Portfolio optimization strategies
- Sentiment analysis integration from news/social media
- Multi-step ahead forecasting
- Automated hyperparameter tuning

## License

This project is for educational purposes. Stock trading involves risk. Past performance does not guarantee future results.

## Acknowledgments

- Data provided by Polygon.io
- Built with TensorFlow and Keras
- Dashboard powered by Streamlit
- Technical indicators from pandas-ta

## Contact

For questions or contributions, please open an issue or submit a pull request.

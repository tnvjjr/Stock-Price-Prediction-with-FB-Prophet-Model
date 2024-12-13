# Stock Price Prediction with Facebook Prophet
A modern web application that predicts stock prices using Facebook's Prophet forecasting model, providing accurate time-series predictions with interactive visualizations.

## Table of Contents
* [General Info](#general-information)
* [Features](#features)
* [Tools and Technologies](#tools-and-technologies)
* [Setup](#setup)
* [Usage](#usage)
* [Credits](#credits)


## General Information
This project leverages Facebook's Prophet model to predict stock prices based on historical data. It provides an intuitive interface for users to analyze stock trends and make informed investment decisions. The application features interactive charts and detailed analysis of stock price movements, making complex financial data easily understandable.


## Features
* AI-Powered Predictions: Utilizes Facebook Prophet for accurate stock price forecasting
* Interactive Visualizations: Dynamic charts showing historical data and predictions
* Multiple Time Horizons: Predict stock prices for different future time periods
* Trend Analysis: Breakdown of trend components including yearly, weekly, and daily patterns
* Modern UI Design: Clean and intuitive interface built with Streamlit
* Real-time Data: Integration with financial APIs for up-to-date stock information
* Customizable Parameters: Adjust prediction parameters for different scenarios


## Tools and Technologies
* **Frontend Framework:**
  - Streamlit (Python web framework)
  - Plotly (for interactive visualizations)

* **AI/ML Technologies:**
  - Facebook Prophet (for time series forecasting)
  - Pandas (for data manipulation)
  - NumPy (for numerical computations)

* **APIs and Libraries:**
  - yfinance (for fetching stock data)
  - SQLite (for data storage)
  - datetime (for time series handling)


## Setup

1. **Install Python:**
   - Install Python 3.8 or higher from the [Python official website](https://www.python.org/)

2. **Install Required Modules:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/stock-price-prediction.git
   ```

2. **Run the Application:**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Using the App:**
   - Enter a stock symbol (e.g., AAPL, GOOGL)
   - Select the prediction timeframe
   - View historical data and predictions
   - Analyze trend components and forecast accuracy

4. **Important Notes:**
   - Internet connection required for fetching stock data
   - Some stocks might have limited historical data
   - Predictions are based on historical patterns and should not be the sole basis for investment decisions

## Additional Notes:
* The app requires an active internet connection
* Historical data availability varies by stock
* Prophet model works best with stocks having consistent trading history
* Regular updates ensure accurate predictions

## Credits

Created by Vijay Shrivarshan Vijayaraja  
Powered by Facebook Prophet

---

<div align="center">
Made by Vijay Shrivarshan Vijayaraja
</div>

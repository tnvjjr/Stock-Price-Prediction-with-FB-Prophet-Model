from django.shortcuts import render
import pandas as pd
import yfinance as yf
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

def get_stock_data(symbol, start_date, end_date):
    """Download stock data from Yahoo Finance"""
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        if stock_data.empty:
            raise ValueError(f"No data found for symbol {symbol}. Please check if the stock symbol is correct.")
        return stock_data
    except Exception as e:
        raise Exception(f"Error downloading data: {str(e)}")

def prepare_data_for_prophet(df):
    """Prepare data for Prophet model"""
    df = df.reset_index()
    df = df[['Date', 'Close']]
    df.columns = ['ds', 'y']
    return df

def train_prophet_model(df, periods=365):
    """Train Prophet model and make predictions"""
    model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True,
                   changepoint_prior_scale=0.05)
    model.fit(df)
    
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast

def create_plot(original_df, forecast, symbol):
    """Create plotly figure for stock prediction"""
    fig = go.Figure()
    
    # Plot original data
    fig.add_trace(go.Scatter(
        x=original_df['ds'],
        y=original_df['y'],
        name='Actual Stock Price',
        line=dict(color='blue')
    ))
    
    # Plot predictions
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        name='Predicted Stock Price',
        line=dict(color='red')
    ))
    
    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        fill=None,
        mode='lines',
        line_color='rgba(0,0,0,0)',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        name='Confidence Interval'
    ))
    
    fig.update_layout(
        title=f'{symbol} Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Stock Price ($)',
        hovermode='x',
        template='plotly_white'
    )
    
    return fig.to_html(full_html=False)

def get_prediction_stats(forecast):
    """Get summary statistics from forecast"""
    last_date = forecast['ds'].iloc[-1]
    last_price = forecast['yhat'].iloc[-1]
    max_price = forecast['yhat'].max()
    min_price = forecast['yhat'].min()
    
    return {
        'last_date': last_date.strftime('%Y-%m-%d'),
        'last_price': f"{last_price:.2f}",
        'max_price': f"{max_price:.2f}",
        'min_price': f"{min_price:.2f}"
    }

def home(request):
    context = {}
    if request.method == 'POST':
        try:
            symbol = request.POST.get('symbol', '').upper().strip()
            prediction_days = int(request.POST.get('period', 365))
            
            # Get stock data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365*3)  # 3 years of historical data
            stock_data = get_stock_data(symbol, start_date, end_date)
            
            # Prepare data and train model
            df = prepare_data_for_prophet(stock_data)
            model, forecast = train_prophet_model(df, periods=prediction_days)
            
            # Create plot
            plot_div = create_plot(df, forecast, symbol)
            
            # Get statistics
            stats = get_prediction_stats(forecast)
            
            context = {
                'plot_div': plot_div,
                'stats': stats,
                'symbol': symbol
            }
        except Exception as e:
            context['error'] = str(e)
    
    return render(request, 'predictor/index.html', context)

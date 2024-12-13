import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page config
st.set_page_config(page_title="Stock Price Prediction using Machine Learning", layout="wide")

# Custom CSS with modern design
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
        @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css');
        
        /* Variables */
        :root {
            --primary-color: #3b82f6;
            --secondary-color: #60a5fa;
            --accent-color: #93c5fd;
            --bg-dark: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
        }
        
        /* Global Styles */
        .stApp {
            background: linear-gradient(135deg, var(--bg-dark) 0%, #1e3a8a 100%);
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        
        /* Modern Container */
        .main-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        /* Header Styles */
        .header {
            text-align: center;
            padding: 3rem 0;
            margin-bottom: 3rem;
            background: rgba(15, 23, 42, 0.7);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            position: relative;
            overflow: hidden;
        }
        
        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, 
                transparent 0%, 
                var(--accent-color) 50%, 
                transparent 100%);
            animation: shimmer 3s infinite;
        }
        
        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        .title-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.5rem;
        }
        
        .main-title {
            font-size: 2.75rem;
            font-weight: 700;
            background: linear-gradient(135deg, #fff, var(--accent-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            text-shadow: 0 0 30px rgba(147, 197, 253, 0.3);
        }
        
        .subtitle {
            font-size: 1.1rem;
            color: rgba(255, 255, 255, 0.7);
            font-weight: 400;
            max-width: 600px;
            line-height: 1.6;
            margin-bottom: 0.5rem;
        }
        
        .creator {
            font-size: 1rem;
            color: var(--accent-color);
            font-weight: 500;
            opacity: 0.9;
        }
        
        /* Input Container */
        .input-container {
            background: var(--card-bg);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        
        /* Button Styles */
        .stButton > button {
            width: 100%;
            background: var(--primary-color) !important;
            color: white !important;
            border: none !important;
            padding: 1rem !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            transition: transform 0.3s ease !important;
            margin-top: 1rem !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
        }
        
        /* Card Styles */
        .prediction-card {
            background: var(--card-bg);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            transition: transform 0.3s ease;
        }
        
        .prediction-card:hover {
            transform: translateY(-5px);
        }
        
        /* Input Fields */
        .stTextInput input, .stNumberInput input {
            background: rgba(15, 23, 42, 0.7) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px !important;
            color: white !important;
            padding: 0.75rem 1rem !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important;
        }
        
        .stTextInput input:focus, .stNumberInput input:focus {
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
        }
        
        /* Select Box */
        .stSelectbox select {
            background: rgba(15, 23, 42, 0.7) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px !important;
            color: white !important;
            padding: 0.75rem 1rem !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important;
        }
        
        /* Graph Container */
        .graph-container {
            background: var(--card-bg);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2rem;
            margin-top: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem 0;
            margin-top: 3rem;
            color: rgba(255, 255, 255, 0.7);
            font-size: 0.875rem;
        }
        
        /* Metrics */
        .metrics-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-top: 2rem;
        }
        
        .metric-card {
            background: rgba(15, 23, 42, 0.7);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--primary-color);
            margin-bottom: 0.5rem;
        }
        
        .metric-label {
            font-size: 0.875rem;
            color: rgba(255, 255, 255, 0.7);
        }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
    <div class="header">
        <div class="title-container">
            <h1 class="main-title">Stock Price Prediction using Machine Learning</h1>
            <p class="subtitle">Powered by Facebook Prophet Model with Advanced Time Series Analysis</p>
            <p class="creator">Created by Vijay Shrivarshan Vijayaraja</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Main container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

def get_stock_data(symbol, start_date, end_date):
    """Download stock data from Yahoo Finance"""
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        if stock_data.empty:
            st.error(f"No data found for symbol {symbol}. Please check if the stock symbol is correct.")
            return None
        return stock_data
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return None

def prepare_data_for_prophet(df):
    """Prepare data for Prophet model"""
    df = df.reset_index()
    df = df[['Date', 'Close']]
    df.columns = ['ds', 'y']
    return df

def train_prophet_model(df, periods=365):
    """Train Prophet model and make predictions"""
    with st.spinner('Training model... This may take a few moments.'):
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
        line=dict(color='#10b981', width=2)  # Green for actual data
    ))
    
    # Plot predictions
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        name='Predicted Stock Price',
        line=dict(color='#f59e0b', width=2)  # Orange for predictions
    ))
    
    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        fill=None,
        mode='lines',
        line_color='rgba(245, 158, 11, 0)',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(245, 158, 11, 0)',
        name='Confidence Interval',
        fillcolor='rgba(245, 158, 11, 0.1)'
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(30, 41, 59, 0.7)',
        paper_bgcolor='rgba(30, 41, 59, 0)',
        font=dict(
            family='Plus Jakarta Sans',
            color='white',
            size=12
        ),
        xaxis=dict(
            title='Date',
            gridcolor='rgba(255, 255, 255, 0.1)',
            zerolinecolor='rgba(255, 255, 255, 0.1)',
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            title='Stock Price ($)',
            gridcolor='rgba(255, 255, 255, 0.1)',
            zerolinecolor='rgba(255, 255, 255, 0.1)',
            tickfont=dict(size=10),
            tickprefix='$'
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(255, 255, 255, 0.1)',
            font=dict(size=10)
        ),
        hovermode='x unified',
        height=600,
        margin=dict(t=30),
        hoverlabel=dict(
            bgcolor='rgba(30, 41, 59, 0.9)',
            font_size=12,
            font_family='Plus Jakarta Sans'
        )
    )
    
    return fig

# Input Section
with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, GOOGL)", value="AAPL")
    
    with col2:
        prediction_period = st.selectbox("Select Prediction Period (Days)", 
                                       options=[7, 14, 30, 60, 90],
                                       index=2)
    
    if st.button("Generate Prediction"):
        # Get stock data
        df = get_stock_data(stock_symbol.upper(), datetime.now() - timedelta(days=365*2), datetime.now())
        
        if df is not None:
            # Prepare data and train model
            prophet_df = prepare_data_for_prophet(df)
            model, forecast = train_prophet_model(prophet_df, periods=prediction_period)
            
            # Display predictions
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="graph-container">', unsafe_allow_html=True)
            st.plotly_chart(create_plot(prophet_df, forecast, stock_symbol), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show prediction stats
            st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <i class="fas fa-clock" style="font-size: 2.5rem; color: var(--primary-color);"></i>
                    <h5 style="margin: 1rem 0;">Latest Prediction</h5>
                    <div class="metric-value">${forecast['yhat'].iloc[-1]:.2f}</div>
                    <div class="metric-label">Predicted on {forecast['ds'].iloc[-1].strftime('%Y-%m-%d')}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <i class="fas fa-arrow-up" style="font-size: 2.5rem; color: var(--success-color);"></i>
                    <h5 style="margin: 1rem 0;">Maximum Price</h5>
                    <div class="metric-value">${forecast['yhat'].max():.2f}</div>
                    <div class="metric-label">Predicted High</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <i class="fas fa-arrow-down" style="font-size: 2.5rem; color: var(--danger-color);"></i>
                    <h5 style="margin: 1rem 0;">Minimum Price</h5>
                    <div class="metric-value">${forecast['yhat'].min():.2f}</div>
                    <div class="metric-label">Predicted Low</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("""
    <p>Created by <span style="color: var(--primary-color); font-weight: 600;">Vijay Shrivarshan Vijayaraja</span></p>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

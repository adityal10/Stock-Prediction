import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
# from fbprophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go 

START = '2020-01-01'
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction")

stocks = ('^NSEI','RELIANCE.NS','IRFC.NS','CIPLA.BO','ANGELONE.NS','ZOMATO.NS')
selected_stocks = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediciton:", 1, 4)
period = n_years*365

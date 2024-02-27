import streamlit as st
# import pandas_datareader as web
from pandas_datareader import data as pdr
import yfinance as yf
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
# import plotly.figure_factory as ff
from plotly import graph_objs as go

import datetime


st.title("Stock Prediction")

yf.pdr_override()

stocks = ['^NSEI', 'RELIANCE.NS', 'CIPLA.NS', 'IRFC.NS', 'ZOMATO.NS', 'ANGELONE.NS']

@st.cache_data
def fetch_data():
    global my_dict
    my_dict = dict()
    present_date = datetime.datetime.now().strftime('%Y-%m-%d')
    st.write(f'Fetching Stock data from the year 2014 January to present date.')
    for stock in stocks:
        data = pdr.get_data_yahoo(f"{stock}", start="2014-01-01", end=f"{present_date}")
        my_dict[stock] = data
    return my_dict
stock_data = fetch_data()


def plot_opening_price(_df):
    fig = go.Figure()

    for stock_symbol, data in _df.items():
        # Convert JSON to DataFrame
        df = pd.DataFrame(data)

        # Parse the 'Date' column as datetime
        df['Date'] = pd.to_datetime(df.index)

        # Add trace for each stock
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Open'], name=f'{stock_symbol}_open'))

    fig.update_layout(
        title_text='Time Series Data - Opening Prices', 
        xaxis_rangeslider_visible=True,
        width=900,  # Set the width of the plot
        height=600,   # Set the height of the plot
        legend=dict(
            title=dict(text='Stock Symbol', font=dict(size=15)),
            font=dict(size=12)
        ),
            xaxis=dict(title='Date', tickfont=dict(size=14)),
            yaxis=dict(title='Opening Price', tickfont=dict(size=14))
        )
    st.plotly_chart(fig)

plot_opening_price(stock_data)




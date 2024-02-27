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
from io import StringIO


st.title("Stock Prediction")

yf.pdr_override()

stocks = ['^NSEI', 'RELIANCE.NS', 'CIPLA.NS', 'IRFC.NS', 'ZOMATO.NS', 'ANGELONE.NS']

@st.cache_data
def fetch_data():
    global my_dict
    my_dict = dict()
    present_date = datetime.datetime.now().strftime('%Y-%m-%d')
    for stock in stocks:
        data = pdr.get_data_yahoo(f"{stock}", start="2014-01-01", end=f"{present_date}")
        my_dict[stock] = data
    return my_dict

data_load_state = st.text("Load Data...")
stock_data = fetch_data()
data_load_state.text("Load Data...Done!")
com_df = pd.concat({k: pd.DataFrame(v) for k, v in stock_data.items()}, axis=0, names=['Stock']).reset_index(level=1)
# Convert the DataFrame to CSV string
csv_string = com_df.to_csv(index=True)

st.download_button(
        label="Download Combined CSV",
        data=csv_string,
        file_name="combined_stock_data.csv",
        key="download_combined_csv"
    )


st.text("Raw Data - NIFTY50")
st.dataframe(stock_data['^NSEI'].tail(), use_container_width=True)
# st.write(stock_data['^NSEI'][f'{selected_column}'])

st.subheader("Time Series Plot")
columns_list = list(pd.DataFrame(stock_data['ANGELONE.NS']).columns)
selected_column = st.selectbox("Select column for plot", columns_list)

def plot_opening_price(_df, col):
    fig = go.Figure()

    for stock_symbol, data in _df.items():
        # Convert JSON to DataFrame
        df = pd.DataFrame(data)
        # Parse the 'Date' column as datetime
        df['Date'] = pd.to_datetime(df.index)

        # Add trace for each stock
        fig.add_trace(go.Scatter(x=df['Date'], y=df[f'{col}'], name=f'{stock_symbol}_{col}'))

    fig.update_layout(
        title_text=f'Time Series Data - {col} Prices', 
        xaxis_rangeslider_visible=True,
        width=900,  # Set the width of the plot
        height=600,   # Set the height of the plot
        legend=dict(
            title=dict(text='Stock Symbol', font=dict(size=15)),
            font=dict(size=12)
        ),
            xaxis=dict(title='Date', tickfont=dict(size=14)),
            yaxis=dict(title=f'{col} Price', tickfont=dict(size=14))
        )
    st.plotly_chart(fig)

plot_opening_price(stock_data, selected_column)


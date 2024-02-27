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

import datetime


st.title("Stock Prediction")

yf.pdr_override()

stocks = ['^NSEI', 'RELIANCE.NS', 'CIPLA.NS', 'IRFC.NS', 'ZOMATO.NS', 'ANGELONE.NS']

@st.cache_data
def fetch_data():
    global my_dict
    my_dict = dict()
    present_date = datetime.datetime.now().strftime('%Y-%m-%d')
    st.write(f'Getting Stock data from the year 2014 January to present date.')
    for stock in stocks:
        data = pdr.get_data_yahoo(f"{stock}", start="2014-01-01", end=f"{present_date}")
        my_dict[stock] = data
    return my_dict
stock_data = fetch_data()




## ------------ TESTING ------------
# my_dict = dict()
# for stock in stocks:
#     data = pdr.get_data_yahoo(f"{stock}", start="2014-01-01", end="2024-02-27")
#     my_dict[stock] = data
# st.json(my_dict)
st.dataframe(stock_data['^NSEI'], use_container_width=True)

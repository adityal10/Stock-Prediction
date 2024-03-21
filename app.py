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
from sklearn.metrics import r2_score

import datetime
from io import StringIO
import pickle

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

col1, col2 = st.columns(2)
data_load_state = col1.text("Load Data...")
stock_data = fetch_data()
data_load_state.text("Load Data...Done!")
com_df = pd.concat({k: pd.DataFrame(v) for k, v in stock_data.items()}, axis=0, names=['Stock']).reset_index(level=1)

# Convert the DataFrame to CSV string
csv_string = com_df.to_csv(index=True)

col2.download_button(
        label="Download Combined CSV",
        data=csv_string,
        file_name="combined_stock_data.csv",
        key="download_combined_csv"
    )


st.text("Raw Data - NIFTY50")
# st.dataframe(stock_data['^NSEI'].tail(), use_container_width=True)
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


def predict_model(df, num, stock_name):
    #loading the model
    model = pickle.load(open('models/linearregressionmodel.pkl', 'rb'))

    com_df = df
    stock_name = stock_name
    #calling preprocessing function to get
    #processed data
    df = preprocessing_data(com_df, num, stock_name)

    #set x_forecast equal to the last 30 rows of the original dataset from adj close column
    x_forecast = np.array(df.drop(['Prediction'], axis=1))[-num:]
    pred = model.predict(x_forecast)

    prediction_df = df.tail(num)
    prediction_df['Prediction'] = pred

    r_square = r2_score(x_forecast, pred)*100
    # st.write('R Square: ',r_square,"%")

    col1, col2 = st.columns(2)
    st.write(stock_name)
    st.write("Actual Highest price:", prediction_df['Adj Close'].max())
    st.write("Predicted Highest price:", prediction_df['Prediction'].max())
    
    return prediction_df, r_square

def preprocessing_data(df, num, stock_name):
    """
    preprocessing the data. For now we only took cipla stock value
    converts date object to datetime
    a variable for predicting 'num' days out into the future
    """
    df = df.reset_index()
    # print(df.columns)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    # st.dataframe(df)
    df = df[df['Stock']==f'{stock_name}']
    #get the adj. close price
    df = df[['Adj Close']]
    df['Prediction'] = df[['Adj Close']].shift(-num)

    return df

def plotting_predictions(predict_df, df, stock_name):
    df = df.reset_index()
    # print(df.columns)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    # st.dataframe(df)
    df = df[df['Stock']==f'{stock_name}']
    #get the adj. close price
    df = df[['Adj Close']]
    # predict_df = predict_df.reset_index()

    fig = go.Figure()
    # Add trace for each stock
    fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close'], name='Train'))
    fig.add_trace(go.Scatter(x=predict_df.index, y=predict_df['Adj Close'], name='Validation'))
    fig.add_trace(go.Scatter(x=predict_df.index, y=predict_df['Prediction'], name='Predictions'))
    # plt.plot(predict_df[['Adj Close', 'Prediction']])

    fig.update_layout(
        title_text=f'Predictions Time Series Data {stock_name}- Adj Close Prices', 
        xaxis_rangeslider_visible=True,
        width=900,  # Set the width of the plot
        height=600,   # Set the height of the plot
        legend=dict(
            title=dict(text='Data', font=dict(size=15)),
            font=dict(size=12)
        ),
            xaxis=dict(title='Date', tickfont=dict(size=14)),
            yaxis=dict(title='Price', tickfont=dict(size=14))
        )

    st.plotly_chart(fig)

st.subheader("Predictions - Time Series Plot")
col1, col2 = st.columns(2)
selected_column = col1.selectbox("Select stock for plot", stocks)
num = col2.number_input('Number of days: ', step=1, min_value=25)
if num and selected_column:
    output, r_square = predict_model(com_df, num, selected_column)
    plotting_predictions(output, com_df, selected_column)

    #display dataframe
    st.write("Dataset Overview")
    st.dataframe(output, use_container_width=True)
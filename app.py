from flask import Flask, render_template, url_for, flash, redirect, request
import time,os
from IPython.display import display

from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pmdarima.arima import AutoARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

import logging
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.display.max_rows = 500
warnings.filterwarnings('ignore', message='No supported index is available.')
warnings.filterwarnings('ignore', category=ValueWarning)

app = Flask(__name__,template_folder='template')

app.config['UPLOAD_FOLDER'] = 'static'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
n=20

users = {
    "deepak": "password",
    "isoni": "password"
}

def generate_save_plot(symbol):
    # Filtering data based on symbol
    # Reading the CSV file
    df = pd.read_csv("ts_ma.csv")
    df["date"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d")
    df.drop(['timestamp'], inplace=True, axis = 1)
    # Filtering data based on symbol
    filename = f'models/{symbol}_model.pkl'
    print(filename)
    
    # load the saved model
    with open(filename, "rb") as f:
        model = pickle.load(f)
        
    symbol_data = df[df["symbol"] == symbol]
    symbol_data = symbol_data.iloc[::-1]

    # Spliting the data into training and testing sets
    train_data = symbol_data[symbol_data["date"] <= "2022-03-31"] 
                                         
    test_data = symbol_data[symbol_data["date"] > "2022-03-31"]#.tail(12)
        

    # Making predictions on the test data
    predictions = model.predict(n_periods=12)
    
    # Evaluating the model using mean absolute error and mean squared error
    y_true = test_data["adjusted close"]
    y_pred = predictions
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print(f"Symbol: {symbol}, MAE: {mae}, MSE: {mse}")

    # Plotting actual and predicted stock prices
    dates_train = train_data["date"][train_data["date"] > "2010-12-31"]
    actual_prices_train = train_data["adjusted close"][train_data["date"] > "2010-12-31"]

    dates_test = test_data["date"]
    actual_prices_test = test_data["adjusted close"]
    predicted_prices_test = predictions

    plt.figure(figsize=(8, 6))
    plt.plot(dates_train, actual_prices_train, label="Train")
    plt.plot(dates_test, actual_prices_test, label="Actual")
    plt.plot(dates_test, predicted_prices_test, label="Predicted")
    plt.title(symbol)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.savefig(f"static/{symbol}_plot.png")
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.plot(dates_test, actual_prices_test, label="Actual")
    plt.plot(dates_test, predicted_prices_test, label="Predicted")
    plt.title(symbol)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.savefig(f"static/{symbol}_plot2.png")
    plt.show()
    
    # Evaluating model using mean absolute error and mean squared error
    metrics = ["adjusted close", "open", "low", "high"]
    plt.figure(figsize=(8, 6))
    for metric in metrics:
        y_true = test_data[metric].values
        y_pred = predictions#.to_numpy().reshape(-1, 1)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        print(f"Symbol: {symbol}, {metric} : MAE : {mae}, MSE: {mse}")
        
        # Plotting actual and predicted stock prices
        dates = test_data["date"].tolist()
        actual_prices = test_data[metric].tolist()
        predicted_prices = predictions.tolist()
        
        plt.plot(dates, actual_prices, label=f"Actual {metric}")
    plt.plot(dates, predicted_prices, label=f"Predicted")
    
    plt.title(symbol)
    plt.xlabel("Date")
    plt.ylabel("Price")
    # plt.xticks(dates, rotation=90)
    plt.legend()
    plt.savefig(f"static/{symbol}_plot3.png")
    plt.show()
    
    print(model.summary())
    return (f"static/{symbol}_plot.png",f"static/{symbol}_plot2.png",f"static/{symbol}_plot3.png")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def do_login():
    username = request.form["username"]
    password = request.form["password"]
    if username not in users or users[username] != password:
        return redirect(url_for("login"))
    else:
        return render_template("main_page.html", img_data=username)

def form():
    ticker = request.form.get("ticker")
    return ticker

# def generate_save_plot(ticker):
#     path='static/'+ticker+'.JPG'
#     return path

@app.route("/ticker",methods=['GET','POST'])
def ticker():
    if request.method == "POST":
        ticker= form()
        plot_img=generate_save_plot(ticker)
        # plot_img='static/bucket.png'
        # print(plot_img)
        if os.path.exists(plot_img[0]) and os.path.exists(plot_img[1]) and os.path.exists(plot_img[2]):
            print("Here")
            return render_template("plot.html", img_data=plot_img[0], img_data1=plot_img[1], img_data2=plot_img[2])
    return render_template("form.html")

app.run()
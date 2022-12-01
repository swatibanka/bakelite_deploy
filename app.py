import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import datetime as dt
from datetime import date, timedelta

# Create flask app
app = Flask(__name__)


@app.route("/", methods = ["POST","GET"])
def Home():
    return jsonify("Hello World")

@app.route("/predict", methods = ["GET","POST"])
def predict():
    startpoint = date.today()
    start = startpoint.strftime('%m/%d/%Y')
    endpoint = startpoint+ timedelta(days = 180)
    end = endpoint.strftime('%m/%d/%Y')

    x_future_date = pd.date_range(start = start, end = end)

    x_future_dates = pd.DataFrame()

    x_future_dates["Dates"] = pd.to_datetime(x_future_date)

    x_future_dates.index = x_future_dates["Dates"]
    df1 = x_future_dates

    def create_features(df1, label=None):
        df1['date'] = df1.index
        df1['hour'] = df1['date'].dt.hour
        df1['dayofweek'] = df1['date'].dt.dayofweek
        df1['quarter'] = df1['date'].dt.quarter
        df1['month'] = df1['date'].dt.month
        df1['year'] = df1['date'].dt.year
        df1['dayofyear'] = df1['date'].dt.dayofyear
        df1['dayofmonth'] = df1['date'].dt.day
        df1['weekofyear'] = df1['date'].dt.weekofyear
        
        X = df1[['hour','dayofweek','quarter','month','year',
            'dayofyear','dayofmonth','weekofyear']]
        if label:
            y = df1[label]
            return X, y
        return X

    X, y = create_features(x_future_dates, label='Dates')

    # First model pickle

    model_xg = pickle.load(open("boost_model.pkl", "rb"))

    y_future_total_tickets = model_xg.predict(X)
    
    #print(y_future_total_tickets)

    json_str = json.dumps( y_future_total_tickets.tolist())
    return jsonify(json_str)

@app.route("/priority", methods = ["GET","POST"])
def priority():
    startpoint = date.today()
    start = startpoint.strftime('%m/%d/%Y')
    endpoint = startpoint+ timedelta(days = 180)
    end = endpoint.strftime('%m/%d/%Y')

    x_future_date = pd.date_range(start = start, end = end)

    x_future_dates = pd.DataFrame()

    x_future_dates["Dates"] = pd.to_datetime(x_future_date)

    x_future_dates.index = x_future_dates["Dates"]
    df1 = x_future_dates

    def create_features(df1, label=None):
        df1['date'] = df1.index
        df1['hour'] = df1['date'].dt.hour
        df1['dayofweek'] = df1['date'].dt.dayofweek
        df1['quarter'] = df1['date'].dt.quarter
        df1['month'] = df1['date'].dt.month
        df1['year'] = df1['date'].dt.year
        df1['dayofyear'] = df1['date'].dt.dayofyear
        df1['dayofmonth'] = df1['date'].dt.day
        df1['weekofyear'] = df1['date'].dt.weekofyear
        
        X = df1[['hour','dayofweek','quarter','month','year',
            'dayofyear','dayofmonth','weekofyear']]
        if label:
            y = df1[label]
            return X, y
        return X

    X, y = create_features(x_future_dates, label='Dates')

    # First model pickle

    model_xg = pickle.load(open("boost_model.pkl", "rb"))

    y_future_total_tickets = model_xg.predict(X)
    
    print(y_future_total_tickets)

    # 2nd model pickle
    x_future_dates["Predicted Tickets"] = y_future_total_tickets
    x_future_dates.drop("Dates", inplace = True, axis = 1)
    model_mr = pickle.load(open("regressor_model.pkl", "rb"))
    y_future_prediction = model_mr.predict(np.array(x_future_dates["Predicted Tickets"]).reshape(181,1))
    y_future_prediction = pd.DataFrame(y_future_prediction)
    #df2 = y_future_total_tickets.to_json(orient = 'table')
    y_future_prediction.rename(columns = {0:'Very High', 
                                      1:'High',
                                      2:'Medium',
                                      3:'Low',
                                      4:'Finance'}, inplace = True)
    #print(y_future_prediction)
    df2 = y_future_prediction.to_json(orient = 'table')
    #converting nparray to list to finally convert it into json
    #future_pred = y_future_prediction.tolist()
    return (df2)


if __name__ == "__main__":
        app.run()
    #final_jsonformat = json.dumps(future_pred)

    #print(final_jsonformat)


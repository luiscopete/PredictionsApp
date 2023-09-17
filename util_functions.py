import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression


def remove_outliers(data, col) -> pd.DataFrame:
    """ Remove outliers from a list of numbers """
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    data = data[(data[col] > lower_bound) & (data[col] < upper_bound)]
    return data


def train_linear_model(data: pd.DataFrame, x_cols: list, y_col: pd.Series) -> LinearRegression:
    """ Train a model using a dataframe """
    X = data[x_cols].values
    y = data[y_col].values
    model = LinearRegression()
    model.fit(X, y)
    return model


def predict_data(model, data: pd.DataFrame, x_col: pd.Series) -> pd.DataFrame:
    """ Predict values using a trained model """
    X_pred = data[x_col].values.reshape(-1, 1)
    y_pred = model.predict(X_pred)
    data['y_pred'] = y_pred
    return data


@st.cache
def load_data():
    df = pd.read_csv('data.csv')
    return df

import numpy as np
import pandas as pd
import streamlit as st


def remove_outliers(data, cols: list) -> pd.DataFrame:
    """ Remove outliers from a list of numbers """
    for col in cols:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        data = data[(data[col] > lower_bound) & (data[col] < upper_bound)]
    return data


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

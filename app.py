import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


st.title('Modelo de regresión. :chart_with_upwards_trend:')
st.write('En esta web app podrás predecir datos de un dataset.')

st.sidebar.header('Parámetros de la predicción')

with st.sidebar:
    st.write(
        'Selecciona el archivo en formato CSV y las columnas dependientes e independientes.')
    file = st.file_uploader('Archivo CSV', type=['csv'])
    df = None
    if file is not None:
        df = pd.read_csv(file)
        cols = df.columns.tolist()
        cols.insert(0, 'Sin seleccionar')
        st.write('Selecciona las columnas dependientes e independientes')
        x_col = st.selectbox('Columna independiente', cols)
        y_col = st.selectbox('Columna dependiente', cols)


if df is not None:
    st.write('Preview de los datos')
    st.dataframe(df.head())
    if x_col != 'Sin seleccionar' and y_col != 'Sin seleccionar':
        fig, ax = plt.subplots(figsize=(4, 4))
        st.write('Gráfica de los datos')
        plt.scatter(df[x_col], df[y_col])
        st.pyplot(fig)

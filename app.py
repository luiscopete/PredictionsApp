import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


st.title('Modelo de regresión. :chart_with_upwards_trend:')
st.write('En esta web app podrás predecir datos de un dataset.')

st.sidebar.header('Parámetros de la predicción')

# Elementos de la barra lateral
with st.sidebar:
    st.write(
        'Selecciona el archivo en formato CSV y las columnas dependientes e independientes.')
    file = st.file_uploader('Archivo CSV', type=['csv'])
    df = None
    if file is not None:
        df = pd.read_csv(file)
        cols = df.columns.tolist()
        cols = [col for col in cols if df[col].dtype in [
            np.int64, np.int32, np.float64, np.float32]]
        cols.insert(0, 'Sin seleccionar')
        st.write('Selecciona las columnas dependientes e independientes')
        x_col = st.selectbox('Columna independiente', cols)
        y_col = st.selectbox('Columna dependiente', cols)
        entrenar_modelo = None
        if x_col != 'Sin seleccionar' and y_col != 'Sin seleccionar':
            if st.button('Entrenar modelo'):
                entrenar_modelo = True
        if entrenar_modelo is True:
            df = df.dropna()
            X = df[x_col].values.reshape(-1, 1)
            y = df[y_col].values.reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            st.write('Modelo entrenado')

# Elementos de la página principal
if df is not None:
    st.write('Preview de los datos')
    st.dataframe(df.head())
    graficar_modelo = None
    if entrenar_modelo is True:
        progress_text = 'Graficando recta de regresión...'
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.05)
            my_bar.progress(percent_complete + 1, text=progress_text)
        fig, ax = plt.subplots(figsize=(4, 4))
        st.write('Gráfica de los datos')
        plt.scatter(df[x_col], df[y_col])
        plt.plot(df[x_col], y_pred, color='red', label='Regresión lineal')
        plt.title('Gráfica de los datos')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.legend()
        st.pyplot(fig)

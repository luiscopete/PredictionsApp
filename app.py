import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import util_functions as uf


st.title('Modelo de regresión. :chart_with_upwards_trend:')
st.write('Bienvenido a esta web app de predicción. Aquí podrás predecir datos utilizando un modelo de regresión.')

cover_image_path = 'images/data.png'
st.sidebar.image(cover_image_path, width=200)
st.sidebar.header('Parámetros de la predicción')

# Elementos de la barra lateral
with st.sidebar:
    st.write(
        'Selecciona el archivo en formato CSV y configura los parámetros de la predicción.')
    file = st.file_uploader('Cargar un archivo CSV', type=['csv'])
    df = None
    if file is not None:
        df = pd.read_csv(file)
        st.session_state.df = df
        cols = df.columns.tolist()
        cols = [col for col in cols if df[col].dtype in [
            np.int64, np.int32, np.float64, np.float32]]
        cols.insert(0, 'Sin seleccionar')
        st.write('Selecciona las columnas dependientes e independientes')
        x_col = st.selectbox('Columna independiente (X)', cols)
        y_col = st.selectbox('Columna dependiente (y)', cols)
        remover_outliers = st.checkbox('Remover outliers')

        entrenar_modelo = None
        if x_col != 'Sin seleccionar' and y_col != 'Sin seleccionar' and file:
            if st.button('Entrenar modelo'):
                entrenar_modelo = True
        if entrenar_modelo is True:
            if remover_outliers is True:
                df = uf.remove_outliers(df, [x_col, y_col])
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
        # graficar recta con seaborn
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.set_theme()
        sns.set_style('whitegrid')
        sns.regplot(x=x_col, y=y_col, data=df)
        fig.savefig('regression_plot.png')  # save the figure
        st.image('regression_plot.png')  # display the saved figure

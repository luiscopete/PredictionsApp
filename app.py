# -------- interface ------------
import streamlit as st
from streamlit_option_menu import option_menu
# ----- data visualization -----
import matplotlib.pyplot as plt
import seaborn as sns
# ----- Data processing --------
import pandas as pd
import numpy as np
# ---- Machine learning --------
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# ----- other ------------------
import time
import util_functions as uf


def main():

    # -------------------- Main page configuration --------------------
    st.image('main_logo.png', width=700)
    # st.title('¡Bienvenido a PredictionsApp!')
    with open('messages.txt', 'r', encoding='utf-8') as f:
        message = f.read()
    st.write(message)
    # st.write('¡Bienvenido a nuestra plataforma de predicción! Aquí puedes cargar un archivo de datos y seleccionar las variables que deseas utilizar para entrenar tu modelo de regresión. También tienes la opción de ingresar manualmente los datos que quieras predecir. Simplemente elige tus variables, carga tus datos y ¡prepárate para obtener predicciones precisas en un abrir y cerrar de ojos!')
    st.markdown("---")

    # ------------------- Elementos de la barra lateral -------------------
    with st.sidebar:
        selected = option_menu(
            'Selecciona el tipo de regresión deseada',
            ['Regresión Lineal', 'Regresión Logistica'],
            orientation='vertical',
            key='menu_principal',
            menu_icon='chart_with_upwards_trend')
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
            # seleccion de columnas
            st.write('Selecciona las columnas dependientes e independientes')
            # seleccionar multiples columnas
            x_col = st.multiselect('Columnas independientes (X)', cols)
            y_col = st.selectbox('Columna dependiente (y)', cols)
            if y_col in x_col:
                st.error(
                    'La columna dependiente no puede ser igual a la columna independiente')
                y_col = 'Sin seleccionar'
            remover_outliers = st.checkbox('Remover outliers')

            entrenar_modelo = None
            if x_col != 'Sin seleccionar' and y_col != 'Sin seleccionar' and file:
                if st.button('Entrenar modelo'):
                    entrenar_modelo = True
            if entrenar_modelo is True:
                if remover_outliers is True:
                    df = uf.remove_outliers(df, y_col)
                if selected == 'Regresión Lineal':
                    df = df.dropna()
                    model = uf.train_linear_model(df, x_col, y_col)
                    st.write('Modelo entrenado')

    # -------------------- Elementos de la página principal --------------------
    if df is not None:
        st.write('Preview de los datos')
        st.dataframe(df.head())
        st.write('Parametros de la predicción')
        if len(x_col) > 0:
            st.write(
                'Escribe los valores de las columnas independientes para predecir')
            valores_ingresados = {}
            for col in x_col:
                event_name = f'Ingresa el valor para {col}'
                st.number_input(event_name, key=event_name)
                valor_ingresado = st.session_state.get(event_name)
                valores_ingresados[col] = valor_ingresado
        graficar_modelo = None
        if entrenar_modelo is True:
            progress_text = 'Calculando...'
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):  # barra de progreso
                time.sleep(0.001)
                my_bar.progress(percent_complete + 1, text=progress_text)
            if selected == 'Regresión Lineal':
                r2 = model.score(df[x_col], df[y_col])
                intercept = model.intercept_
                coef = model.coef_
                st.write(
                    'Coeficiente de determinación (R2): {:.2f}'.format(r2))
                st.write(
                    'Coeficientes de la intersección (b): {}'.format(intercept))
                st.write('Coeficientes de la pendiente (m): {}'.format(coef))
                # usar valores ingresados para predecir
                predict_df = pd.DataFrame(valores_ingresados, index=[0])
                y_pred = model.predict(predict_df)
                st.write(
                    f'El valor calculado para la variable {y_col} es {y_pred}')
                st.markdown("---")
                for x in x_col:
                    df['y_pred'] = model.predict(df[x_col])
                    fig, ax = plt.subplots()
                    sns.scatterplot(x=x, y=y_col, data=df)
                    sns.lineplot(x=x, y='y_pred', data=df, color='red')
                    plt.xlabel(x)
                    plt.ylabel(y_col)
                    plt.title(
                        f'Regresión lineal entre {x} y {y_col}')
                    st.pyplot(fig)
            if selected == 'Regresión Logistica':
                pass


if __name__ == '__main__':
    main()
    with st.sidebar:
        st.markdown("---")
        st.markdown(
            '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://github.com/luiscopete">luiscopete</a></h6>',
            unsafe_allow_html=True)
        st.markdown(
            '<a href="https://www.linkedin.com/in/luiscopete/"><img src="https://www.edigitalagency.com.au/wp-content/uploads/Linkedin-logo-png.png" alt="LinkedIn logo" height="16"></a>',
            unsafe_allow_html=True)

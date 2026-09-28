import streamlit as st

import cuenta
import estilo as E
import sesion as S

st.set_page_config(page_title=E.NOMBRE_APP, page_icon="favicon.png", layout="wide", initial_sidebar_state="auto")
E.aplicar_estilo()

inicio = st.Page("paginas/inicio.py", title="Inicio", icon=":material/home:", default=True)
secciones = {
    "Pronosticar": [
        st.Page("paginas/datos.py", title="1. Tus datos", icon=":material/upload_file:"),
        st.Page("paginas/pronostico.py", title="2. Pronóstico", icon=":material/show_chart:"),
        st.Page("paginas/decisiones.py", title="3. Decisiones", icon=":material/inventory_2:"),
        st.Page("paginas/escenarios.py", title="4. Escenarios", icon=":material/thunderstorm:"),
    ],
    "Explorar": [st.Page("paginas/analisis.py", title="Análisis de tus datos", icon=":material/insights:")],
}
if cuenta.login_disponible():
    secciones["Tu cuenta"] = [st.Page("paginas/mis_pronosticos.py", title="Mis pronósticos", icon=":material/folder_open:")]
if S.modo_dev():
    secciones["Desarrollador"] = [st.Page("paginas/tecnico.py", title="Detalles técnicos", icon=":material/code:")]

st.logo("favicon.png", size="large")
# El menú se dibuja a mano para poder poner la cuenta justo debajo de Inicio
pg = st.navigation([inicio] + [p for ps in secciones.values() for p in ps], position="hidden")
with st.sidebar:
    st.page_link(inicio)
cuenta.caja_cuenta()
with st.sidebar:
    for nombre, ps in secciones.items():
        st.markdown(f'<div class="seccion-menu">{nombre}</div>', unsafe_allow_html=True)
        for p in ps:
            st.page_link(p)
    st.divider()
pg.run()
E.pie_pagina()

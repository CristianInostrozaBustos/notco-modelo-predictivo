import streamlit as st

import estilo as E
from sesion import panel_dataset

panel_dataset()

st.markdown(
    f"""
    <div class="hero">
      <div class="eyebrow">{E.NOMBRE_APP}</div>
      <h1>Pronostica la demanda de cualquier negocio y decide cuánto abastecer</h1>
      <p>Sube tu historial de ventas o demanda. El motor detecta la estructura de los datos, elige solo cómo
      entrenar una red LSTM según cuántos datos tienes y entrega pronósticos con rango de incertidumbre,
      política de inventario y escenarios what-if.</p>
      <div class="chips">
        <span class="chip">CSV o Excel</span>
        <span class="chip">Diario, semanal o mensual</span>
        <span class="chip">Una serie o miles</span>
        <span class="chip">Pronóstico P10 · P50 · P90</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
st.markdown("### Cómo funciona")
pasos = [
    ("1", "Sube tus datos", "CSV o Excel con tu historial. El sistema reconoce las columnas y limpia huecos y duplicados."),
    ("2", "Elige cuánto pronosticar", "Días, semanas o meses hacia adelante. El sistema elige solo la mejor forma de entrenar."),
    ("3", "Mira el pronóstico", "El valor más probable y el rango en que se moverá la demanda, con su precisión medida."),
    ("4", "Decide qué pedir", "Stock de seguridad, punto de reorden, cuánto pedir y cuándo, para cada producto."),
]
cols = st.columns(4)
for col, (n, titulo, texto) in zip(cols, pasos):
    col.markdown(f'<div class="paso"><div class="n">{n}</div><h4>{titulo}</h4><p>{texto}</p></div>',
                 unsafe_allow_html=True)

st.write("")
with st.container(border=True):
    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    c1.markdown("**Empieza por tus datos**")
    c1.caption("Sube un CSV o Excel, o prueba con uno de los datasets de ejemplo.")
    with c2:
        st.page_link("paginas/datos.py", label="Ir a Datos", icon=":material/arrow_forward:")

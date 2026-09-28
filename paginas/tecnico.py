"""Panel de desarrollador (visible solo con ?dev=1)."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import estilo as E
import sesion as S
from motor import datos as D
from motor import reglas as R

S.panel_dataset()
E.encabezado("Modo desarrollador", "Detalles técnicos",
             "Lo que el sistema decidió y midió por dentro. Esta página no aparece para los usuarios.")

dp = st.session_state.get("dp")
if dp is None:
    st.info("Carga un dataset primero.")
    st.stop()

plan = R.planificar(dp)
st.markdown("## Plan de entrenamiento")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Modo", plan.modo)
c2.metric("Esquema", plan.esquema)
c3.metric("Bloque de validación", f"{plan.validacion} {dp.freq_info['unidad_pl']}")
c4.metric("Horizonte máximo", f"{plan.horizonte_max} {dp.freq_info['unidad_pl']}")
st.caption(plan.arquitectura["descripcion"])
for r in plan.razones:
    st.markdown(f"- {r}")

st.markdown("## Preparación de los datos")
st.markdown("**Roles:** " + " · ".join(f"{k} = `{v}`" for k, v in dp.etiquetas.items()))
st.markdown("**Variables del modelo:** " + ", ".join(f"`{v}`" for v in dp.variables_modelo) + " + calendario")
for linea in dp.reporte:
    st.markdown(f"- {linea}")
resumen = D.resumen_por_entidad(dp)
resumen["historia"] = resumen["registros"].apply(lambda n: R.clase_entidad(n, dp.config.frecuencia))
st.dataframe(resumen, width="stretch", hide_index=True)

_, res = S.resultado()
if res is None:
    st.info("Todavía no se entrena un modelo para este dataset.")
    st.stop()

st.markdown("## Búsqueda de ventana")
st.caption(f"Ventana elegida: {res.ventana} {dp.freq_info['unidad_pl']} · épocas del modelo final: {res.epocas} · "
           f"tiempo total: {res.segundos:.0f} s")
st.dataframe(res.busqueda.round(2), width="stretch", hide_index=True)

st.markdown("## Métricas en el bloque de prueba")
st.caption("wape/mape/cobertura: pronóstico recursivo de todo el bloque (como se usa hacia el futuro). "
           "mape_1paso: un período adelante con datos reales, comparable con la validación clásica.")
st.dataframe(res.metricas_entidad.round(2), width="stretch", hide_index=True)

h = res.historial_perdida
if h:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=h["loss"], name="Entrenamiento", line=dict(color=E.AZUL)))
    if "val_loss" in h:
        fig.add_trace(go.Scatter(y=h["val_loss"], name="Selección", line=dict(color=E.NARANJO)))
    fig.update_layout(title="Curva de pérdida del modelo final (pinball)", xaxis_title="Época", height=320)
    E.grafico(fig, key="fig_loss")

st.markdown("## Indicadores disponibles")
st.dataframe(R.tabla_indicadores(R.indicadores(dp)), width="stretch", hide_index=True)

st.markdown("## Modelos guardados")
alm = S.almacen_persistente()
st.caption(f"Archivos: {alm.nombre} · Registro de usuarios: {S.repositorio().nombre}. "
           "Configurable en los secrets del sitio (sección [almacen]).")
try:
    lista = alm.listar()
    st.dataframe(pd.DataFrame(lista), width="stretch", hide_index=True) if lista else st.caption("Todavía no hay modelos guardados.")
except Exception as e:  # noqa: BLE001
    st.error(f"No se pudo listar: {e}")
st.caption(f"Origen del modelo actual: {st.session_state.get('resultado', {}).get('origen', '—')}")

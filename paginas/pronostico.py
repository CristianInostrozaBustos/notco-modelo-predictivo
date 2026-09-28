import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import estilo as E
import sesion as S

dp, res = S.requiere_pronostico()
S.panel_dataset()
fut = S.pronostico()
entidades = list(fut)
fi = dp.freq_info
H = S.horizonte()
obj = dp.etiquetas["objetivo"]

TOTAL = "Todas (total)"
opciones = ([TOTAL] if len(entidades) > 1 else []) + entidades
if st.session_state.get("entidad") not in opciones:
    st.session_state["entidad"] = opciones[0]
ent = S.selector_entidad(opciones, dp)

E.encabezado(
    "Paso 2",
    "Pronóstico",
    f"{obj} esperado para los próximos {H} {fi['unidad_pl']}, con el rango en que probablemente se moverá.",
)

# ---------------------------------------------------------------- datos de la vista
hist = dp.df if ent == TOTAL else dp.df[dp.df["entidad"] == ent]
hist = hist.groupby("fecha", as_index=False)["objetivo"].sum()
if ent == TOTAL:
    pr = pd.concat(fut.values()).groupby("fecha", as_index=False)[["P10", "P50", "P90"]].sum()
else:
    pr = fut[ent]
anterior = hist["objetivo"].tail(H).sum()
total_p50 = pr["P50"].sum()
var = (total_p50 - anterior) / anterior * 100 if anterior > 0 else None

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Total esperado ({H} {fi['unidad_pl']})", E.num(total_p50))
c2.metric(f"Promedio por {fi['unidad']}", E.num(pr["P50"].mean()))
c3.metric(f"Vs. últimos {H} {fi['unidad_pl']}", E.pct(var) if var is not None else "—",
          delta=("sube" if var > 0 else "baja") if var is not None else None,
          delta_color="off", delta_arrow="off")
c4.metric("Escenario alto (P90)", E.num(pr["P90"].sum()), delta=f"bajo (P10): {E.num(pr['P10'].sum())}",
          delta_color="off", delta_arrow="off",
          help="Rango probable del total entre el escenario bajo y el alto. En la vista total es una aproximación (suma de rangos).")

with st.container(border=True):
    atras = min(len(hist), max(3 * H, {"D": 120, "W": 52, "M": 36, "Q": 12}[dp.config.frecuencia]))
    h_ver = hist.tail(atras)
    fig = E.fig_banda(pr["fecha"], pr["P10"], pr["P50"], pr["P90"], nombre_banda="Rango probable (P10–P90)",
                      nombre_p50="Pronóstico")
    fig.add_trace(go.Scatter(x=h_ver["fecha"], y=h_ver["objetivo"], name="Historial",
                             line=dict(color=E.TINTA, width=1.5), hovertemplate="%{y:,.0f}"))
    fig.add_vline(x=h_ver["fecha"].iloc[-1], line=dict(color=E.EJE, width=1, dash="dot"))
    fig.update_layout(title=f"{obj} · {ent}", yaxis_title=obj, height=440)
    E.grafico(fig, key="fig_pron")
    supuestos = []
    if dp.tiene("precio"):
        supuestos.append("el precio se mantiene en su último valor")
    if dp.tiene("promocion"):
        supuestos.append("no hay promociones")
    st.caption("El pronóstico (azul) es el valor más probable. En 8 de cada 10 períodos la realidad debería caer dentro de la banda."
               + (" Supone que " + " y ".join(supuestos) + "." if supuestos else ""))

# ---------------------------------------------------------------- vistas opcionales
st.write("")
vistas = st.pills(
    "Qué más quieres ver",
    ["Tabla del pronóstico", "Qué tan preciso es", "Prueba con datos pasados"],
    selection_mode="multi", default=["Tabla del pronóstico"], key="vistas_pron",
)

if "Tabla del pronóstico" in (vistas or []):
    st.markdown("### Tabla del pronóstico")
    tabla = pr.copy()
    tabla["fecha"] = tabla["fecha"].dt.strftime("%d/%m/%Y")
    tabla = tabla.rename(columns={"fecha": "Fecha", "P50": "Pronóstico", "P10": "Escenario bajo (P10)",
                                  "P90": "Escenario alto (P90)"})[["Fecha", "Pronóstico", "Escenario bajo (P10)", "Escenario alto (P90)"]]
    st.dataframe(tabla, width="stretch", hide_index=True, height=min(420, 38 + 35 * len(tabla)),
                 column_config={c: st.column_config.NumberColumn(format="%.0f") for c in tabla.columns[1:]})

met = res.metricas_entidad.set_index("entidad")
if "Qué tan preciso es" in (vistas or []):
    st.markdown("### Qué tan preciso es")
    st.caption(f"Medido pronosticando los últimos {res.plan.validacion} {fi['unidad_pl']} de tu historial como si "
               "no los conociéramos, y comparando con lo que realmente pasó.")
    if ent == TOTAL:
        m = met
        wape = (m["wape"] * 1).mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("Error promedio", E.pct(wape), help="Promedio entre entidades del error absoluto sobre el total real (WAPE).")
        c2.metric("Aciertos dentro del rango", E.pct(m["cobertura"].mean()))
        c3.metric("Mejora vs. repetir la temporada anterior", E.pct((m["wape_naive"].mean() - wape) / m["wape_naive"].mean() * 100))
        vista = pd.DataFrame({
            S.nombre_entidad(dp).capitalize(): m.index,
            "Error": m["wape"].map(lambda v: E.pct(v)),
            "Dentro del rango": m["cobertura"].map(lambda v: E.pct(v)),
            "Error sin modelo": m["wape_naive"].map(lambda v: E.pct(v)),
        })
        st.dataframe(vista, width="stretch", hide_index=True)
    else:
        m = met.loc[ent]
        c1, c2, c3 = st.columns(3)
        c1.metric("Error promedio", E.pct(m["wape"]),
                  help="Suma de los errores absolutos dividida por la demanda real total (WAPE).")
        c2.metric("Aciertos dentro del rango", E.pct(m["cobertura"]))
        mejora = (m["wape_naive"] - m["wape"]) / m["wape_naive"] * 100
        c3.metric("Mejora vs. repetir la temporada anterior", E.pct(mejora),
                  delta="mejor" if mejora > 0 else "peor", delta_color="normal" if mejora > 0 else "inverse")

if "Prueba con datos pasados" in (vistas or []):
    st.markdown("### Prueba con datos pasados")
    if ent == TOTAL:
        bt = pd.concat(res.backtest.values()).groupby("fecha", as_index=False)[["real", "P10", "P50", "P90", "naive"]].sum()
    else:
        bt = res.backtest[ent]
    with st.container(border=True):
        fig = E.fig_banda(bt["fecha"], bt["P10"], bt["P50"], bt["P90"], nombre_p50="Lo que pronosticó")
        fig.add_trace(go.Scatter(x=bt["fecha"], y=bt["real"], name="Lo que pasó", line=dict(color=E.TINTA, width=1.8),
                                 hovertemplate="%{y:,.0f}"))
        fig.add_trace(go.Scatter(x=bt["fecha"], y=bt["naive"], name="Sin modelo (temporada anterior)",
                                 line=dict(color=E.NARANJO, width=1.4, dash="dot"), hovertemplate="%{y:,.0f}"))
        fig.update_layout(title=f"Pronóstico vs. realidad · últimos {res.plan.validacion} {fi['unidad_pl']}", height=400)
        E.grafico(fig, key="fig_bt")

# ---------------------------------------------------------------- descarga
todo = pd.concat([f.assign(entidad=e) for e, f in fut.items()])
todo = todo[["entidad", "fecha", "P50", "P10", "P90"]].rename(columns={
    "entidad": S.nombre_entidad(dp).capitalize(), "fecha": "Fecha", "P50": "Pronóstico",
    "P10": "Escenario bajo (P10)", "P90": "Escenario alto (P90)"}).round(1)
todo["Fecha"] = todo["Fecha"].dt.date
st.download_button("Descargar pronóstico (Excel)", E.excel_bytes({"Pronóstico": todo}),
                   file_name="pronostico.xlsx", icon=":material/download:",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

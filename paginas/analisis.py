import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import estilo as E
import sesion as S

dp = st.session_state.get("dp")
S.panel_dataset()
if dp is None:
    st.info("Primero carga tus datos.", icon=":material/info:")
    st.page_link("paginas/datos.py", label="Ir a Datos", icon=":material/arrow_forward:")
    st.stop()

fi = dp.freq_info
obj = dp.etiquetas["objetivo"]
nom = S.nombre_entidad(dp).capitalize()
n_ent = dp.df["entidad"].nunique()
df = dp.df

E.encabezado("Explorar", "Explora tus datos", "Elige qué quieres analizar.")

disponibles = ["Estacionalidad"]
if n_ent > 1:
    disponibles.append("Ranking ABC")
if dp.tiene("promocion"):
    disponibles.append("Efecto de las promociones")
if dp.tiene("quiebre") or dp.tiene("inventario"):
    disponibles.append("Quiebres de stock")
if n_ent > 1:
    disponibles.append("Variabilidad por " + nom.lower())

elegidas = st.pills("Análisis", disponibles, selection_mode="multi", default=[disponibles[0]],
                    label_visibility="collapsed", key="pills_analisis")
no_disp = []
if not dp.tiene("promocion"):
    no_disp.append("efecto de promociones (requiere columna de promoción)")
if not (dp.tiene("quiebre") or dp.tiene("inventario")):
    no_disp.append("quiebres de stock (requiere inventario o quiebre)")
if no_disp:
    st.caption(":material/info: También disponible con más columnas: " + "; ".join(no_disp) + ".")

ents = sorted(df["entidad"].unique())


def colores(lista):
    return {e: E.SERIES[i % len(E.SERIES)] for i, e in enumerate(lista)}


# ---------------------------------------------------------------- estacionalidad
if "Estacionalidad" in (elegidas or []):
    st.markdown("## Estacionalidad")
    sel = st.selectbox(nom, ["Todas"] + ents, key="est_ent") if n_ent > 1 else ents[0]
    sub = df if sel == "Todas" else df[df["entidad"] == sel]
    serie = sub.groupby("fecha")["objetivo"].sum()
    cols = st.columns(2 if dp.config.frecuencia == "D" else 1)
    with cols[0].container(border=True):
        por_mes = serie.groupby(serie.index.month).mean()
        meses = [E.MESES_ES[m - 1][:3].capitalize() for m in por_mes.index]
        idx = por_mes / por_mes.mean() * 100
        fig = go.Figure(go.Bar(x=meses, y=idx, marker=dict(color=[E.AZUL if v >= 100 else E.GRILLA for v in idx],
                                                               cornerradius=4),
                               hovertemplate="%{y:.0f} (100 = promedio)<extra></extra>"))
        fig.add_hline(y=100, line=dict(color=E.EJE, width=1))
        fig.update_layout(title="Índice por mes (100 = promedio)", height=320, hovermode="closest")
        E.grafico(fig, key="fig_mes")
    if dp.config.frecuencia == "D":
        with cols[1].container(border=True):
            dias = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
            por_dia = serie.groupby(serie.index.dayofweek).mean()
            idx = por_dia / por_dia.mean() * 100
            fig = go.Figure(go.Bar(x=[dias[i] for i in por_dia.index], y=idx,
                                   marker=dict(color=[E.AZUL if v >= 100 else E.GRILLA for v in idx], cornerradius=4),
                                   hovertemplate="%{y:.0f}<extra></extra>"))
            fig.add_hline(y=100, line=dict(color=E.EJE, width=1))
            fig.update_layout(title="Índice por día de la semana", height=320, hovermode="closest")
            E.grafico(fig, key="fig_dia")
    alto = [meses[i] for i, v in enumerate(por_mes / por_mes.mean()) if v >= 1.05]
    bajo = [meses[i] for i, v in enumerate(por_mes / por_mes.mean()) if v <= 0.95]
    if alto or bajo:
        E.nota(("Meses altos: <b>" + ", ".join(alto) + "</b>. " if alto else "") +
               ("Meses bajos: <b>" + ", ".join(bajo) + "</b>." if bajo else ""))

# ---------------------------------------------------------------- ABC
if "Ranking ABC" in (elegidas or []):
    st.markdown("## Ranking ABC")
    if dp.tiene("precio"):
        valor = (df["objetivo"] * df["precio"]).groupby(df["entidad"]).sum()
        medida = "ingresos"
    else:
        valor = df.groupby("entidad")["objetivo"].sum()
        medida = obj
    abc = valor.sort_values(ascending=False).to_frame("valor")
    abc["participacion"] = abc["valor"] / abc["valor"].sum() * 100
    abc["acumulado"] = abc["participacion"].cumsum()
    abc["clase"] = np.where(abc["acumulado"] - abc["participacion"] < 80, "A",
                            np.where(abc["acumulado"] - abc["participacion"] < 95, "B", "C"))
    color_clase = {"A": E.AZUL, "B": E.AQUA, "C": E.GRILLA}
    with st.container(border=True):
        top = abc.head(30)
        fig = go.Figure(go.Bar(x=top.index.astype(str), y=top["participacion"],
                               marker=dict(color=[color_clase[c] for c in top["clase"]], cornerradius=4),
                               customdata=top["clase"], hovertemplate="%{y:.1f}% · clase %{customdata}<extra></extra>"))
        fig.add_trace(go.Scatter(x=top.index.astype(str), y=top["acumulado"], mode="lines+markers", name="Acumulado",
                                 line=dict(color=E.TINTA, width=1.5), marker=dict(size=6), hovertemplate="%{y:.1f}%<extra></extra>"))
        fig.update_layout(title=f"Participación en {medida}" + (" (30 principales)" if len(abc) > 30 else ""),
                          yaxis_ticksuffix="%", height=360, showlegend=False, hovermode="closest")
        E.grafico(fig, key="fig_abc")
    c = abc["clase"].value_counts()
    E.nota(f"Clase A: <b>{c.get('A', 0)}</b> {nom.lower()}(s) concentran el 80% de los {medida}. "
           f"Clase B: {c.get('B', 0)} · Clase C: {c.get('C', 0)}.")

# ---------------------------------------------------------------- promociones
if "Efecto de las promociones" in (elegidas or []):
    st.markdown("## Efecto de las promociones")
    g = df.groupby(["entidad", "promocion"])["objetivo"].mean().unstack()
    g = g.rename(columns={0: "sin", 1: "con"}).dropna()
    g["efecto"] = (g["con"] / g["sin"] - 1) * 100
    total = (df[df["promocion"] == 1]["objetivo"].mean() / df[df["promocion"] == 0]["objetivo"].mean() - 1) * 100
    c1, c2 = st.columns([1, 2.4], gap="large")
    c1.metric("Aumento promedio con promoción", E.pct(total))
    c1.metric("Períodos con promoción", E.pct(df["promocion"].mean() * 100))
    with c2.container(border=True):
        g = g.sort_values("efecto")
        fig = go.Figure(go.Bar(y=g.index.astype(str), x=g["efecto"], orientation="h",
                               marker=dict(color=E.AZUL, cornerradius=4), hovertemplate="%{x:+.1f}%<extra></extra>"))
        fig.update_layout(title=f"Aumento de {obj} en promoción por {nom.lower()}", xaxis_ticksuffix="%",
                          height=max(240, 34 * len(g) + 90), hovermode="closest")
        E.grafico(fig, key="fig_promo")

# ---------------------------------------------------------------- quiebres
if "Quiebres de stock" in (elegidas or []):
    st.markdown("## Quiebres de stock")
    q = df["quiebre"] if dp.tiene("quiebre") else (df["inventario"] <= 0).astype(int)
    tasa = q.groupby(df["entidad"]).mean() * 100
    c1, c2 = st.columns([1, 2.4], gap="large")
    c1.metric("Tasa de quiebre global", E.pct(q.mean() * 100))
    c1.caption("Porcentaje de períodos sin stock." if dp.tiene("quiebre") else "Períodos con inventario en cero.")
    with c2.container(border=True):
        tasa = tasa.sort_values()
        fig = go.Figure(go.Bar(y=tasa.index.astype(str), x=tasa.values, orientation="h",
                               marker=dict(color=E.ROJO, cornerradius=4), hovertemplate="%{x:.1f}%<extra></extra>"))
        fig.update_layout(title=f"Tasa de quiebre por {nom.lower()}", xaxis_ticksuffix="%",
                          height=max(240, 34 * len(tasa) + 90), hovermode="closest")
        E.grafico(fig, key="fig_quiebre")

# ---------------------------------------------------------------- variabilidad
if "Variabilidad por " + nom.lower() in (elegidas or []):
    st.markdown("## Variabilidad por " + nom.lower())
    v = df.groupby("entidad")["objetivo"].agg(["mean", "std"])
    v["cv"] = v["std"] / v["mean"].replace(0, np.nan)
    with st.container(border=True):
        fig = go.Figure(go.Scatter(x=v["mean"], y=v["cv"], mode="markers+text" if len(v) <= 15 else "markers",
                                   text=v.index.astype(str), textposition="top center",
                                   marker=dict(size=11, color=E.AZUL, line=dict(color="white", width=1.5)),
                                   hovertemplate="%{text}<br>promedio %{x:,.0f} · CV %{y:.2f}<extra></extra>"))
        fig.update_layout(title="Volumen vs. variabilidad (más arriba = más difícil de pronosticar)",
                          xaxis_title=f"{obj} promedio por {fi['unidad']}", yaxis_title="Coeficiente de variación",
                          height=400, hovermode="closest")
        E.grafico(fig, key="fig_cv")

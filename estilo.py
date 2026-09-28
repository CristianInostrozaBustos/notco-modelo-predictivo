"""Identidad visual del sitio: colores, CSS, plantilla de gráficos y componentes."""

import io
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

NOMBRE_APP = "Motor Predictivo de Abastecimiento"
SUBTITULO_APP = "Pronóstico de demanda y política de inventarios para cualquier dataset"

# ---------------------------------------------------------------- paleta
# Paleta categórica validada para daltonismo (orden fijo, nunca rotado)
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TINTA_MUTED = "#898781"
GRILLA = "#e1e0d9"
EJE = "#c3c2b7"
SUPERFICIE = "#ffffff"
FONDO = "#f7f7f5"

AZUL = "#2a78d6"
AZUL_OSCURO = "#1c5cab"
AZUL_BANDA = "rgba(42,120,214,0.16)"
NARANJO = "#eb6834"
AMARILLO = "#eda100"
ROSA = "#e87ba4"
VERDE = "#008300"
AQUA = "#1baf7a"
ROJO = "#d03b3b"

SERIES = [AZUL, NARANJO, AQUA, AMARILLO, ROSA, VERDE, "#4a3aa7", "#e34948"]

MESES_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
            "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


def color_sku(sku, skus):
    """El color sigue al producto, no a su posición en un filtro."""
    return SERIES[sorted(skus).index(sku) % len(SERIES)]


# ---------------------------------------------------------------- formato chileno

def num(x, dec=0):
    """1234567.8 -> '1.234.568' ; con dec=2 -> '1.234.567,80'."""
    s = f"{x:,.{dec}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def pct(x, dec=1):
    return f"{num(x, dec)}%"


def clp(x):
    signo = "−" if x < 0 else ""
    return f"{signo}${num(abs(x))}"


def mes_es(periodo):
    ts = periodo.to_timestamp() if hasattr(periodo, "to_timestamp") else pd.Timestamp(periodo)
    return f"{MESES_ES[ts.month - 1].capitalize()} {ts.year}"


# ---------------------------------------------------------------- plantilla Plotly

def registrar_plantilla():
    fuente = "Inter, system-ui, -apple-system, Segoe UI, sans-serif"
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family=fuente, size=13, color=TINTA_2),
        title=dict(font=dict(size=15, color=TINTA), x=0, xanchor="left", xref="container", y=1, yref="container", yanchor="top", pad=dict(t=12, l=4)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=SERIES,
        margin=dict(l=8, r=8, t=72, b=8),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", bordercolor=GRILLA, font=dict(family=fuente, size=12, color=TINTA)),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=TINTA_2)),
        xaxis=dict(showgrid=False, linecolor=EJE, ticks="outside", tickcolor=EJE,
                   tickfont=dict(color=TINTA_MUTED), title=dict(font=dict(color=TINTA_MUTED, size=12)),
                   zeroline=False, automargin=True),
        yaxis=dict(gridcolor=GRILLA, gridwidth=1, zeroline=False, linecolor="rgba(0,0,0,0)",
                   tickfont=dict(color=TINTA_MUTED), title=dict(font=dict(color=TINTA_MUTED, size=12)),
                   separatethousands=True, automargin=True),
        separators=",.",
    )
    t.data.scatter = [go.Scatter(line=dict(width=2))]
    pio.templates["motor"] = t
    pio.templates.default = "motor"


# ---------------------------------------------------------------- CSS

CSS = """
<style>

.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1240px; }
h1, h2, h3 { letter-spacing: -0.02em; color: #0b0b0b; }
h1 { font-weight: 800 !important; }
h2 { font-weight: 700 !important; font-size: 1.45rem !important; }
h3 { font-weight: 650 !important; font-size: 1.12rem !important; }

/* Tarjetas KPI */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid rgba(11,11,11,0.08);
    border-radius: 14px;
    padding: 16px 18px 14px 18px;
    box-shadow: 0 1px 2px rgba(11,11,11,0.04);
}
[data-testid="stMetricLabel"] p { font-size: 0.82rem !important; color: #52514e !important; font-weight: 500; }
[data-testid="stMetricValue"] { font-size: 1.75rem !important; font-weight: 700; color: #0b0b0b; letter-spacing: -0.02em; }

/* Contenedores con borde como tarjetas blancas */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {
    border-radius: 16px;
}
div[data-testid="stVerticalBlockBorderWrapper"] { background: #ffffff; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid rgba(11,11,11,0.07); }
section[data-testid="stSidebar"] .marca { padding: 4px 0 10px 0; }

/* Menú lateral hecho a mano */
section[data-testid="stSidebar"] .seccion-menu { font-size: .86rem; font-weight: 600; color: #3d3c39; margin: 18px 0 6px 8px; }
section[data-testid="stSidebar"] [data-testid="stPageLink"] a { padding: 5px 8px; border-radius: 8px; }
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 0.5rem; }
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.35rem; }

/* Botones */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    border-radius: 10px; font-weight: 600; padding: 0.5rem 1.1rem;
}

/* Tablas */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* --- componentes propios --- */
.hero {
    background: linear-gradient(135deg, #0d366b 0%, #1c5cab 55%, #2a78d6 100%);
    border-radius: 22px; padding: 44px 44px 40px 44px; color: #ffffff;
    position: relative; overflow: hidden; margin-bottom: 8px;
}
.hero:after {
    content: ""; position: absolute; right: -60px; top: -60px; width: 320px; height: 320px;
    border-radius: 50%; background: radial-gradient(circle, rgba(237,161,0,0.35), rgba(237,161,0,0) 70%);
}
.hero .eyebrow { font-size: 0.78rem; letter-spacing: 0.12em; text-transform: uppercase; opacity: 0.8; font-weight: 600; }
.hero h1 { color: #ffffff !important; font-size: 2.35rem !important; line-height: 1.15; margin: 10px 0 12px 0; max-width: 760px; padding: 0; }
.hero p { font-size: 1.05rem; opacity: 0.9; max-width: 680px; line-height: 1.55; margin: 0; }
.hero .chips { margin-top: 20px; display: flex; gap: 8px; flex-wrap: wrap; }
.hero .chip { background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.22);
    padding: 5px 12px; border-radius: 999px; font-size: 0.8rem; font-weight: 500; }

.encabezado { margin: 4px 0 18px 0; }
.encabezado .eyebrow { font-size: 0.76rem; letter-spacing: 0.1em; text-transform: uppercase; color: #1c5cab; font-weight: 700; }
.encabezado h1 { font-size: 2rem !important; margin: 4px 0 6px 0; padding: 0; }
.encabezado p { color: #52514e; font-size: 1rem; margin: 0; max-width: 820px; line-height: 1.5; }

.paso { background: #ffffff; border: 1px solid rgba(11,11,11,0.08); border-radius: 16px; padding: 20px; height: 100%; }
.paso .n { width: 30px; height: 30px; border-radius: 9px; background: #e8f0fb; color: #1c5cab;
    display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.9rem; margin-bottom: 12px; }
.paso h4 { margin: 0 0 6px 0; font-size: 1rem; font-weight: 650; color: #0b0b0b; padding: 0; }
.paso p { margin: 0; color: #52514e; font-size: 0.9rem; line-height: 1.5; }

.sku-card-titulo { display: flex; align-items: center; gap: 8px; font-weight: 650; color: #0b0b0b; font-size: 0.98rem; }
.punto { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.sku-card-sub { color: #898781; font-size: 0.78rem; margin-top: 2px; }
.sku-card-nums { display: flex; gap: 18px; margin-top: 10px; }
.sku-card-nums .v { font-size: 1.2rem; font-weight: 700; color: #0b0b0b; letter-spacing: -0.01em; }
.sku-card-nums .l { font-size: 0.72rem; color: #898781; }

.nota { background: #fbf6e9; border: 1px solid #f1e2b8; border-radius: 12px; padding: 12px 16px;
    color: #5b4a1a; font-size: 0.9rem; line-height: 1.5; }
.ficha { display: flex; flex-direction: column; gap: 2px; }
.ficha .l { font-size: 0.72rem; color: #898781; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }
.ficha .v { font-size: 0.95rem; color: #0b0b0b; font-weight: 600; }
.pie { color: #898781; font-size: 0.8rem; text-align: center; margin-top: 36px; padding-top: 16px;
    border-top: 1px solid rgba(11,11,11,0.07); }

@media (max-width: 640px) {
    .hero { padding: 28px 22px; }
    .hero h1 { font-size: 1.6rem !important; }
}
</style>
"""


def aplicar_estilo():
    st.markdown(CSS, unsafe_allow_html=True)
    registrar_plantilla()


# ---------------------------------------------------------------- componentes

def encabezado(eyebrow, titulo, descripcion=""):
    st.markdown(
        f'<div class="encabezado"><div class="eyebrow">{eyebrow}</div>'
        f'<h1>{titulo}</h1><p>{descripcion}</p></div>',
        unsafe_allow_html=True,
    )


def nota(texto):
    st.markdown(f'<div class="nota">{texto}</div>', unsafe_allow_html=True)


def pie_pagina():
    st.markdown(
        '<div class="pie">Motor Predictivo de Abastecimiento · Proyecto de título, Ingeniería Industrial UNAB</div>',
        unsafe_allow_html=True,
    )


def grafico(fig, key=None, alto=None):
    if alto:
        fig.update_layout(height=alto)
    st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False, "locale": "es"})


def excel_bytes(hojas):
    """hojas: dict nombre -> DataFrame. Devuelve el .xlsx en memoria."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for nombre, df in hojas.items():
            df.to_excel(writer, sheet_name=nombre[:31], index=False)
            hoja = writer.sheets[nombre[:31]]
            for col in hoja.columns:
                ancho = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                hoja.column_dimensions[col[0].column_letter].width = min(40, max(12, ancho + 2))
    return buffer.getvalue()


# ---------------------------------------------------------------- figuras reutilizables

def fig_banda(fechas, p10, p50, p90, nombre_banda="Rango P10–P90", nombre_p50="Pronóstico (P50)"):
    """Banda de incertidumbre + línea P50. Devuelve la figura para agregarle más trazas."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fechas, y=p90, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=fechas, y=p10, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=AZUL_BANDA,
                             name=nombre_banda, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fechas, y=p50, mode="lines", name=nombre_p50, line=dict(color=AZUL, width=2.2),
                             hovertemplate="%{y:,.0f} u."))
    # P10/P90 en el tooltip sin dibujar línea
    fig.add_trace(go.Scatter(x=fechas, y=p90, name="P90", mode="lines", line=dict(width=0),
                             showlegend=False, hovertemplate="%{y:,.0f} u."))
    fig.add_trace(go.Scatter(x=fechas, y=p10, name="P10", mode="lines", line=dict(width=0),
                             showlegend=False, hovertemplate="%{y:,.0f} u."))
    return fig


def insignia(texto, tipo="neutro"):
    colores = {
        "ok": ("#e7f4ea", "#0f6b1f"), "aviso": ("#fbf1dc", "#7a5300"), "error": ("#fbe6e6", "#9b2020"),
        "neutro": ("#eef0f3", "#3d3c39"), "azul": ("#e8f0fb", "#1c5cab"),
    }
    bg, fg = colores[tipo]
    return (f'<span style="background:{bg};color:{fg};padding:2px 9px;border-radius:999px;'
            f'font-size:.76rem;font-weight:600;white-space:nowrap">{texto}</span>')

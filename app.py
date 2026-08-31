import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from pagina_dataset_propio import render_seccion_dataset_propio

st.set_page_config(page_title="Modelo Predictivo", page_icon="favicon.png", layout="wide")

VARIABLES = ["demanda_unidades", "promocion", "indice_estres_insumos", "precio_clp"]
VENTANA = 30
HORIZONTE = 84
SEMILLA = 42
Z = 1.28
P_REVISION = 30

COLORS = {
    "NotMilk": "#1f77b4", "NotBurger": "#d62728", "NotMayo": "#2ca02c",
    "NotIceCream": "#9467bd", "NotHotDog": "#ff7f0e",
}

DATASET_PATH = "dataset_notco_5sku_5anos.csv"
MODELO_PATH = "modelo_global.keras"


@st.cache_data
def cargar_dataset():
    ds = pd.read_csv(DATASET_PATH)
    ds["fecha"] = pd.to_datetime(ds["fecha"])
    ds = ds.sort_values(["sku", "fecha"]).reset_index(drop=True)
    ds["fecha_dt"] = ds["fecha"]
    return ds


@st.cache_resource
def preparar_scalers(_ds, skus_lista):
    scalers = {}
    for sku in skus_lista:
        mask = _ds["sku"] == sku
        sc = MinMaxScaler()
        sc.fit(_ds.loc[mask, VARIABLES].values.astype("float32"))
        scalers[sku] = sc
    return scalers


def construir_lstm_global(n_skus):
    entrada_serie = layers.Input(shape=(VENTANA, len(VARIABLES)), name="serie")
    entrada_sku = layers.Input(shape=(1,), dtype="int32", name="sku_id")

    x = layers.LSTM(64, return_sequences=True)(entrada_serie)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation="relu")(x)

    p50_por_sku, p90_por_sku, p10_por_sku = [], [], []
    for i in range(n_skus):
        cabeza = layers.Dense(16, activation="relu", name=f"cabeza_{i}")(x)
        p50_i = layers.Dense(1, name=f"p50_{i}")(cabeza)
        delta90_i = layers.Dense(1, activation="softplus", name=f"delta90_{i}")(cabeza)
        delta10_i = layers.Dense(1, activation="softplus", name=f"delta10_{i}")(cabeza)
        p90_i = layers.Add()([p50_i, delta90_i])
        p10_i = layers.Subtract()([p50_i, delta10_i])
        p50_por_sku.append(p50_i)
        p90_por_sku.append(p90_i)
        p10_por_sku.append(p10_i)

    p50_stack = layers.Concatenate(axis=1)(p50_por_sku)
    p90_stack = layers.Concatenate(axis=1)(p90_por_sku)
    p10_stack = layers.Concatenate(axis=1)(p10_por_sku)

    seleccionar = lambda t: tf.gather(t[0], tf.cast(t[1], tf.int32), batch_dims=1)
    salida_p50 = layers.Lambda(seleccionar, output_shape=(1,), name="P50")([p50_stack, entrada_sku])
    salida_p90 = layers.Lambda(seleccionar, output_shape=(1,), name="P90")([p90_stack, entrada_sku])
    salida_p10 = layers.Lambda(seleccionar, output_shape=(1,), name="P10")([p10_stack, entrada_sku])

    return keras.Model(
        inputs={"serie": entrada_serie, "sku_id": entrada_sku},
        outputs={"P10": salida_p10, "P50": salida_p50, "P90": salida_p90},
    )


@st.cache_resource
def cargar_modelo(n_skus):
    np.random.seed(SEMILLA)
    tf.random.set_seed(SEMILLA)
    model = construir_lstm_global(n_skus=n_skus)
    model.load_weights(MODELO_PATH)
    return model


def desescalar_sku(pred_esc, sku, scalers):
    dummy = np.zeros((len(pred_esc), len(VARIABLES)))
    dummy[:, 0] = pred_esc
    return scalers[sku].inverse_transform(dummy)[:, 0]


@st.cache_data
def calcular_validacion(_ds, _model, _scalers, skus_lista, sku_a_id):
    filas = []
    for sku in skus_lista:
        datos_sku = _ds[_ds["sku"] == sku].reset_index(drop=True)
        datos_esc = _scalers[sku].transform(datos_sku[VARIABLES].values.astype("float32"))

        n = len(datos_esc)
        inicio_test = n - HORIZONTE
        X_test, y_test_real, fechas_test = [], [], []
        for i in range(inicio_test, n):
            ventana = datos_esc[i - VENTANA:i, :]
            X_test.append(ventana)
            y_test_real.append(datos_sku["demanda_unidades"].iloc[i])
            fechas_test.append(datos_sku["fecha_dt"].iloc[i])

        X_test = np.array(X_test)
        sku_id_arr = np.full((len(X_test), 1), sku_a_id[sku])
        pred = _model.predict({"serie": X_test, "sku_id": sku_id_arr}, verbose=0)

        p50 = np.maximum(0, desescalar_sku(pred["P50"].flatten(), sku, _scalers))
        p10 = np.maximum(0, desescalar_sku(pred["P10"].flatten(), sku, _scalers))
        p90 = np.maximum(0, desescalar_sku(pred["P90"].flatten(), sku, _scalers))

        for f, real, pp10, pp50, pp90 in zip(fechas_test, y_test_real, p10, p50, p90):
            filas.append({"sku": sku, "fecha": f, "demanda_real": real, "P10": pp10, "P50": pp50, "P90": pp90})

    return pd.DataFrame(filas)


def calcular_meta_periodico(demanda, sigma, lead_time, p, z):
    ss = z * sigma * np.sqrt(lead_time)
    meta = demanda * (lead_time + p) + ss
    return round(meta), round(ss)


def calcular_mape_naive_estacional(ds_sku, dias_atras=7):
    n = len(ds_sku)
    test = ds_sku.iloc[n - HORIZONTE:]
    pred, real = [], []
    for _, fila in test.iterrows():
        fecha_objetivo = fila["fecha"] - pd.Timedelta(days=dias_atras)
        match = ds_sku[ds_sku["fecha"] == fecha_objetivo]
        if len(match) > 0:
            pred.append(match["demanda_unidades"].values[0])
            real.append(fila["demanda_unidades"])
    pred, real = np.array(pred), np.array(real)
    return np.mean(np.abs((real - pred) / real)) * 100


def calcular_rop_diario(demanda, sigma, lead_time, z):
    ss = z * sigma * np.sqrt(lead_time)
    rop = demanda * lead_time + ss
    return round(rop), round(ss)


def predecir_dia(model, ventana, sku_id, scaler, sku):
    entrada = {"serie": ventana.reshape(1, VENTANA, len(VARIABLES)), "sku_id": np.array([[sku_id]])}
    pred = model.predict(entrada, verbose=0)
    p10 = max(0, desescalar_sku(pred["P10"].flatten(), sku, {sku: scaler})[0])
    p50 = max(0, desescalar_sku(pred["P50"].flatten(), sku, {sku: scaler})[0])
    p90 = max(0, desescalar_sku(pred["P90"].flatten(), sku, {sku: scaler})[0])
    return p10, p50, p90


ds = cargar_dataset()
skus_lista = sorted(ds["sku"].unique())
sku_a_id = {s: i for i, s in enumerate(skus_lista)}
scalers_por_sku = preparar_scalers(ds, skus_lista)
model_global = cargar_modelo(len(skus_lista))
validacion = calcular_validacion(ds, model_global, scalers_por_sku, skus_lista, sku_a_id)


vista = st.radio(
    "Vista", ["NotCo", "Sube tu propio dataset"],
    horizontal=True, label_visibility="collapsed",
)
st.markdown("---")

if vista == "NotCo":
    st.sidebar.title("Modelo Predictivo")
    st.sidebar.markdown("Modelo LSTM global multi-SKU (tronco compartido + cabeza independiente por producto)")
    sku_sel = st.sidebar.selectbox("Selecciona un SKU", skus_lista, index=skus_lista.index("NotHotDog") if "NotHotDog" in skus_lista else 0)

    ds_sku = ds[ds["sku"] == sku_sel].sort_values("fecha").reset_index(drop=True)
    scaler_sku = scalers_por_sku[sku_sel]
    id_sku = sku_a_id[sku_sel]
    insumo_nombre = ds_sku["insumo_critico"].iloc[0]
    proc_primaria = ds_sku["procedencia_primaria"].iloc[0]
    proc_secundaria = ds_sku["procedencia_secundaria"].iloc[0]
    precio_secundario_normal = ds_sku["precio_unitario_secundario_clp"].iloc[0]
    precio_insumo_normal = ds_sku["precio_unitario_primario_clp"].iloc[0]

    sub_val = validacion[validacion["sku"] == sku_sel].copy()
    mape_sku = np.mean(np.abs((sub_val["demanda_real"] - sub_val["P50"]) / sub_val["demanda_real"])) * 100
    cobertura_sku = ((sub_val["demanda_real"] >= sub_val["P10"]) & (sub_val["demanda_real"] <= sub_val["P90"])).mean() * 100
    sigma_demanda = sub_val["demanda_real"].std()

    st.sidebar.markdown("---")
    st.sidebar.metric("MAPE (P50)", f"{mape_sku:.2f}%")
    st.sidebar.metric("Cobertura P10–P90", f"{cobertura_sku:.1f}%")
    st.sidebar.markdown(f"**Insumo crítico:** {insumo_nombre}")
    st.sidebar.markdown(f"**Procedencia primaria:** {proc_primaria}")
    st.sidebar.markdown(f"**Procedencia secundaria:** {proc_secundaria}")

    seccion = st.sidebar.radio(
        "Sección",
        ["1. Pronóstico y política ROP/SS", "2. Escenario what-if (estrés de abastecimiento)", "3. Simulación de inventario"],
    )


    st.title(f"{sku_sel}")

    if seccion.startswith("1"):
        st.header("Validación del modelo y política de inventario")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sub_val["fecha"], y=sub_val["demanda_real"], name="Demanda real",
                                  line=dict(color="white", width=2)))
        fig.add_trace(go.Scatter(x=sub_val["fecha"], y=sub_val["P50"], name="Predicción (P50)",
                                  line=dict(color="red", width=2)))
        fig.add_trace(go.Scatter(x=sub_val["fecha"], y=sub_val["P90"], name="P90", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=sub_val["fecha"], y=sub_val["P10"], name="Rango P10–P90", line=dict(width=0),
                                  fill="tonexty", fillcolor="rgba(31,119,180,0.2)"))
        fig.update_layout(title=f"Validación (últimos {HORIZONTE} días) — {sku_sel}",
                           xaxis_title="Fecha", yaxis_title="Unidades/día", height=450)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Demanda promedio de validación", f"{sub_val['P50'].mean():.0f} u./día")
        with col2:
            lead_time_normal = ds_sku["lead_time_dias"].mean()
            st.metric("Lead time promedio", f"{lead_time_normal:.0f} días")

        st.subheader("Política de revisión periódica (P = 30 días)")
        st.latex(r"T = d \times (L + P) + SS \qquad\qquad SS = Z \times \sigma \times \sqrt{L}")

        demanda_actual = sub_val["P50"].mean()
        meta, ss = calcular_meta_periodico(demanda_actual, sigma_demanda, lead_time_normal, P_REVISION, Z)
        rop_diario, _ = calcular_rop_diario(demanda_actual, sigma_demanda, lead_time_normal, Z)

        c1, c2, c3 = st.columns(3)
        c1.metric("Meta (T) del mes", f"{meta:,} u.")
        c2.metric("Stock de seguridad (SS)", f"{ss:,} u.")
        c3.metric("ROP diario (alarma)", f"{rop_diario:,} u.")

        st.subheader("Mejora del LSTM respecto a un enfoque sin modelo predictivo")
        mape_naive = calcular_mape_naive_estacional(ds_sku, dias_atras=7)
        mejora_pct = (mape_naive - mape_sku) / mape_naive * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("MAPE sin modelo", f"{mape_naive:.1f}%")
        c2.metric("MAPE modelo LSTM", f"{mape_sku:.1f}%")
        c3.metric("Mejora", f"{mejora_pct:+.1f}%")

    elif seccion.startswith("2"):
        st.header(f"Escenario what-if: estrés de abastecimiento — {insumo_nombre} ({proc_primaria})")
        st.caption("Simula un evento de estrés (ej. sequía, disrupción logística) en el proveedor primario del insumo crítico de este SKU.")

        with st.form("form_escenario"):
            c1, c2, c3 = st.columns(3)
            fecha_inicio = c1.date_input("Inicio del evento", pd.Timestamp("2025-12-15"))
            fecha_recuperacion = c2.date_input("Inicio de recuperación", pd.Timestamp("2026-02-03"))
            fecha_fin_evento = c3.date_input("Recuperación completa", pd.Timestamp("2026-02-15"))
            c4, c5 = st.columns(2)
            incremento_precio = c4.slider("Sobrecosto del insumo durante el evento (%)", 0, 100, 19) / 100
            elasticidad = c5.slider("Elasticidad precio-demanda", -1.0, 0.0, -0.3, 0.05)
            fecha_fin_pron = st.date_input("Horizonte de pronóstico hasta", pd.Timestamp("2026-04-30"))
            ejecutar = st.form_submit_button("Ejecutar escenario", type="primary")

        if ejecutar:
            fecha_inicio_sequia = pd.Timestamp(fecha_inicio)
            fecha_inicio_lluvia = pd.Timestamp(fecha_recuperacion)
            fecha_fin_cosecha = pd.Timestamp(fecha_fin_evento)
            fecha_fin_pronostico = pd.Timestamp(fecha_fin_pron)

            np.random.seed(SEMILLA)
            umbral_estres = ds_sku["indice_estres_insumos"].quantile(0.85)
            estres_normal_vals = ds_sku["indice_estres_insumos"].values
            estres_altos_vals = ds_sku[ds_sku["indice_estres_insumos"] >= umbral_estres]["indice_estres_insumos"].values
            lead_time_normal = ds_sku["lead_time_dias"].mean()
            lead_time_sequia = ds_sku[ds_sku["indice_estres_insumos"] >= umbral_estres]["lead_time_dias"].mean()

            precio_normal = ds_sku[ds_sku["fecha_dt"] < fecha_inicio_sequia]["precio_clp"].iloc[-1]
            precio_sequia = precio_normal * (1 + incremento_precio)
            precio_primario_sequia = precio_insumo_normal * (1 + incremento_precio)

            ds_sku_escenario = ds_sku.copy()
            ds_sku_escenario["precio_clp"] = ds_sku_escenario["precio_clp"].astype("float64")
            mask_dic = ds_sku_escenario["fecha_dt"] >= fecha_inicio_sequia
            n_dias_evento_hist = mask_dic.sum()
            if n_dias_evento_hist > 0:
                ds_sku_escenario.loc[mask_dic, "indice_estres_insumos"] = np.random.choice(estres_altos_vals, n_dias_evento_hist)
                ds_sku_escenario.loc[mask_dic, "precio_clp"] = precio_sequia

            futuro = pd.DataFrame({"fecha_dt": pd.date_range(ds_sku["fecha_dt"].max() + pd.Timedelta(days=1), fecha_fin_pronostico, freq="D")})
            futuro["promocion"] = 0

            def estres_precio_dia(fecha):
                if fecha < fecha_inicio_lluvia:
                    return np.random.choice(estres_altos_vals), precio_sequia
                elif fecha < fecha_fin_cosecha:
                    avance = (fecha - fecha_inicio_lluvia).days / max(1, (fecha_fin_cosecha - fecha_inicio_lluvia).days)
                    estres = np.random.choice(estres_altos_vals) * (1 - avance) + np.random.choice(estres_normal_vals) * avance
                    precio = precio_sequia + (precio_normal - precio_sequia) * avance
                    return estres, precio
                else:
                    return np.random.choice(estres_normal_vals), precio_normal

            futuro[["indice_estres_insumos", "precio_clp"]] = futuro["fecha_dt"].apply(lambda f: pd.Series(estres_precio_dia(f)))

            historial_esc = scaler_sku.transform(ds_sku_escenario[VARIABLES].values.astype("float32"))
            ventana = historial_esc[-VENTANA:].copy()

            progreso = st.progress(0.0, text="Generando pronóstico recursivo día a día...")
            pred_futuro = {"fecha": [], "P10": [], "P50": [], "P90": []}
            for idx, (_, fila) in enumerate(futuro.iterrows()):
                entrada = {"serie": ventana.reshape(1, VENTANA, len(VARIABLES)), "sku_id": np.array([[id_sku]])}
                pred = model_global.predict(entrada, verbose=0)
                p10_b = max(0, desescalar_sku(pred["P10"].flatten(), sku_sel, scalers_por_sku)[0])
                p50_b = max(0, desescalar_sku(pred["P50"].flatten(), sku_sel, scalers_por_sku)[0])
                p90_b = max(0, desescalar_sku(pred["P90"].flatten(), sku_sel, scalers_por_sku)[0])

                factor = 1 + elasticidad * ((fila["precio_clp"] / precio_normal) - 1)
                pred_futuro["fecha"].append(fila["fecha_dt"])
                pred_futuro["P10"].append(p10_b * factor)
                pred_futuro["P50"].append(p50_b * factor)
                pred_futuro["P90"].append(p90_b * factor)

                nueva_fila = np.array([[p50_b, fila["promocion"], fila["indice_estres_insumos"], fila["precio_clp"]]], dtype="float32")
                ventana = np.vstack([ventana[1:], scaler_sku.transform(nueva_fila)])
                progreso.progress((idx + 1) / len(futuro))
            progreso.empty()

            pronostico_futuro = pd.DataFrame(pred_futuro)
            st.session_state["pronostico_futuro"] = pronostico_futuro
            st.session_state["params_escenario"] = dict(
                fecha_inicio_sequia=fecha_inicio_sequia, fecha_inicio_lluvia=fecha_inicio_lluvia,
                fecha_fin_cosecha=fecha_fin_cosecha, lead_time_normal=lead_time_normal, lead_time_sequia=lead_time_sequia,
                precio_primario_sequia=precio_primario_sequia,
            )

        if "pronostico_futuro" in st.session_state:
            pronostico_futuro = st.session_state["pronostico_futuro"]
            p = st.session_state["params_escenario"]

            hist_reciente = ds_sku[ds_sku["fecha_dt"] >= ds_sku["fecha_dt"].max() - pd.Timedelta(days=90)]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist_reciente["fecha_dt"], y=hist_reciente["demanda_unidades"],
                                      name="Historial reciente", line=dict(color="gray", width=1.5)))
            fig.add_trace(go.Scatter(x=pronostico_futuro["fecha"], y=pronostico_futuro["P90"], line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=pronostico_futuro["fecha"], y=pronostico_futuro["P10"], line=dict(width=0),
                                      fill="tonexty", fillcolor="rgba(31,119,180,0.2)", name="Rango P10–P90"))
            fig.add_trace(go.Scatter(x=pronostico_futuro["fecha"], y=pronostico_futuro["P50"], name="Predicción (P50)",
                                      line=dict(color="red", width=2)))
            fig.add_vrect(x0=p["fecha_inicio_lluvia"], x1=p["fecha_fin_cosecha"], fillcolor="orange", opacity=0.15,
                          annotation_text="Recuperación", line_width=0)
            fig.update_layout(title="Pronóstico bajo el escenario simulado", xaxis_title="Fecha",
                               yaxis_title="Unidades/día", height=450)
            st.plotly_chart(fig, use_container_width=True)

            demanda_evento = pronostico_futuro[pronostico_futuro.fecha < p["fecha_inicio_lluvia"]]["P50"].mean()
            demanda_recuperada = pronostico_futuro[pronostico_futuro.fecha >= p["fecha_fin_cosecha"]]["P50"].mean()
            c1, c2 = st.columns(2)
            c1.metric("Demanda proyectada durante el evento", f"{demanda_evento:.0f} u./día")
            c2.metric("Demanda proyectada ya recuperado", f"{demanda_recuperada:.0f} u./día")

            st.subheader(f"Costo de abastecimiento: {proc_primaria} (afectado) vs. {proc_secundaria} (alternativo)")
            meses_resumen = pronostico_futuro.copy()
            meses_resumen["mes"] = meses_resumen["fecha"].dt.to_period("M")
            tabla_costo = []
            for periodo, grupo in meses_resumen.groupby("mes"):
                d_mes = grupo["P50"].mean()
                dias = len(grupo)
                precio_evento = p["precio_primario_sequia"] if periodo.to_timestamp() < p["fecha_inicio_lluvia"] else precio_insumo_normal
                costo_primario = d_mes * dias * precio_evento
                costo_secundario = d_mes * dias * precio_secundario_normal
                tabla_costo.append({"Mes": periodo.strftime("%B %Y"), f"Costo {proc_primaria}": f"${costo_primario:,.0f}",
                                     f"Costo {proc_secundaria}": f"${costo_secundario:,.0f}",
                                     "Ahorro cambiando de proveedor": f"${costo_primario - costo_secundario:,.0f}"})
            st.dataframe(pd.DataFrame(tabla_costo), use_container_width=True, hide_index=True)

    elif seccion.startswith("3"):
        st.header("Simulación de la política de inventario")
        st.caption("Sistema híbrido: revisión periódica mensual (nivel meta) + monitoreo diario con gatillo de compra de emergencia.")

        if "pronostico_futuro" not in st.session_state:
            st.warning("Primero ejecuta un escenario en la sección 2 para poder comparar 'normal' vs. 'evento simulado' en esta simulación.")
        else:
            pronostico_evento = st.session_state["pronostico_futuro"]
            p = st.session_state["params_escenario"]
            lead_time_normal = p["lead_time_normal"]
            lead_time_evento = p["lead_time_sequia"]
            precio_evento = p["precio_primario_sequia"]

            inventario_inicial = ds_sku["inventario_unidades"].iloc[-1]
            st.metric("Inventario inicial (último dato real del dataset)", f"{inventario_inicial:,.0f} u.")

            meses = sorted(pronostico_evento["fecha"].dt.to_period("M").unique())

            def simular():
                inventario = inventario_inicial
                filas = []
                emergencias = 0
                for periodo in meses:
                    grupo = pronostico_evento[pronostico_evento["fecha"].dt.to_period("M") == periodo]
                    demanda_mes = grupo["P50"].mean()
                    dias_mes = len(grupo)
                    lt_mes = lead_time_evento if periodo.to_timestamp() < p["fecha_inicio_lluvia"] else lead_time_normal

                    meta_mes, ss_mes = calcular_meta_periodico(demanda_mes, sigma_demanda, lt_mes, P_REVISION, Z)
                    rop_diario, _ = calcular_rop_diario(demanda_mes, sigma_demanda, lt_mes, Z)

                    remanente_inicio = inventario
                    pedido = max(0, meta_mes - remanente_inicio)
                    inventario = remanente_inicio + pedido

                    emergencia = False
                    monto_emergencia = 0
                    consumo_total = 0
                    for _ in range(dias_mes):
                        inventario -= demanda_mes
                        consumo_total += demanda_mes
                        if inventario < rop_diario and not emergencia:
                            monto_emergencia = meta_mes - inventario
                            inventario += monto_emergencia
                            emergencia = True
                            emergencias += 1
                    inventario = max(0, inventario)

                    filas.append({"Mes": periodo.strftime("%B %Y"), "Remanente inicio": round(remanente_inicio),
                                   "Meta (T)": meta_mes, "ROP diario (alarma)": rop_diario, "Pedido mensual": round(pedido),
                                   "Consumo del mes": round(consumo_total), "Emergencia": "Sí" if emergencia else "No",
                                   "Monto emergencia": round(monto_emergencia), "Inventario fin de mes": round(inventario)})
                return pd.DataFrame(filas), emergencias

            tabla_sim, n_emergencias = simular()

            st.subheader("Simulación mes a mes bajo el escenario ejecutado en la sección 2")
            st.dataframe(tabla_sim, use_container_width=True, hide_index=True)
            st.metric("Compras de emergencia disparadas", n_emergencias)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tabla_sim["Mes"], y=tabla_sim["Inventario fin de mes"], name="Inventario fin de mes",
                                      mode="lines+markers", line=dict(color="steelblue", width=2)))
            fig.add_trace(go.Scatter(x=tabla_sim["Mes"], y=tabla_sim["ROP diario (alarma)"], name="Umbral de alarma (ROP diario)",
                                      mode="lines", line=dict(color="firebrick", dash="dash")))
            fig.add_trace(go.Scatter(x=tabla_sim["Mes"], y=tabla_sim["Meta (T)"], name="Meta (T)",
                                      mode="lines", line=dict(color="green", dash="dot")))
            fig.update_layout(title="Evolución del inventario vs. umbrales de la política", xaxis_title="Mes",
                               yaxis_title="Unidades", height=420)
            st.plotly_chart(fig, use_container_width=True)


else:
    render_seccion_dataset_propio()

st.markdown("---")
st.caption("Proyecto de título UNAB — Ingeniería Industrial. Modelo LSTM global multi-SKU, entrenado con datos simulados anclados a fuentes reales de mercado.")

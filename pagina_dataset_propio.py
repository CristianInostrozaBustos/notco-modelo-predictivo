import os
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.saving import register_keras_serializable
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass, field

CARPETA_MODELOS_CACHE = "modelos_subidos_cache"
LIMITE_FILAS_DATASET = 50_000
LIMITE_EPOCAS = 100
PACIENCIA_EARLY_STOPPING = 10


@dataclass
class EsquemaDetectado:
    columna_fecha: "str | None"
    columna_entidad: "str | None"
    candidatas_objetivo: list = field(default_factory=list)
    candidatas_exogenas: list = field(default_factory=list)
    confianza_fecha: float = 0.0
    confianza_entidad: float = 0.0
    advertencias: list = field(default_factory=list)


def _score_columna_fecha(serie):
    if pd.api.types.is_numeric_dtype(serie):
        return 0.0
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(serie, errors="coerce")
        return parsed.notna().mean()
    except Exception:
        return 0.0


def _score_columna_entidad(serie, n_filas):
    n_unicos = serie.nunique(dropna=True)
    if n_unicos < 3 or n_unicos > 500:
        return 0.0
    repeticion_promedio = n_filas / n_unicos
    if repeticion_promedio < 5:
        return 0.0
    score = min(1.0, repeticion_promedio / 100) * (1 - min(1.0, n_unicos / 200))
    if pd.api.types.is_numeric_dtype(serie):
        score *= 0.3
    conteos = serie.value_counts()
    uniformidad = 1 - (conteos.std() / conteos.mean()) if conteos.mean() > 0 else 0
    uniformidad = max(0.0, min(1.0, uniformidad))
    score *= (0.5 + 0.5 * uniformidad)
    return score


def _es_numerica_continua(serie):
    return pd.api.types.is_numeric_dtype(serie) and serie.nunique() > 10


def detectar_esquema(df):
    n_filas = len(df)
    advertencias = []
    scores_fecha = {col: _score_columna_fecha(df[col]) for col in df.columns}
    col_fecha = max(scores_fecha, key=scores_fecha.get) if scores_fecha else None
    conf_fecha = scores_fecha.get(col_fecha, 0.0) if col_fecha else 0.0
    if conf_fecha < 0.8:
        advertencias.append(f"Columna de fecha detectada ('{col_fecha}') con baja confianza ({conf_fecha:.0%}).")
        if conf_fecha == 0.0:
            col_fecha = None

    candidatas_entidad = [c for c in df.columns if c != col_fecha]
    scores_entidad = {col: _score_columna_entidad(df[col], n_filas) for col in candidatas_entidad}
    col_entidad = None
    if scores_entidad:
        mejor = max(scores_entidad, key=scores_entidad.get)
        if scores_entidad[mejor] > 0.1:
            col_entidad = mejor
        else:
            advertencias.append("No se detectó una columna de entidad clara. Se asumirá una sola serie temporal.")

    excluidas = {col_fecha, col_entidad}
    candidatas_objetivo = [c for c in df.columns if c not in excluidas and _es_numerica_continua(df[c])]
    candidatas_exogenas = [
        c for c in df.columns
        if c not in excluidas and pd.api.types.is_numeric_dtype(df[c])
    ]

    return EsquemaDetectado(
        columna_fecha=col_fecha, columna_entidad=col_entidad,
        candidatas_objetivo=candidatas_objetivo, candidatas_exogenas=candidatas_exogenas,
        confianza_fecha=conf_fecha,
        confianza_entidad=scores_entidad.get(col_entidad, 0.0) if col_entidad else 0.0,
        advertencias=advertencias,
    )


@dataclass
class ConfiguracionColumnas:
    columna_fecha: str
    columna_entidad: "str | None"
    columna_objetivo: str
    columnas_exogenas: list


@dataclass
class DatosPreparados:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    entidad_ids_train: np.ndarray
    entidad_ids_test: np.ndarray
    entidad_a_id: dict
    scalers_por_entidad: dict
    variables: list
    ventana: int
    horizonte: int


def preparar_datos(df, config, ventana=30, horizonte=84):
    ds = df.copy()
    ds[config.columna_fecha] = pd.to_datetime(ds[config.columna_fecha])
    variables = [config.columna_objetivo] + list(config.columnas_exogenas)

    no_numericas = [v for v in variables if not pd.api.types.is_numeric_dtype(ds[v])]
    if no_numericas:
        raise ValueError(
            f"Las siguientes columnas no son numéricas y no se pueden usar como objetivo "
            f"ni como variable exógena todavía: {', '.join(no_numericas)}. "
            "El motor por ahora solo soporta variables numéricas (categóricas como texto "
            "requerirían codificación adicional, no implementada)."
        )

    if config.columna_entidad is None:
        ds["_entidad_generica"] = "serie_unica"
        col_entidad = "_entidad_generica"
    else:
        col_entidad = config.columna_entidad

    ds = ds.sort_values([col_entidad, config.columna_fecha]).reset_index(drop=True)
    ds[variables] = ds[variables].astype("float64")

    entidades = sorted(ds[col_entidad].unique())
    entidad_a_id = {ent: i for i, ent in enumerate(entidades)}

    if len(ds) < ventana + horizonte:
        raise ValueError(f"El dataset tiene {len(ds)} filas totales, pero se necesitan al menos {ventana + horizonte} por entidad.")
    for ent in entidades:
        n_filas_ent = (ds[col_entidad] == ent).sum()
        if n_filas_ent < ventana + horizonte:
            raise ValueError(f"La entidad '{ent}' tiene solo {n_filas_ent} registros (se necesitan {ventana + horizonte}).")

    scalers_por_entidad = {}
    ds_escalado = ds.copy()
    for ent in entidades:
        mask = ds[col_entidad] == ent
        scaler = MinMaxScaler()
        ds_escalado.loc[mask, variables] = scaler.fit_transform(ds.loc[mask, variables].values.astype("float32"))
        scalers_por_entidad[ent] = scaler

    X, y, entidad_ids = [], [], []
    for ent in entidades:
        datos_ent = ds_escalado[ds_escalado[col_entidad] == ent][variables].values.astype("float32")
        id_ent = entidad_a_id[ent]
        for i in range(ventana, len(datos_ent)):
            X.append(datos_ent[i - ventana:i, :])
            y.append(datos_ent[i, 0])
            entidad_ids.append(id_ent)

    X, y, entidad_ids = np.array(X), np.array(y), np.array(entidad_ids)

    train_mask = np.ones(len(X), dtype=bool)
    for ent in entidades:
        id_ent = entidad_a_id[ent]
        idx_ent = np.where(entidad_ids == id_ent)[0]
        train_mask[idx_ent[-horizonte:]] = False

    return DatosPreparados(
        X_train=X[train_mask], X_test=X[~train_mask], y_train=y[train_mask], y_test=y[~train_mask],
        entidad_ids_train=entidad_ids[train_mask], entidad_ids_test=entidad_ids[~train_mask],
        entidad_a_id=entidad_a_id, scalers_por_entidad=scalers_por_entidad,
        variables=variables, ventana=ventana, horizonte=horizonte,
    )


@register_keras_serializable(package="motor_generico")
class SeleccionarPorEntidad(layers.Layer):
    def call(self, inputs):
        valores, entidad_id = inputs
        return tf.gather(valores, tf.cast(entidad_id, tf.int32), batch_dims=1)


def construir_modelo_generico(ventana, n_variables, n_entidades,
                               unidades_lstm_1=64, unidades_lstm_2=32,
                               unidades_dense_tronco=16, unidades_dense_cabeza=16, dropout=0.2):
    entrada_serie = layers.Input(shape=(ventana, n_variables), name="serie")
    entrada_entidad = layers.Input(shape=(1,), dtype="int32", name="entidad_id")

    x = layers.LSTM(unidades_lstm_1, return_sequences=True)(entrada_serie)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(unidades_lstm_2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(unidades_dense_tronco, activation="relu")(x)

    p50_por_entidad, p90_por_entidad, p10_por_entidad = [], [], []
    for i in range(n_entidades):
        cabeza = layers.Dense(unidades_dense_cabeza, activation="relu", name=f"cabeza_{i}")(x)
        p50_i = layers.Dense(1, name=f"p50_{i}")(cabeza)
        delta90_i = layers.Dense(1, activation="softplus", name=f"delta90_{i}")(cabeza)
        delta10_i = layers.Dense(1, activation="softplus", name=f"delta10_{i}")(cabeza)
        p90_i = layers.Add()([p50_i, delta90_i])
        p10_i = layers.Subtract()([p50_i, delta10_i])
        p50_por_entidad.append(p50_i)
        p90_por_entidad.append(p90_i)
        p10_por_entidad.append(p10_i)

    p50_stack = layers.Concatenate(axis=1)(p50_por_entidad)
    p90_stack = layers.Concatenate(axis=1)(p90_por_entidad)
    p10_stack = layers.Concatenate(axis=1)(p10_por_entidad)

    salida_p50 = SeleccionarPorEntidad(name="P50")([p50_stack, entrada_entidad])
    salida_p90 = SeleccionarPorEntidad(name="P90")([p90_stack, entrada_entidad])
    salida_p10 = SeleccionarPorEntidad(name="P10")([p10_stack, entrada_entidad])

    return keras.Model(
        inputs=[entrada_serie, entrada_entidad],
        outputs={"P50": salida_p50, "P90": salida_p90, "P10": salida_p10},
    )


def _pinball(q):
    def loss(y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))
    return loss


def estimar_viabilidad(n_filas_dataset, n_entidades):
    if n_filas_dataset > LIMITE_FILAS_DATASET:
        return False, f"El dataset tiene {n_filas_dataset:,} filas, por sobre el límite de {LIMITE_FILAS_DATASET:,} para entrenar en este sitio."
    if n_entidades > 30:
        return False, f"El dataset tiene {n_entidades} entidades, por sobre el máximo recomendado (30) para este sitio."
    return True, "OK"


def entrenar_modelo_generico(modelo, datos, epocas_solicitadas=100, batch_size=32):
    epocas = min(epocas_solicitadas, LIMITE_EPOCAS)
    modelo.compile(
        optimizer="adam",
        loss={
            "P50": lambda yt, yp: tf.reduce_mean(tf.abs(yt - yp)),
            "P90": _pinball(0.9),
            "P10": _pinball(0.1),
        },
    )
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="loss", patience=PACIENCIA_EARLY_STOPPING, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=5, min_lr=1e-5),
    ]
    historial = modelo.fit(
        [datos.X_train, datos.entidad_ids_train.reshape(-1, 1)],
        {"P50": datos.y_train, "P90": datos.y_train, "P10": datos.y_train},
        epochs=epocas, batch_size=batch_size, verbose=0, callbacks=callbacks,
    )
    return historial


def evaluar_modelo(modelo, datos):
    pred = modelo.predict([datos.X_test, datos.entidad_ids_test.reshape(-1, 1)], verbose=0)

    def desescalar(pred_esc, entidad):
        scaler = datos.scalers_por_entidad[entidad]
        dummy = np.zeros((len(pred_esc), len(datos.variables)))
        dummy[:, 0] = pred_esc.flatten()
        return scaler.inverse_transform(dummy)[:, 0]

    metricas = {}
    predicciones_por_entidad = {}
    id_a_entidad = {v: k for k, v in datos.entidad_a_id.items()}
    for ent_id, entidad in id_a_entidad.items():
        mask = datos.entidad_ids_test == ent_id
        real = desescalar(datos.y_test[mask], entidad)
        p50 = np.maximum(0, desescalar(pred["P50"][mask], entidad))
        p10 = np.maximum(0, desescalar(pred["P10"][mask], entidad))
        p90 = np.maximum(0, desescalar(pred["P90"][mask], entidad))
        mape = np.mean(np.abs((real - p50) / real)) * 100
        cobertura = ((real >= p10) & (real <= p90)).mean() * 100
        metricas[entidad] = {"MAPE": round(mape, 2), "Cobertura_P10_P90": round(cobertura, 1)}
        predicciones_por_entidad[entidad] = {"real": real, "P50": p50, "P10": p10, "P90": p90}

    return metricas, predicciones_por_entidad


VERSION_ARQUITECTURA = "v2-capa-registrada"


def calcular_huella_dataset(df, config, ventana, horizonte):
    resumen_filas = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    config_str = f"{config.columna_fecha}|{config.columna_entidad}|{config.columna_objetivo}|{sorted(config.columnas_exogenas)}|{ventana}|{horizonte}|{VERSION_ARQUITECTURA}"
    hasher = hashlib.sha256()
    hasher.update(resumen_filas)
    hasher.update(config_str.encode())
    return hasher.hexdigest()[:16]


def entrenar_o_cargar_modelo(df, config, ventana, horizonte, epocas_solicitadas=100):
    os.makedirs(CARPETA_MODELOS_CACHE, exist_ok=True)
    huella = calcular_huella_dataset(df, config, ventana, horizonte)
    ruta_modelo = os.path.join(CARPETA_MODELOS_CACHE, f"modelo_{huella}.keras")

    datos = preparar_datos(df, config, ventana=ventana, horizonte=horizonte)

    if os.path.exists(ruta_modelo):
        try:
            modelo = keras.models.load_model(ruta_modelo, compile=False, safe_mode=False)
            entrada_prueba = datos.X_test[:1]
            entidad_prueba = datos.entidad_ids_test[:1].reshape(-1, 1)
            modelo.predict([entrada_prueba, entidad_prueba], verbose=0)
            return modelo, datos, huella, False
        except Exception:
            os.remove(ruta_modelo)

    modelo = construir_modelo_generico(
        ventana=datos.ventana, n_variables=len(datos.variables), n_entidades=len(datos.entidad_a_id),
    )
    entrenar_modelo_generico(modelo, datos, epocas_solicitadas=epocas_solicitadas)
    modelo.save(ruta_modelo)
    return modelo, datos, huella, True


@dataclass
class EventoWhatIf:
    variable_afectada: str
    fecha_inicio: pd.Timestamp
    fecha_fin: pd.Timestamp
    cambio_pct: float
    entidad: str


def construir_trayectoria_escenario(df_historico, config, evento, dias_horizonte, fecha_inicio_pronostico):
    ds = df_historico.copy()
    ds[config.columna_fecha] = pd.to_datetime(ds[config.columna_fecha])
    col_entidad = config.columna_entidad or "_entidad_generica"
    if col_entidad == "_entidad_generica" and col_entidad not in ds.columns:
        ds[col_entidad] = "serie_unica"
    sub = ds[ds[col_entidad] == evento.entidad].sort_values(config.columna_fecha)

    fechas_futuras = pd.date_range(fecha_inicio_pronostico, periods=dias_horizonte, freq="D")
    trayectoria = pd.DataFrame({config.columna_fecha: fechas_futuras})

    for var in config.columnas_exogenas:
        valores = []
        for f in fechas_futuras:
            fecha_analoga = f - pd.DateOffset(years=1)
            match = sub[sub[config.columna_fecha] == fecha_analoga]
            valores.append(match[var].values[0] if len(match) > 0 else sub[var].mean())
        trayectoria[var] = np.array(valores, dtype="float64")

    if evento.variable_afectada in trayectoria.columns:
        mask_evento = (trayectoria[config.columna_fecha] >= evento.fecha_inicio) & (
            trayectoria[config.columna_fecha] <= evento.fecha_fin
        )
        trayectoria.loc[mask_evento, evento.variable_afectada] *= (1 + evento.cambio_pct)

    return trayectoria


def pronostico_recursivo(modelo, datos, df_historico, config, entidad, trayectoria_futura):
    scaler = datos.scalers_por_entidad[entidad]
    ent_id = datos.entidad_a_id[entidad]
    variables = datos.variables

    ds = df_historico.copy()
    ds[config.columna_fecha] = pd.to_datetime(ds[config.columna_fecha])
    col_entidad = config.columna_entidad or "_entidad_generica"
    if col_entidad == "_entidad_generica" and col_entidad not in ds.columns:
        ds[col_entidad] = "serie_unica"
    sub = ds[ds[col_entidad] == entidad].sort_values(config.columna_fecha)

    ultimos = sub[variables].values[-datos.ventana:].astype("float32")
    secuencia = scaler.transform(ultimos)

    resultados = []
    for _, fila in trayectoria_futura.iterrows():
        entrada = secuencia.reshape(1, datos.ventana, len(variables))
        entrada_ent = np.array([[ent_id]])
        pred = modelo.predict([entrada, entrada_ent], verbose=0)
        p50_esc, p90_esc, p10_esc = pred["P50"][0, 0], pred["P90"][0, 0], pred["P10"][0, 0]

        dummy = np.zeros((1, len(variables)))
        dummy[0, 0] = p50_esc
        p50 = max(0, scaler.inverse_transform(dummy)[0, 0])
        dummy[0, 0] = p90_esc
        p90 = max(0, scaler.inverse_transform(dummy)[0, 0])
        dummy[0, 0] = p10_esc
        p10 = max(0, scaler.inverse_transform(dummy)[0, 0])

        resultados.append({config.columna_fecha: fila[config.columna_fecha], "P10": p10, "P50": p50, "P90": p90})

        fila_siguiente = np.zeros((1, len(variables)))
        fila_siguiente[0, 0] = p50_esc
        if config.columnas_exogenas:
            exogenas_sin_escalar = fila[config.columnas_exogenas].values.astype("float32")
            dummy_exog = np.zeros((1, len(variables)))
            dummy_exog[0, 1:] = exogenas_sin_escalar
            fila_siguiente[0, 1:] = scaler.transform(dummy_exog)[0, 1:]

        secuencia = np.vstack([secuencia[1:], fila_siguiente])

    return pd.DataFrame(resultados)


def resumen_roles_columnas(df, esquema):
    lineas = []
    lineas.append(f"- **Fecha**: {esquema.columna_fecha if esquema.columna_fecha else 'No detectada'}")
    lineas.append(
        f"- **Entidad (SKU/tienda/producto)**: "
        f"{esquema.columna_entidad if esquema.columna_entidad else 'No detectada — se tratará como una sola serie'}"
    )
    if esquema.candidatas_objetivo:
        lineas.append(f"- **Variable a pronosticar (candidatas)**: {', '.join(esquema.candidatas_objetivo)}")
    else:
        lineas.append("- **Variable a pronosticar**: No detectada — no hay ninguna columna numérica que se pueda usar como objetivo")
    if esquema.candidatas_exogenas:
        lineas.append(f"- **Variables exógenas (factores que podrían influir)**: {', '.join(esquema.candidatas_exogenas)}")
    else:
        lineas.append("- **Variables exógenas**: No detectadas")
    return "\n".join(lineas)


def resumen_temporal_dataset(df, columna_fecha, columna_entidad):
    fechas = pd.to_datetime(df[columna_fecha])
    fecha_min, fecha_max = fechas.min(), fechas.max()
    dias_totales = (fecha_max - fecha_min).days
    anios_aprox = dias_totales / 365.25
    n_entidades = df[columna_entidad].nunique() if columna_entidad else 1
    return {
        "fecha_min": fecha_min, "fecha_max": fecha_max,
        "dias_totales": dias_totales, "anios_aprox": anios_aprox,
        "n_entidades": n_entidades,
    }


def graficar_serie_mensual(df, columna_fecha, columna_entidad, columna_objetivo):
    ds = df.copy()
    ds[columna_fecha] = pd.to_datetime(ds[columna_fecha])
    col_ent = columna_entidad
    if col_ent is None:
        ds["_entidad_generica"] = "serie_unica"
        col_ent = "_entidad_generica"
    ds["_mes"] = ds[columna_fecha].dt.to_period("M").dt.to_timestamp()
    resumen = ds.groupby([col_ent, "_mes"])[columna_objetivo].mean().reset_index()

    fig = go.Figure()
    for ent in sorted(resumen[col_ent].astype(str).unique()):
        sub = resumen[resumen[col_ent].astype(str) == ent]
        fig.add_trace(go.Scatter(x=sub["_mes"], y=sub[columna_objetivo], name=str(ent), mode="lines+markers"))
    fig.update_layout(
        title=f"{columna_objetivo} — promedio mensual por entidad",
        xaxis_title="Mes", yaxis_title=columna_objetivo, height=420,
    )
    return fig


def graficar_serie_diaria(df, columna_fecha, columna_entidad, columna_objetivo):
    ds = df.copy()
    ds[columna_fecha] = pd.to_datetime(ds[columna_fecha])
    col_ent = columna_entidad
    if col_ent is None:
        ds["_entidad_generica"] = "serie_unica"
        col_ent = "_entidad_generica"
    ds = ds.sort_values([col_ent, columna_fecha])

    fig = go.Figure()
    for ent in sorted(ds[col_ent].astype(str).unique()):
        sub = ds[ds[col_ent].astype(str) == ent]
        fig.add_trace(go.Scatter(x=sub[columna_fecha], y=sub[columna_objetivo], name=str(ent), mode="lines"))
    fig.update_layout(
        title=f"{columna_objetivo} — demanda diaria por entidad",
        xaxis_title="Fecha", yaxis_title=columna_objetivo, height=380,
    )
    return fig


Z_POR_NIVEL_SERVICIO = {
    "80%": 0.84, "85%": 1.04, "90%": 1.28, "95%": 1.65, "99%": 2.33,
}


def calcular_politica_inventario(predicciones_entidad, lead_time_dias, nivel_servicio, periodo_revision_dias):
    d_prom = float(np.mean(predicciones_entidad["P50"]))
    z_90 = Z_POR_NIVEL_SERVICIO["90%"]
    sigma = max(0.0, float(np.mean(predicciones_entidad["P90"] - predicciones_entidad["P50"])) / z_90)
    z = Z_POR_NIVEL_SERVICIO[nivel_servicio]

    ss = z * sigma * np.sqrt(lead_time_dias)
    rop = d_prom * lead_time_dias + ss
    meta_t = d_prom * (lead_time_dias + periodo_revision_dias) + ss

    return {
        "demanda_promedio": round(d_prom, 1),
        "sigma_demanda": round(sigma, 1),
        "Z": z,
        "ROP": round(rop),
        "SS": round(ss),
        "Meta_T": round(meta_t),
    }


def render_seccion_dataset_propio():
    st.header("Sube tu propio dataset")
    st.markdown(
        "Sube cualquier archivo CSV con datos de series temporales "
        "(ventas, demanda, tráfico, etc.) y el sitio detecta automáticamente "
        "la estructura, entrena un modelo de pronóstico por cuantiles y "
        "muestra los resultados."
    )

    archivo = st.file_uploader("Archivo CSV", type=["csv"])
    if archivo is None:
        st.info("Sube un archivo para comenzar.")
        return

    df = pd.read_csv(archivo)
    st.write(f"Dataset cargado: {df.shape[0]:,} filas, {df.shape[1]} columnas.")
    st.dataframe(df.head(5))

    esquema = detectar_esquema(df)
    for adv in esquema.advertencias:
        st.warning(adv)

    st.markdown(resumen_roles_columnas(df, esquema))

    st.subheader("Confirma las columnas detectadas")
    col1, col2 = st.columns(2)
    with col1:
        columna_fecha = st.selectbox(
            "Columna de fecha", options=list(df.columns),
            index=list(df.columns).index(esquema.columna_fecha) if esquema.columna_fecha in df.columns else 0,
        )
        opciones_entidad = ["(ninguna — una sola serie)"] + list(df.columns)
        idx_entidad = opciones_entidad.index(esquema.columna_entidad) if esquema.columna_entidad in opciones_entidad else 0
        columna_entidad_sel = st.selectbox("Columna de entidad (SKU/tienda/producto)", options=opciones_entidad, index=idx_entidad)
        columna_entidad = None if columna_entidad_sel.startswith("(ninguna") else columna_entidad_sel
    with col2:
        candidatas_obj = esquema.candidatas_objetivo or list(df.columns)
        columna_objetivo = st.selectbox("Variable a pronosticar (objetivo)", options=candidatas_obj)
        candidatas_exog = [c for c in esquema.candidatas_exogenas if c != columna_objetivo]
        columnas_exogenas = st.multiselect("Variables exógenas (opcional)", options=candidatas_exog, default=candidatas_exog[:3])

    col3, col4 = st.columns(2)
    with col3:
        ventana = st.number_input("Ventana (días de historial por secuencia)", min_value=7, max_value=180, value=30)
    with col4:
        horizonte = st.number_input("Horizonte (días a pronosticar / dejar para test)", min_value=7, max_value=180, value=30)

    if columna_entidad:
        filas_por_entidad_min = df.groupby(columna_entidad).size().min()
    else:
        filas_por_entidad_min = len(df)
    if horizonte > filas_por_entidad_min * 0.35:
        st.warning(
            f"El horizonte que elegiste ({horizonte} días) es una porción grande del historial "
            f"disponible por entidad (la entidad con menos datos tiene {filas_por_entidad_min} filas). "
            "Un horizonte muy grande le deja poco margen de entrenamiento al modelo, y puede hacer que "
            "las predicciones se aplanen (que casi no varíen día a día) en vez de seguir el patrón real. "
            f"Se recomienda un horizonte de hasta ~{int(filas_por_entidad_min * 0.35)} días para este dataset."
        )

    config = ConfiguracionColumnas(
        columna_fecha=columna_fecha, columna_entidad=columna_entidad,
        columna_objetivo=columna_objetivo, columnas_exogenas=columnas_exogenas,
    )

    st.subheader("Resumen del dataset")
    resumen = resumen_temporal_dataset(df, columna_fecha, columna_entidad)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Desde", resumen["fecha_min"].strftime("%Y-%m-%d"))
    c2.metric("Hasta", resumen["fecha_max"].strftime("%Y-%m-%d"))
    c3.metric("Años cubiertos (aprox.)", f"{resumen['anios_aprox']:.1f}")
    c4.metric("Entidades", resumen["n_entidades"])

    fig_diaria = graficar_serie_diaria(df, columna_fecha, columna_entidad, columna_objetivo)
    st.plotly_chart(fig_diaria, use_container_width=True, key="chart_resumen_diaria")

    fig_resumen = graficar_serie_mensual(df, columna_fecha, columna_entidad, columna_objetivo)
    st.plotly_chart(fig_resumen, use_container_width=True, key="chart_resumen_mensual")
    st.caption(
        "El gráfico de arriba muestra la demanda día a día, tal cual; el de abajo, "
        "el promedio mensual — sirve para anticipar qué forma debería tener el "
        "pronóstico (ej. si esperas un peak en ciertos meses, debería notarse acá primero)."
    )

    n_entidades_estimado = df[columna_entidad].nunique() if columna_entidad else 1
    viable, mensaje = estimar_viabilidad(len(df), n_entidades_estimado)
    if not viable:
        st.error(mensaje)
        return

    if st.button("Entrenar / cargar modelo"):
        with st.spinner("Preparando datos y entrenando (puede tardar varios minutos la primera vez)..."):
            try:
                modelo, datos, huella, se_entreno = entrenar_o_cargar_modelo(
                    df, config, ventana=int(ventana), horizonte=int(horizonte),
                )
            except ValueError as e:
                st.error(str(e))
                return
            metricas, predicciones = evaluar_modelo(modelo, datos)

        st.session_state["motor_generico_resultado"] = {
            "metricas": metricas, "predicciones": predicciones, "se_entreno": se_entreno,
        }
        st.session_state["motor_generico_modelo"] = modelo
        st.session_state["motor_generico_datos"] = datos
        st.session_state["motor_generico_df"] = df
        st.session_state["motor_generico_config"] = config

    resultado = st.session_state.get("motor_generico_resultado")
    if resultado is None:
        return

    if resultado["se_entreno"]:
        st.success("Modelo entrenado y guardado en caché.")
    else:
        st.success("Dataset ya reconocido — modelo cargado desde caché, sin reentrenar.")

    st.subheader("Métricas por entidad")
    tabla_metricas = pd.DataFrame(resultado["metricas"]).T
    st.dataframe(tabla_metricas)

    st.subheader("Validación: real vs. pronóstico")
    st.caption(
        "Este gráfico NO es un pronóstico a futuro. Muestra qué tan bien predijo el "
        "modelo un período histórico que se dejó aparte durante el entrenamiento "
        "(el 'horizonte' que configuraste), comparando lo que el modelo predijo "
        "contra lo que realmente pasó en esos días. Sirve para evaluar la precisión "
        "del modelo, no para ver qué va a pasar adelante."
    )
    entidad_graficar = st.selectbox("Entidad a graficar", options=list(resultado["predicciones"].keys()))
    pred_ent = resultado["predicciones"][entidad_graficar]

    fig = go.Figure()
    x = list(range(len(pred_ent["real"])))
    fig.add_trace(go.Scatter(x=x, y=pred_ent["real"], name="Real", line=dict(color="white", width=2)))
    fig.add_trace(go.Scatter(x=x, y=pred_ent["P50"], name="Predicción (P50)", line=dict(color="red", width=2)))
    fig.add_trace(go.Scatter(x=x, y=pred_ent["P90"], name="P90", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(
        x=x, y=pred_ent["P10"], name="Rango P10–P90", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(70,130,180,0.3)",
    ))
    fig.update_layout(
        title=f"Validación sobre datos históricos (no es pronóstico a futuro) — {entidad_graficar}",
        xaxis_title="Días del período de prueba", yaxis_title="Valor",
    )
    st.plotly_chart(fig, use_container_width=True, key="chart_validacion")

    st.subheader("Política de inventario según el histórico")
    st.caption(
        "Calcula el Punto de Reorden, el Stock de Seguridad y la meta de "
        "inventario usando la validación del modelo sobre el período de "
        "prueba (el gráfico de arriba). Refleja cómo se habría comportado "
        "la política si se hubiera aplicado en ese período histórico, no "
        "una proyección a futuro."
    )
    entidad_politica_hist = st.selectbox(
        "Entidad", options=list(resultado["predicciones"].keys()), key="entidad_politica_hist",
    )
    c1, c2, c3 = st.columns(3)
    lead_time_hist = c1.number_input("Lead time asumido (días)", min_value=1, max_value=365, value=30, key="lt_hist")
    nivel_servicio_hist = c2.selectbox(
        "Nivel de servicio deseado", options=list(Z_POR_NIVEL_SERVICIO.keys()), index=2, key="ns_hist",
    )
    periodo_revision_hist = c3.number_input(
        "Período entre revisiones — P (días)", min_value=1, max_value=90, value=30, key="pr_hist",
    )
    if st.button("Calcular política histórica"):
        politica_hist = calcular_politica_inventario(
            resultado["predicciones"][entidad_politica_hist], lead_time_hist, nivel_servicio_hist, periodo_revision_hist,
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Punto de Reorden (ROP)", f"{politica_hist['ROP']:,}")
        c2.metric("Stock de Seguridad (SS)", f"{politica_hist['SS']:,}")
        c3.metric("Meta de inventario (T)", f"{politica_hist['Meta_T']:,}")
        st.caption(
            f"Demanda promedio de validación: {politica_hist['demanda_promedio']:,} · "
            f"Desviación estándar (P90-P50): {politica_hist['sigma_demanda']:,} · Z: {politica_hist['Z']}"
        )

    modelo = st.session_state["motor_generico_modelo"]
    datos = st.session_state["motor_generico_datos"]
    df_guardado = st.session_state["motor_generico_df"]
    config_guardada = st.session_state["motor_generico_config"]

    st.divider()
    st.markdown("## 🔮 Escenario what-if")
    st.caption("Simula un evento futuro y proyecta la demanda y la política de inventario bajo ese escenario, comparada contra un escenario sin el evento.")

    contenedor_whatif = st.container(border=True)
    with contenedor_whatif:
        if not config_guardada.columnas_exogenas:
            st.info("Este dataset no tiene variables exógenas seleccionadas, así que no hay ninguna variable que se pueda simular en un escenario de estrés.")
        else:
            rangos_historicos = {
                var: (df_guardado[var].min(), df_guardado[var].max())
                for var in config_guardada.columnas_exogenas
            }
            texto_rangos = " · ".join(
                f"**{var}**: {mn:.1f} a {mx:.1f}" for var, (mn, mx) in rangos_historicos.items()
            )
            st.caption(f"Rango histórico real de cada variable (para elegir un % de cambio realista): {texto_rangos}")

            with st.form("form_whatif_generico"):
                c1, c2 = st.columns(2)
                entidad_whatif = c1.selectbox("Entidad a simular", options=list(datos.entidad_a_id.keys()), key="entidad_whatif")
                variable_afectada = c2.selectbox("Variable exógena afectada por el evento", options=config_guardada.columnas_exogenas)

                df_guardado[config_guardada.columna_fecha] = pd.to_datetime(df_guardado[config_guardada.columna_fecha])
                fecha_min_pronostico = df_guardado[config_guardada.columna_fecha].max() + pd.Timedelta(days=1)
                st.caption(
                    f"El pronóstico solo puede proyectarse hacia adelante desde el fin del historial "
                    f"({fecha_min_pronostico.strftime('%Y-%m-%d')}). El evento puede ubicarse en cualquier "
                    f"punto dentro de ese horizonte (ej. más adelante en el año), pero no en fechas ya "
                    f"cubiertas por el historial."
                )
                c3, c4 = st.columns(2)
                fecha_inicio_evento_input = c3.date_input(
                    "Inicio del evento", value=fecha_min_pronostico.date(), min_value=fecha_min_pronostico.date(),
                )
                fecha_fin_evento_input = c4.date_input(
                    "Fin del evento", value=(fecha_min_pronostico + pd.Timedelta(days=29)).date(), min_value=fecha_min_pronostico.date(),
                )
                c5, c6 = st.columns(2)
                cambio_pct = c5.slider("Cambio en la variable durante el evento (%)", -90, 200, 30) / 100
                dias_horizonte_whatif = c6.number_input(
                    "Horizonte total del pronóstico, desde el fin del historial (días)",
                    min_value=30, max_value=365, value=90,
                )
                ejecutar_whatif = st.form_submit_button("Ejecutar escenario", type="primary")

            if ejecutar_whatif:
                fecha_inicio_evento = pd.Timestamp(fecha_inicio_evento_input)
                fecha_fin_evento = pd.Timestamp(fecha_fin_evento_input)
                ultima_fecha_pronostico = fecha_min_pronostico + pd.Timedelta(days=int(dias_horizonte_whatif) - 1)

                if fecha_fin_evento < fecha_inicio_evento:
                    st.error("La fecha de fin del evento no puede ser anterior a la fecha de inicio.")
                    st.stop()
                if fecha_fin_evento > ultima_fecha_pronostico:
                    st.error(
                        f"El evento termina el {fecha_fin_evento.strftime('%Y-%m-%d')}, pero el horizonte "
                        f"total del pronóstico solo llega hasta el {ultima_fecha_pronostico.strftime('%Y-%m-%d')}. "
                        "Aumenta el horizonte total o acorta las fechas del evento."
                    )
                    st.stop()

                evento = EventoWhatIf(
                    variable_afectada=variable_afectada, fecha_inicio=fecha_inicio_evento,
                    fecha_fin=fecha_fin_evento, cambio_pct=cambio_pct, entidad=entidad_whatif,
                )
                evento_base = EventoWhatIf(
                    variable_afectada=variable_afectada, fecha_inicio=fecha_inicio_evento,
                    fecha_fin=fecha_fin_evento, cambio_pct=0.0, entidad=entidad_whatif,
                )
                with st.spinner("Generando pronóstico recursivo día a día..."):
                    trayectoria = construir_trayectoria_escenario(
                        df_guardado, config_guardada, evento,
                        dias_horizonte=int(dias_horizonte_whatif), fecha_inicio_pronostico=fecha_min_pronostico,
                    )
                    trayectoria_base = construir_trayectoria_escenario(
                        df_guardado, config_guardada, evento_base,
                        dias_horizonte=int(dias_horizonte_whatif), fecha_inicio_pronostico=fecha_min_pronostico,
                    )
                    pronostico_whatif = pronostico_recursivo(
                        modelo, datos, df_guardado, config_guardada, entidad_whatif, trayectoria
                    )
                    pronostico_base = pronostico_recursivo(
                        modelo, datos, df_guardado, config_guardada, entidad_whatif, trayectoria_base
                    )

                mask_evento = (trayectoria[config_guardada.columna_fecha] >= fecha_inicio_evento) & (
                    trayectoria[config_guardada.columna_fecha] <= fecha_fin_evento
                )
                valores_evento = trayectoria.loc[mask_evento, variable_afectada]
                hist_min, hist_max = rangos_historicos[variable_afectada]
                fuera_de_rango = (valores_evento.min() < hist_min) or (valores_evento.max() > hist_max)

                dias_evento = (fecha_fin_evento - fecha_inicio_evento).days + 1
                mask_dias_evento_pron = (pronostico_whatif[config_guardada.columna_fecha] >= fecha_inicio_evento) & (
                    pronostico_whatif[config_guardada.columna_fecha] <= fecha_fin_evento
                )
                diferencia_promedio = (
                    pronostico_whatif.loc[mask_dias_evento_pron, "P50"].values
                    - pronostico_base.loc[mask_dias_evento_pron, "P50"].values
                ).mean()

                historial_reciente = df_guardado[
                    (df_guardado[config_guardada.columna_entidad] == entidad_whatif)
                    & (df_guardado[config_guardada.columna_fecha] < fecha_min_pronostico)
                ].sort_values(config_guardada.columna_fecha).tail(90)

                st.session_state["motor_generico_whatif"] = {
                    "pronostico": pronostico_whatif, "pronostico_base": pronostico_base, "entidad": entidad_whatif,
                    "fecha_inicio_evento": fecha_inicio_evento, "fecha_fin_evento": fecha_fin_evento,
                    "fuera_de_rango": fuera_de_rango,
                    "rango_evento": (round(valores_evento.min(), 1), round(valores_evento.max(), 1)),
                    "rango_historico": (round(hist_min, 1), round(hist_max, 1)),
                    "diferencia_promedio": round(diferencia_promedio, 2),
                    "historial_reciente": historial_reciente,
                }

            whatif_resultado = st.session_state.get("motor_generico_whatif")
            if whatif_resultado is not None:
                if whatif_resultado["fuera_de_rango"]:
                    rmin, rmax = whatif_resultado["rango_historico"]
                    emin, emax = whatif_resultado["rango_evento"]
                    st.warning(
                        f"El escenario lleva la variable a un rango de {emin} a {emax}, "
                        f"por fuera del rango histórico observado ({rmin} a {rmax}). "
                        "El modelo nunca vio valores así durante el entrenamiento, así que "
                        "está extrapolando: los resultados en esta zona pueden no ser "
                        "confiables o comportarse de forma poco intuitiva. Prueba un "
                        "porcentaje de cambio más moderado para un escenario más realista."
                    )
                pron = whatif_resultado["pronostico"]
                pron_base = whatif_resultado["pronostico_base"]
                mask_dias_evento_pron_actual = (
                    (pron[config_guardada.columna_fecha] >= whatif_resultado["fecha_inicio_evento"])
                    & (pron[config_guardada.columna_fecha] <= whatif_resultado["fecha_fin_evento"])
                )
                hist_reciente = whatif_resultado["historial_reciente"]
                fig_wi = go.Figure()
                if len(hist_reciente) > 0:
                    fig_wi.add_trace(go.Scatter(
                        x=hist_reciente[config_guardada.columna_fecha], y=hist_reciente[config_guardada.columna_objetivo],
                        name="Historial reciente (real)", line=dict(color="white", width=1.5),
                    ))
                fig_wi.add_trace(go.Scatter(x=pron[config_guardada.columna_fecha], y=pron["P90"], line=dict(width=0), showlegend=False))
                fig_wi.add_trace(go.Scatter(
                    x=pron[config_guardada.columna_fecha], y=pron["P10"], line=dict(width=0),
                    fill="tonexty", fillcolor="rgba(70,130,180,0.3)", name="Rango P10–P90 (con evento)",
                ))
                fig_wi.add_trace(go.Scatter(x=pron[config_guardada.columna_fecha], y=pron["P50"], name="Con evento (P50)", line=dict(color="red", width=2)))
                fig_wi.add_vrect(
                    x0=whatif_resultado["fecha_inicio_evento"], x1=whatif_resultado["fecha_fin_evento"],
                    fillcolor="orange", opacity=0.15, annotation_text="Evento", line_width=0,
                )
                fig_wi.update_layout(
                    title=f"Pronóstico bajo el escenario — {whatif_resultado['entidad']}",
                    xaxis_title="Fecha", yaxis_title=config_guardada.columna_objetivo, height=420,
                )
                st.plotly_chart(fig_wi, use_container_width=True, key="chart_whatif")
                st.caption(
                    f"Diferencia promedio durante el evento, respecto a un escenario sin el evento: "
                    f"{whatif_resultado['diferencia_promedio']:+.2f} {config_guardada.columna_objetivo}/día. "
                    "La línea blanca es historial real; desde ahí en adelante todo es pronóstico."
                )

                st.subheader("Política de inventario proyectada")
                entidad_pol_wi = whatif_resultado["entidad"]
                c1, c2, c3 = st.columns(3)
                lead_time_wi = c1.number_input("Lead time asumido (días)", min_value=1, max_value=365, value=30, key="lt_wi")
                nivel_servicio_wi = c2.selectbox(
                    "Nivel de servicio deseado", options=list(Z_POR_NIVEL_SERVICIO.keys()), index=2, key="ns_wi",
                )
                periodo_revision_wi = c3.number_input(
                    "Período entre revisiones — P (días)", min_value=1, max_value=90, value=30, key="pr_wi",
                )

                if st.button("Calcular política proyectada"):
                    pred_con_evento = {
                        "P50": pron.loc[mask_dias_evento_pron_actual, "P50"].values,
                        "P90": pron.loc[mask_dias_evento_pron_actual, "P90"].values,
                    }
                    pred_sin_evento = {
                        "P50": pron_base.loc[mask_dias_evento_pron_actual, "P50"].values,
                        "P90": pron_base.loc[mask_dias_evento_pron_actual, "P90"].values,
                    }
                    politica_con = calcular_politica_inventario(pred_con_evento, lead_time_wi, nivel_servicio_wi, periodo_revision_wi)
                    politica_sin = calcular_politica_inventario(pred_sin_evento, lead_time_wi, nivel_servicio_wi, periodo_revision_wi)

                    col_con, col_sin = st.columns(2)
                    with col_con:
                        st.markdown(f"**Con evento** — {entidad_pol_wi}")
                        st.metric("Punto de Reorden (ROP)", f"{politica_con['ROP']:,}")
                        st.metric("Stock de Seguridad (SS)", f"{politica_con['SS']:,}")
                        st.metric("Meta de inventario (T)", f"{politica_con['Meta_T']:,}")
                    with col_sin:
                        st.markdown(f"**Sin evento** — {entidad_pol_wi}")
                        st.metric("Punto de Reorden (ROP)", f"{politica_sin['ROP']:,}")
                        st.metric("Stock de Seguridad (SS)", f"{politica_sin['SS']:,}")
                        st.metric("Meta de inventario (T)", f"{politica_sin['Meta_T']:,}")


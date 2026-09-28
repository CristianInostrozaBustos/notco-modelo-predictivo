import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import estilo as E
import sesion as S
from motor import datos as D
from motor import reglas as R

E.encabezado(
    "Paso 1",
    "Tus datos",
    "Sube tu historial de ventas o demanda en CSV o Excel. El sistema reconoce las columnas y deja todo listo para pronosticar.",
)

# ---------------------------------------------------------------- origen
with st.container(border=True):
    origen = st.segmented_control("Origen", ["Subir archivo", "Usar un ejemplo"], default="Subir archivo",
                                  label_visibility="collapsed", key="origen_datos")
    if origen == "Usar un ejemplo":
        opciones = list(S.EJEMPLOS)
        elegido = st.radio("Dataset de ejemplo", opciones, format_func=lambda a: S.EJEMPLOS[a][0],
                           captions=[S.EJEMPLOS[a][1] for a in opciones], label_visibility="collapsed")
        st.caption(":material/info: Datos de ejemplo generados para probar el sistema.")
        if st.button("Cargar ejemplo", type="primary", icon=":material/download:"):
            st.session_state["df_raw"] = S.cargar_ejemplo(elegido)
            st.session_state["nombre_dataset"] = elegido
    else:
        archivo = st.file_uploader("Archivo CSV o Excel", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
        if archivo is not None and archivo.name != st.session_state.get("nombre_dataset"):
            try:
                st.session_state["df_raw"] = S.leer(archivo.name, archivo.getvalue())
                st.session_state["archivo_bytes"] = archivo.getvalue()
                st.session_state["nombre_dataset"] = archivo.name
            except Exception as e:  # noqa: BLE001
                st.error(f"No pudimos leer el archivo: {e}")

df = st.session_state.get("df_raw")
if df is None:
    S.panel_dataset()
    st.stop()

nombre = st.session_state["nombre_dataset"]
k = f"{nombre}_{len(df)}"
det = D.detectar_roles(df)

# ---------------------------------------------------------------- columnas (plegado)
NINGUNA = "(no tiene)"
columnas = list(df.columns)
problemas = not det.roles.get("fecha") or not det.roles.get("objetivo")
roles = {}
with st.expander("Revisar columnas detectadas", icon=":material/view_column:", expanded=problemas):
    for a in det.advertencias:
        st.warning(a, icon=":material/warning:")
    c1, c2, c3 = st.columns(3)
    for col, rol in zip((c1, c2, c3), ("fecha", "objetivo", "entidad")):
        opciones = ([NINGUNA] if rol == "entidad" else []) + columnas
        d = det.roles.get(rol)
        sel = col.selectbox(D.ROLES[rol]["etiqueta"], opciones, index=opciones.index(d) if d in opciones else 0,
                            key=f"rol_{rol}_{k}")
        roles[rol] = None if sel == NINGUNA else sel
    st.caption("Opcionales: si tu archivo las tiene, el sistema las usa para mejorar el pronóstico y las decisiones.")
    cols = st.columns(3)
    for i, rol in enumerate(["precio", "promocion", "lead_time", "inventario", "quiebre", "costo_unitario"]):
        opciones = [NINGUNA] + columnas
        d = det.roles.get(rol)
        sel = cols[i % 3].selectbox(D.ROLES[rol]["etiqueta"], opciones, index=opciones.index(d) if d in opciones else 0,
                                    key=f"rol_{rol}_{k}")
        roles[rol] = None if sel == NINGUNA else sel
    asignadas = {v for v in roles.values() if v}
    candidatas = [c for c in columnas if c not in asignadas and pd.api.types.is_numeric_dtype(df[c])]
    exogenas = st.multiselect("Otras variables que influyen en la demanda", candidatas,
                              default=[c for c in det.exogenas if c in candidatas], key=f"exog_{k}",
                              placeholder="Ninguna",
                              help="Por ejemplo clima, tráfico o un índice de mercado.")

    st.caption("Opciones avanzadas")
    fechas_tmp, _ = D.parsear_fechas(df[roles["fecha"]]) if roles.get("fecha") else (None, None)
    freq_det, finfo = ("D", {"frac_duplicadas": 0})
    if fechas_tmp is not None and fechas_tmp.notna().mean() >= 0.5:
        freq_det, finfo = D.detectar_frecuencia(fechas_tmp, df[roles["entidad"]] if roles.get("entidad") else None)
    a1, a2, a3 = st.columns(3)
    codigos = list(D.FRECUENCIAS)
    frecuencia = a1.selectbox("Agrupar los datos por", codigos, index=codigos.index(freq_det), key=f"freq_{k}",
                              format_func=lambda c: D.FRECUENCIAS[c]["unidad"].capitalize())
    relleno = a2.selectbox("Períodos sin registro", ["interpolar", "cero"], key=f"relleno_{k}",
                           index=1 if finfo["frac_duplicadas"] > 0.2 else 0,
                           format_func=lambda x: {"interpolar": "Faltan datos (interpolar)", "cero": "No hubo ventas (cero)"}[x])
    negativos = a3.toggle("Tratar negativos como cero", value=True, key=f"neg_{k}")

repetidas = [c for c in asignadas if list(roles.values()).count(c) > 1]
if repetidas:
    st.error(f"La columna '{repetidas[0]}' está asignada dos veces. Revisa las columnas.")
    st.stop()
if not roles.get("fecha") or not roles.get("objetivo"):
    st.error("Indica cuál columna tiene la fecha y cuál la cantidad que quieres pronosticar.")
    st.stop()
if fechas_tmp is None or fechas_tmp.notna().mean() < 0.5:
    st.error(f"La columna '{roles['fecha']}' no contiene fechas reconocibles.")
    st.stop()

try:
    dp = S.preparar_cacheado(df, tuple(sorted(roles.items())), tuple(exogenas), frecuencia, relleno, negativos)
except Exception as e:  # noqa: BLE001
    st.error(f"No pudimos preparar los datos: {e}")
    st.stop()
st.session_state["dp"] = dp
st.session_state["config_actual"] = dict(roles=roles, exogenas=list(exogenas), frecuencia=frecuencia, relleno=relleno,
                                         negativos=bool(negativos))
plan = R.planificar(dp)
fi = dp.freq_info
n_ent = dp.df["entidad"].nunique()

# ---------------------------------------------------------------- resumen amable
f0, f1 = dp.df["fecha"].min(), dp.df["fecha"].max()
c1, c2, c3 = st.columns(3)
c1.metric(S.nombre_entidad(dp, n_ent != 1).capitalize(), E.num(n_ent))
c2.metric("Historial", f"{f0:%m/%Y} – {f1:%m/%Y}")
c3.metric(f"Total {dp.etiquetas['objetivo']}", E.num(dp.df["objetivo"].sum()))

with st.container(border=True):
    total = dp.df.groupby("fecha")["objetivo"].sum().reset_index()
    fig = go.Figure(go.Scatter(x=total["fecha"], y=total["objetivo"], line=dict(color=E.AZUL, width=1.5),
                               hovertemplate="%{y:,.0f}", name="Total"))
    fig.update_layout(title=f"{dp.etiquetas['objetivo']} por {fi['unidad']}", height=280, showlegend=False)
    E.grafico(fig, key="fig_total")

if not plan.viable:
    st.error(f"El historial es demasiado corto para pronosticar: se necesitan al menos "
             f"{R.minimo_registros(dp.config.frecuencia)} {fi['unidad_pl']} por {S.nombre_entidad(dp)}.")
    S.panel_dataset()
    st.stop()
if plan.entidades_excluidas:
    st.warning(f"{len(plan.entidades_excluidas)} {S.nombre_entidad(dp, True)} tienen muy poco historial y no se "
               f"pronosticarán: {', '.join(map(str, plan.entidades_excluidas[:8]))}"
               f"{'…' if len(plan.entidades_excluidas) > 8 else ''}.", icon=":material/warning:")

# ---------------------------------------------------------------- horizonte y botón
st.markdown("## ¿Cuánto quieres pronosticar?")
clave = S.clave_dataset(dp)
_, res = S.resultado()
with st.container(border=True):
    h_prev = st.session_state.get("horizonte") or plan.horizonte_defecto
    h = st.slider(f"{fi['unidad_pl'].capitalize()} hacia adelante", 1, plan.horizonte_max,
                  min(h_prev, plan.horizonte_max), key=f"h_{clave}")
    st.caption(f"Hasta {plan.horizonte_max} {fi['unidad_pl']} según el historial disponible. "
               f"El pronóstico empieza el {(f1 + pd.tseries.frequencies.to_offset(D.FRECUENCIAS[dp.config.frecuencia]['pandas'])):%d/%m/%Y}.")
    if res is None:
        generar = st.button("Generar pronóstico", type="primary", icon=":material/auto_graph:")
        if generar:
            barra = st.progress(0.0, text="Preparando…")
            res_nuevo, origen = S.entrenar(clave, dp, plan, lambda frac, txt: barra.progress(min(frac, 1.0), text=txt))
            barra.empty()
            st.session_state["resultado"] = {"clave": clave, "res": res_nuevo, "origen": origen}
            st.session_state["horizonte"] = h
            st.rerun()
        if S.modelo_guardado(clave):
            st.caption(":material/bolt: Estos datos ya se analizaron antes: el pronóstico sale al instante.")
        else:
            st.caption(":material/schedule: Toma entre 1 y 4 minutos según el tamaño de tus datos. El sistema elige solo "
                       "la mejor forma de entrenar para tu dataset, y la guarda para la próxima vez.")
    else:
        st.session_state["horizonte"] = h
        S.actualizar_registro(horizonte=int(h))
        origen = st.session_state["resultado"].get("origen")
        import cuenta
        error_guardado = S.guardar_pronostico_actual()
        if error_guardado:
            st.session_state.setdefault("avisos_almacen", []).append(error_guardado)
        st.success("Pronóstico listo." + (" Recuperado de un análisis anterior de estos mismos datos." if origen == "guardado" else ""),
                   icon=":material/check_circle:")
        reg = S.registro_actual()
        if reg:
            st.caption(f":material/cloud_done: Guardado en **Mis pronósticos** como “{reg['nombre']}”.")
        elif cuenta.login_disponible() and not cuenta.usuario():
            st.caption(":material/lock_open: Estás usando el modo abierto: el pronóstico no se guarda a tu nombre. "
                       "Inicia sesión (barra lateral) para guardarlo y volver a él cuando quieras.")
        for aviso in st.session_state.pop("avisos_almacen", []):
            if S.modo_dev():
                st.warning(aviso, icon=":material/cloud_off:")
        b1, b2, b3 = st.columns(3)
        b1.page_link("paginas/pronostico.py", label="Ver pronóstico", icon=":material/show_chart:")
        b2.page_link("paginas/decisiones.py", label="Ver decisiones de abastecimiento", icon=":material/inventory_2:")
        b3.page_link("paginas/analisis.py", label="Explorar mis datos", icon=":material/insights:")

S.panel_dataset()

"""Estado compartido entre páginas: dataset activo, modelo entrenado, pronóstico y modo desarrollador."""

import hashlib
import os

import pandas as pd
import streamlit as st

import estilo as E
from motor import datos as D
from motor import reglas as R

CARPETA_EJEMPLOS = "ejemplos"

EJEMPLOS = {
    "retail_diario_tiendas.csv": ("Retail diario · 8 tiendas",
                                  "3 años de ventas diarias con precio, promoción, stock y lead time."),
    "ventas_semanales.csv": ("Ventas semanales · 3 productos", "2 años de ventas semanales con precio."),
    "transacciones.csv": ("Transacciones · 2 SKUs", "Un año de ventas registradas por transacción."),
    "mensual_serie_unica.csv": ("Demanda mensual · 1 producto", "6 años de demanda mensual."),
    "muchos_skus.csv": ("Catálogo grande · 80 SKUs", "Algo más de un año de demanda diaria de 80 productos."),
}


# ---------------------------------------------------------------- modo desarrollador

def modo_dev() -> bool:
    if st.query_params.get("dev") == "1":
        st.session_state["dev"] = True
    elif st.query_params.get("dev") == "0":
        st.session_state["dev"] = False
    return st.session_state.get("dev", False)


# ---------------------------------------------------------------- lectura y preparación

@st.cache_data(show_spinner="Leyendo el archivo...")
def leer(nombre, contenido):
    return D.leer_archivo(nombre, contenido)


def cargar_ejemplo(archivo):
    with open(os.path.join(CARPETA_EJEMPLOS, archivo), "rb") as fh:
        contenido = fh.read()
    st.session_state["archivo_bytes"] = contenido
    return leer(archivo, contenido)


@st.cache_data(show_spinner="Preparando los datos...")
def preparar_cacheado(df, roles_items, exogenas, frecuencia, relleno, negativos):
    cfg = D.Configuracion(roles=dict(roles_items), exogenas=list(exogenas), frecuencia=frecuencia,
                          relleno_objetivo=relleno, negativos_a_cero=negativos)
    return D.preparar(df, cfg)


def clave_dataset(dp) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(dp.df, index=False).values.tobytes())
    h.update(str(dp.variables_modelo).encode())
    h.update(dp.config.frecuencia.encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- entrenamiento y pronóstico

@st.cache_resource(show_spinner=False)
def _almacen_modelos():
    """Modelos en memoria, compartidos entre usuarios del mismo servidor (clave = huella del dataset)."""
    return {}


def _config_almacen():
    try:
        return dict(st.secrets["almacen"]) if "almacen" in st.secrets else None
    except Exception:  # noqa: BLE001  (sin archivo de secrets)
        return None


@st.cache_resource(show_spinner=False)
def almacen_persistente():
    """Dónde se guardan modelos y archivos: Supabase Storage si hay secrets, si no una carpeta local."""
    from motor import almacen as A
    return A.crear(_config_almacen())


@st.cache_resource(show_spinner=False)
def repositorio():
    """Registro de Mis pronósticos: Supabase Postgres si hay secrets, si no SQLite local."""
    from motor import repositorio as Rp
    return Rp.crear(_config_almacen())


MAX_MODELOS_EN_MEMORIA = 4


def _a_memoria(clave, res):
    memoria = _almacen_modelos()
    while len(memoria) >= MAX_MODELOS_EN_MEMORIA:
        memoria.pop(next(iter(memoria)))
    memoria[clave] = res


def modelo_guardado(clave) -> bool:
    """True si ya existe un modelo entrenado para estos datos (en memoria o guardado)."""
    if clave in _almacen_modelos():
        return True
    from motor import almacen as A
    return A.existe_modelo(almacen_persistente(), clave)


def entrenar(clave, dp, plan, progreso=None):
    """Recupera el modelo si ya existe (memoria → almacén); si no, entrena y lo guarda.

    Devuelve (res, origen) con origen "memoria", "guardado" o "nuevo".
    No se usa st.cache_* directo porque la barra de progreso vive fuera de la función.
    """
    from motor import almacen as A
    memoria = _almacen_modelos()
    if clave in memoria:
        if progreso:
            progreso(1.0, "Listo")
        return memoria[clave], "memoria"
    alm = almacen_persistente()
    if progreso:
        progreso(0.02, "Buscando si estos datos ya se entrenaron…")
    res, err = A.cargar(alm, clave)
    if err:
        st.session_state.setdefault("avisos_almacen", []).append(f"No se pudo leer el modelo guardado: {err}")
    if res is not None:
        _a_memoria(clave, res)
        if progreso:
            progreso(1.0, "Listo")
        return res, "guardado"
    from motor import modelo as M
    res = M.entrenar_motor(dp, plan, progreso)
    _a_memoria(clave, res)
    err = A.guardar(alm, clave, res)
    if err:
        st.session_state.setdefault("avisos_almacen", []).append(f"No se pudo guardar el modelo: {err}")
    return res, "nuevo"


def resultado():
    """(DatasetPreparado, ResultadoModelo) si hay un modelo entrenado para el dataset activo."""
    dp = st.session_state.get("dp")
    r = st.session_state.get("resultado")
    if dp is None or r is None or r["clave"] != clave_dataset(dp):
        return dp, None
    return dp, r["res"]


def horizonte():
    return st.session_state.get("horizonte")


@st.cache_data(show_spinner="Calculando el pronóstico...", max_entries=20)
def _pronostico_cacheado(clave, h, freq):
    res = st.session_state["resultado"]["res"]
    from motor import modelo as M
    return M.pronosticar(res, h, freq)


def pronostico(h=None):
    dp, res = resultado()
    if res is None:
        return None
    return _pronostico_cacheado(st.session_state["resultado"]["clave"], h or horizonte(), dp.config.frecuencia)


@st.cache_data(show_spinner="Simulando el escenario...", max_entries=20)
def _pronostico_escenario_cacheado(clave, h, freq, cambios, entidades):
    res = st.session_state["resultado"]["res"]
    from motor import modelo as M
    return M.pronosticar(res, h, freq, cambios=[dict(c) for c in cambios], entidades=list(entidades))


def pronostico_escenario(h, cambios, entidades):
    """Pronóstico con cambios en variables del modelo (escenarios). Sin cambios, devuelve el base."""
    dp, res = resultado()
    if not cambios:
        base = pronostico(h)
        return {e: base[e] for e in entidades}
    cambios_t = tuple(tuple(sorted(c.items())) for c in cambios)
    return _pronostico_escenario_cacheado(st.session_state["resultado"]["clave"], h, dp.config.frecuencia,
                                          cambios_t, tuple(entidades))


def politica_actual(dp, entidades):
    """Parámetros de la política (lead time, inventario, nivel de servicio, revisión).

    Si el usuario ya los editó en Decisiones se usan esos; si no, los del archivo o valores por defecto.
    """
    from motor import politica as P
    guardada = st.session_state.get("politica")
    if guardada and guardada.get("clave") == clave_dataset(dp):
        return guardada["tabla"], guardada["nivel"], guardada["revision"]
    abierta = st.session_state.get("politica_guardada")
    if abierta and abierta.get("clave") == clave_dataset(dp) and abierta.get("tabla"):
        return pd.DataFrame(abierta["tabla"]), abierta.get("nivel", "90%"), abierta.get("revision", 30)
    filas = []
    for e in entidades:
        g = dp.df[dp.df["entidad"] == e]
        filas.append({
            "entidad": e,
            "lead_time": round(P.lead_time_dataset(g["lead_time"]), 1) if dp.tiene("lead_time") else 14.0,
            "inventario": float(g["inventario"].iloc[-1]) if dp.tiene("inventario") else float("nan"),
        })
    rev = {"D": 30, "W": 28, "M": 30, "Q": 91}[dp.config.frecuencia]
    return pd.DataFrame(filas), "90%", rev


# ---------------------------------------------------------------- UI compartida

def nombre_entidad(dp, plural=False):
    et = dp.etiquetas.get("entidad", "")
    if et == "(una sola serie)" or not et:
        return "series" if plural else "serie"
    et = et.strip().lower()
    if plural:
        return et if et.endswith("s") else et + ("es" if et[-1] in "rlnd" else "s")
    return et


def _sync_ent():
    st.session_state["entidad"] = st.session_state["_entidad_widget"]


def selector_entidad(entidades, dp):
    if st.session_state.get("entidad") not in entidades:
        st.session_state["entidad"] = entidades[0]
    st.session_state["_entidad_widget"] = st.session_state["entidad"]
    if len(entidades) == 1:
        return entidades[0]
    with st.sidebar:
        st.markdown(f"##### {nombre_entidad(dp).capitalize()}")
        return st.selectbox(nombre_entidad(dp), entidades, key="_entidad_widget", on_change=_sync_ent,
                            label_visibility="collapsed")


def panel_dataset():
    dp = st.session_state.get("dp")
    with st.sidebar:
        st.markdown("##### Tus datos")
        if dp is None:
            st.caption("Todavía no cargas un dataset.")
            return
        nombre = st.session_state.get("nombre_dataset", "dataset")
        _, res = resultado()
        estado = E.insignia("Pronóstico listo", "ok") if res is not None else E.insignia("Sin pronóstico", "neutro")
        n = dp.df["entidad"].nunique()
        st.markdown(
            f"""<div style="display:flex;flex-direction:column;gap:10px">
              <div class="ficha"><span class="l">Archivo</span><span class="v" style="word-break:break-all">{nombre}</span></div>
              <div style="display:flex;gap:18px">
                <div class="ficha"><span class="l">{nombre_entidad(dp, n != 1).capitalize()}</span><span class="v">{E.num(n)}</span></div>
                <div class="ficha"><span class="l">Datos</span><span class="v">{dp.freq_info['nombre'].capitalize()}</span></div>
              </div>
              <div>{estado}</div>
            </div>""",
            unsafe_allow_html=True,
        )


def requiere_pronostico():
    """Detiene la página con un mensaje amable si todavía no hay pronóstico."""
    dp, res = resultado()
    if dp is None or res is None:
        panel_dataset()
        st.info("Primero carga tus datos y genera el pronóstico.", icon=":material/info:")
        st.page_link("paginas/datos.py", label="Ir a Datos", icon=":material/arrow_forward:")
        st.stop()
    return dp, res


# ---------------------------------------------------------------- Mis pronósticos

def _usuario():
    import cuenta
    return cuenta.usuario()


def registro_actual():
    """Registro de Mis pronósticos del dataset activo (si hay sesión y ya se guardó)."""
    dp = st.session_state.get("dp")
    r = st.session_state.get("registro")
    if not r or dp is None or r["clave"] != clave_dataset(dp) or not _usuario():
        return None
    return r


def guardar_pronostico_actual():
    """Crea el registro en Mis pronósticos al generar un pronóstico con sesión iniciada. Devuelve el error o None."""
    u = _usuario()
    dp, res = resultado()
    if not u or res is None or registro_actual():
        return None
    from motor import almacen as A
    from motor import repositorio as Rp
    nombre = st.session_state.get("nombre_dataset", "dataset")
    try:
        reg = repositorio().crear(u["correo"], dict(
            nombre=os.path.splitext(nombre)[0].replace("_", " ").capitalize(),
            archivo_nombre=nombre, clave_modelo=clave_dataset(dp), frecuencia=dp.config.frecuencia,
            n_entidades=int(dp.df["entidad"].nunique()), horizonte=int(horizonte() or 0),
            error_pct=float(res.metricas_entidad["wape"].mean()),
            config=st.session_state.get("config_actual", {}), escenarios=[],
        ))
        contenido = st.session_state.get("archivo_bytes")
        if contenido:
            almacen_persistente().escribir(A.ruta_datos(Rp.id_usuario(u["correo"]), reg["id"]), contenido)
        st.session_state["registro"] = {"id": reg["id"], "clave": clave_dataset(dp), "nombre": reg["nombre"],
                                        "escenarios": []}
        return None
    except Exception as e:  # noqa: BLE001
        return f"No se pudo guardar en Mis pronósticos: {type(e).__name__}: {e}"


def actualizar_registro(**cambios):
    """Actualiza el registro activo solo si algo cambió respecto de lo último guardado."""
    r = registro_actual()
    if not r:
        return
    import json
    firma = json.dumps(cambios, sort_keys=True, default=str)
    ultimas = st.session_state.setdefault("_firmas_registro", {})
    clave_firma = (r["id"], tuple(sorted(cambios)))
    if ultimas.get(clave_firma) == firma:
        return
    try:
        repositorio().actualizar(_usuario()["correo"], r["id"], cambios)
        ultimas[clave_firma] = firma
    except Exception as e:  # noqa: BLE001
        st.session_state.setdefault("avisos_almacen", []).append(f"No se pudo actualizar el registro: {e}")


def politica_a_json(tabla, nivel, revision):
    filas = []
    for f in tabla.to_dict("records"):
        filas.append({k: (None if isinstance(v, float) and v != v else v) for k, v in f.items()})
    return {"nivel": nivel, "revision": revision, "tabla": filas}


def abrir_pronostico(reg, progreso=None):
    """Deja la sesión como estaba al guardar: datos, columnas, modelo, horizonte, política y escenarios."""
    from motor import almacen as A
    from motor import repositorio as Rp
    u = _usuario()
    contenido = almacen_persistente().leer(A.ruta_datos(Rp.id_usuario(u["correo"]), reg["id"]))
    if contenido is None:
        raise FileNotFoundError("No se encontró el archivo de datos de este pronóstico.")
    nombre = reg["archivo_nombre"]
    df = leer(nombre, contenido)
    st.session_state.update(df_raw=df, nombre_dataset=nombre, archivo_bytes=contenido)

    cfg = reg.get("config") or {}
    roles = cfg.get("roles", {})
    k = f"{nombre}_{len(df)}"
    for rol, col in roles.items():            # la página Datos muestra las columnas tal como se guardaron
        st.session_state[f"rol_{rol}_{k}"] = col if col else "(no tiene)"
    for campo, llave in (("exogenas", "exog"), ("frecuencia", "freq"), ("relleno", "relleno"), ("negativos", "neg")):
        if campo in cfg:
            st.session_state[f"{llave}_{k}"] = cfg[campo]

    dp = preparar_cacheado(df, tuple(sorted(roles.items())), tuple(cfg.get("exogenas", [])), cfg.get("frecuencia", "D"),
                           cfg.get("relleno", "interpolar"), cfg.get("negativos", True))
    st.session_state["dp"] = dp
    st.session_state["config_actual"] = cfg
    clave = clave_dataset(dp)
    plan = R.planificar(dp)
    res, origen = entrenar(clave, dp, plan, progreso)
    st.session_state["resultado"] = {"clave": clave, "res": res, "origen": origen}
    h = int(reg.get("horizonte") or plan.horizonte_defecto)
    st.session_state["horizonte"] = h
    st.session_state[f"h_{clave}"] = min(h, plan.horizonte_max)
    st.session_state["registro"] = {"id": reg["id"], "clave": clave, "nombre": reg["nombre"],
                                    "escenarios": list(reg.get("escenarios") or [])}
    pol = reg.get("politica")
    st.session_state["politica_guardada"] = {"clave": clave, **pol} if pol else None
    st.session_state.pop("politica", None)
    st.session_state.pop("escenario", None)
    return origen

"""Motor de datos: carga, detección de roles, frecuencia, limpieza y diagnóstico.

Todo el resto del sitio trabaja sobre un `DatasetPreparado`, que tiene nombres de
columna canónicos (fecha, entidad, objetivo, precio, ...), una sola fila por
entidad y período, y una frecuencia regular.
"""

from __future__ import annotations

import io
import re
import unicodedata
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- roles

ROLES = {
    "fecha": dict(etiqueta="Fecha", obligatorio=True,
                  claves=["fecha", "date", "dia", "day", "periodo", "period", "timestamp", "datetime", "semana", "week", "mes", "month", "ds"]),
    "entidad": dict(etiqueta="Entidad (SKU, tienda, producto)", obligatorio=False,
                    claves=["sku", "producto", "product", "item", "articulo", "tienda", "store", "sucursal", "local",
                            "codigo", "code", "id", "categoria", "category", "cliente", "customer", "serie", "unique_id"]),
    "objetivo": dict(etiqueta="Variable a pronosticar", obligatorio=True,
                     claves=["demanda", "demand", "venta", "ventas", "sales", "unidades", "units", "cantidad", "qty",
                             "quantity", "volumen", "volume", "pedidos", "orders", "consumo", "y", "target"]),
    "precio": dict(etiqueta="Precio", obligatorio=False, claves=["precio", "price", "pvp", "tarifa"]),
    "promocion": dict(etiqueta="Promoción (0/1)", obligatorio=False,
                      claves=["promo", "promocion", "promotion", "oferta", "descuento", "discount", "onpromotion", "campana"]),
    "lead_time": dict(etiqueta="Lead time (días)", obligatorio=False,
                      claves=["lead_time", "leadtime", "lead", "plazo_entrega", "tiempo_entrega", "plazo", "reposicion"]),
    "inventario": dict(etiqueta="Inventario disponible", obligatorio=False,
                       claves=["inventario", "stock", "existencia", "existencias", "inventory", "on_hand", "saldo"]),
    "quiebre": dict(etiqueta="Quiebre de stock (0/1)", obligatorio=False,
                    claves=["quiebre", "stockout", "rotura", "faltante", "sin_stock", "out_of_stock"]),
    "costo_unitario": dict(etiqueta="Costo unitario", obligatorio=False,
                           claves=["costo_unitario", "costo", "cost", "unit_cost", "precio_compra", "precio_unitario_primario"]),
}
ROLES_MODELO = ("precio", "promocion")  # roles que, además, entran al modelo como exógenas
ROLES_POLITICA = ("lead_time", "inventario", "quiebre", "costo_unitario")  # solo para indicadores y política


def _normalizar(nombre: str) -> str:
    s = unicodedata.normalize("NFKD", str(nombre)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return s


def _score_nombre(col: str, claves: list[str]) -> float:
    n = _normalizar(col)
    tokens = set(n.split("_"))
    mejor = 0.0
    for c in claves:
        if n == c:
            return 1.0
        if c in tokens or ("_" in c and c in n):
            mejor = max(mejor, 0.85)
        elif len(c) >= 4 and c in n:
            mejor = max(mejor, 0.6)
    return mejor


# ---------------------------------------------------------------- carga

def leer_archivo(nombre: str, contenido: bytes) -> pd.DataFrame:
    """Lee CSV (detecta separador , ; tab y decimales con coma) o Excel."""
    ext = nombre.lower().rsplit(".", 1)[-1]
    if ext in ("xlsx", "xls", "xlsm"):
        return pd.read_excel(io.BytesIO(contenido))
    texto = contenido.decode("utf-8-sig", errors="replace")
    muestra = texto[:20000]
    sep = max([",", ";", "\t", "|"], key=lambda s: muestra.count(s))
    decimal = "," if sep == ";" and re.search(r"\d,\d", muestra) else "."
    return pd.read_csv(io.StringIO(texto), sep=sep, decimal=decimal)


# ---------------------------------------------------------------- fechas

def parsear_fechas(serie: pd.Series) -> tuple[pd.Series, str]:
    """Convierte a fecha probando formato ISO, día/mes y mes/día. Devuelve (fechas, nota)."""
    if pd.api.types.is_datetime64_any_dtype(serie):
        return serie, "ya venía como fecha"
    if pd.api.types.is_numeric_dtype(serie):
        # años sueltos (2019, 2020...) o yyyymmdd
        v = serie.dropna()
        if len(v) and v.between(1900, 2100).all():
            return pd.to_datetime(serie.astype("Int64").astype(str), format="%Y", errors="coerce"), "años"
        if len(v) and v.between(19000101, 21001231).all():
            return pd.to_datetime(serie.astype("Int64").astype(str), format="%Y%m%d", errors="coerce"), "formato AAAAMMDD"
        return pd.Series(pd.NaT, index=serie.index), "numérica, no es fecha"

    texto = serie.astype(str).str.strip()
    candidatos = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for nota, kwargs in [
            ("formato ISO (AAAA-MM-DD)", dict(format="ISO8601")),
            ("día/mes/año", dict(dayfirst=True, format="mixed")),
            ("mes/día/año", dict(dayfirst=False, format="mixed")),
        ]:
            try:
                f = pd.to_datetime(texto, errors="coerce", **kwargs)
            except (ValueError, TypeError):
                continue
            candidatos.append((f.notna().mean(), nota, f))
    if not candidatos:
        return pd.Series(pd.NaT, index=serie.index), "no se pudo interpretar"

    # Entre día/mes y mes/día: si algún primer número es > 12, manda ese orden
    partes = texto.str.extract(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.]\d{2,4}")
    if partes.notna().all(axis=1).mean() > 0.5:
        a = pd.to_numeric(partes[0], errors="coerce")
        b = pd.to_numeric(partes[1], errors="coerce")
        preferido = "día/mes/año" if (a > 12).any() else ("mes/día/año" if (b > 12).any() else "día/mes/año")
        for tasa, nota, f in candidatos:
            if nota == preferido and tasa >= 0.9:
                return f, nota
    tasa, nota, f = max(candidatos, key=lambda c: c[0])
    return f, nota


# ---------------------------------------------------------------- detección de roles

@dataclass
class Deteccion:
    roles: dict                    # rol -> columna original (o None)
    confianza: dict                # rol -> 0..1
    exogenas: list                 # columnas originales sugeridas como exógenas extra
    advertencias: list = field(default_factory=list)


def _es_binaria(s: pd.Series) -> bool:
    v = pd.to_numeric(s, errors="coerce").dropna().unique()
    return len(v) > 0 and set(np.round(v, 6)).issubset({0, 1})


def _score_entidad_estructura(s: pd.Series, n: int) -> float:
    u = s.nunique(dropna=True)
    if u < 2 or u > max(50_000, n // 3):
        return 0.0
    rep = n / u
    if rep < 5:
        return 0.0
    score = min(1.0, rep / 50)
    if pd.api.types.is_float_dtype(s):
        score *= 0.1
    elif pd.api.types.is_integer_dtype(s):
        score *= 0.5
    return score


def detectar_roles(df: pd.DataFrame) -> Deteccion:
    cols = list(df.columns)
    n = len(df)
    roles, conf, adv = {}, {}, []
    usadas = set()

    # 1) fecha: nombre + qué fracción se puede parsear
    mejor, mejor_s = None, 0.0
    for c in cols:
        f, _ = parsear_fechas(df[c].head(2000))
        tasa = f.notna().mean()
        if tasa < 0.8:
            continue
        s = 0.6 * tasa + 0.4 * _score_nombre(c, ROLES["fecha"]["claves"])
        if pd.api.types.is_numeric_dtype(df[c]) and _score_nombre(c, ROLES["fecha"]["claves"]) == 0:
            s *= 0.3
        if s > mejor_s:
            mejor, mejor_s = c, s
    roles["fecha"], conf["fecha"] = mejor, round(mejor_s, 2)
    if mejor is None:
        adv.append("No se encontró una columna de fecha. Elígela manualmente.")
    usadas.add(mejor)

    numericas = [c for c in cols if c not in usadas and pd.api.types.is_numeric_dtype(df[c])]

    # 2) entidad: nombre + estructura (se repite, pocas categorías)
    mejor, mejor_s = None, 0.0
    for c in cols:
        if c in usadas:
            continue
        est = _score_entidad_estructura(df[c], n)
        if est == 0:
            continue
        s = 0.5 * _score_nombre(c, ROLES["entidad"]["claves"]) + 0.5 * est
        if s > mejor_s:
            mejor, mejor_s = c, s
    if mejor is not None and mejor_s >= 0.2:
        roles["entidad"], conf["entidad"] = mejor, round(mejor_s, 2)
        usadas.add(mejor)
    else:
        roles["entidad"], conf["entidad"] = None, 0.0

    # 3) roles opcionales por nombre + validación de tipo
    def validar(rol, s):
        x = pd.to_numeric(s, errors="coerce")
        if x.notna().mean() < 0.9:
            return False
        if rol in ("promocion", "quiebre"):
            return _es_binaria(s)
        if rol == "lead_time":
            return x.min() >= 0 and x.max() <= 730
        if rol in ("precio", "inventario", "costo_unitario"):
            return x.min() >= 0 and not _es_binaria(s)
        return True

    for rol in ("promocion", "quiebre", "lead_time", "inventario", "costo_unitario", "precio"):
        mejor, mejor_s = None, 0.0
        for c in numericas:
            if c in usadas:
                continue
            s = _score_nombre(c, ROLES[rol]["claves"])
            if s >= 0.6 and validar(rol, df[c]) and s > mejor_s:
                mejor, mejor_s = c, s
        roles[rol], conf[rol] = mejor, round(mejor_s, 2)
        if mejor:
            usadas.add(mejor)

    # 4) objetivo: nombre; si no hay, la numérica continua con más variación
    mejor, mejor_s = None, 0.0
    for c in numericas:
        if c in usadas or _es_binaria(df[c]):
            continue
        s = _score_nombre(c, ROLES["objetivo"]["claves"])
        if s > mejor_s:
            mejor, mejor_s = c, s
    if mejor is None:
        restantes = [c for c in numericas if c not in usadas and not _es_binaria(df[c]) and df[c].nunique() > 10]
        if restantes:
            mejor = max(restantes, key=lambda c: df[c].std() / (abs(df[c].mean()) + 1e-9))
            mejor_s = 0.3
            adv.append(f"La variable a pronosticar se eligió por descarte ('{mejor}'). Revísala.")
    roles["objetivo"], conf["objetivo"] = mejor, round(mejor_s, 2)
    if mejor is None:
        adv.append("No se encontró ninguna columna numérica que se pueda pronosticar.")
    usadas.add(mejor)

    # 5) exógenas extra: numéricas que no tomaron rol y que varían en el tiempo
    exogenas = [c for c in numericas if c not in usadas and df[c].nunique() > 1]
    # descartar columnas derivadas de la fecha (año, mes, día de semana)
    derivadas = {"ano", "anio", "year", "mes", "month", "dia", "day", "dia_semana", "weekday", "dayofweek", "semana", "week"}
    exogenas = [c for c in exogenas if _normalizar(c) not in derivadas]
    # descartar las que no cambian en el tiempo dentro de ninguna entidad (atributos fijos del producto)
    if roles.get("entidad"):
        exogenas = [c for c in exogenas if df.groupby(roles["entidad"])[c].nunique().max() > 1]

    return Deteccion(roles=roles, confianza=conf, exogenas=exogenas, advertencias=adv)


# ---------------------------------------------------------------- frecuencia

FRECUENCIAS = {
    "D": dict(nombre="diaria", unidad="día", unidad_pl="días", periodo_estacional=7, pandas="D", dias=1),
    "W": dict(nombre="semanal", unidad="semana", unidad_pl="semanas", periodo_estacional=52, pandas="W-MON", dias=7),
    "M": dict(nombre="mensual", unidad="mes", unidad_pl="meses", periodo_estacional=12, pandas="MS", dias=30.4),
    "Q": dict(nombre="trimestral", unidad="trimestre", unidad_pl="trimestres", periodo_estacional=4, pandas="QS", dias=91.3),
}


def detectar_frecuencia(fechas: pd.Series, entidades: pd.Series | None) -> tuple[str, dict]:
    """Devuelve (código, info). info trae la mediana de días entre registros y si hay
    varias filas por entidad y fecha (datos transaccionales que hay que agregar)."""
    d = pd.DataFrame({"f": fechas, "e": entidades if entidades is not None else "_"}).dropna()
    duplicadas = d.duplicated(["e", "f"]).mean()
    unicas = d.drop_duplicates(["e", "f"]).sort_values(["e", "f"])
    difs = unicas.groupby("e")["f"].diff().dt.days.dropna()
    mediana = float(difs.median()) if len(difs) else np.nan
    if np.isnan(mediana):
        codigo = "D"
    elif mediana <= 1.5:
        codigo = "D"
    elif mediana <= 8:
        codigo = "W"
    elif mediana <= 32:
        codigo = "M"
    else:
        codigo = "Q"
    return codigo, dict(mediana_dias=mediana, frac_duplicadas=float(duplicadas))


# ---------------------------------------------------------------- preparación

@dataclass
class Configuracion:
    roles: dict              # rol -> columna original o None
    exogenas: list           # columnas originales extra que entran al modelo
    frecuencia: str          # D / W / M / Q
    relleno_objetivo: str = "interpolar"   # "interpolar" o "cero"
    negativos_a_cero: bool = True


@dataclass
class DatasetPreparado:
    df: pd.DataFrame          # columnas: fecha, entidad, objetivo, [roles opcionales], [exógenas]
    config: Configuracion
    etiquetas: dict           # columna canónica -> nombre original (para mostrar)
    variables_modelo: list    # objetivo + exógenas que entran al modelo, en orden
    reporte: list             # acciones de limpieza realizadas (texto)
    n_filas_original: int

    @property
    def entidades(self) -> list:
        return sorted(self.df["entidad"].unique().tolist())

    @property
    def freq_info(self) -> dict:
        return FRECUENCIAS[self.config.frecuencia]

    def tiene(self, rol: str) -> bool:
        return rol in self.df.columns


AGREGACION = {
    "objetivo": "sum", "precio": "mean", "promocion": "max", "lead_time": "mean",
    "inventario": "last", "quiebre": "max", "costo_unitario": "mean",
}


def preparar(df_original: pd.DataFrame, config: Configuracion) -> DatasetPreparado:
    roles = config.roles
    if not roles.get("fecha") or not roles.get("objetivo"):
        raise ValueError("Faltan columnas obligatorias: se necesita una fecha y una variable a pronosticar.")

    reporte = []
    n0 = len(df_original)
    renombre = {orig: rol for rol, orig in roles.items() if orig}
    exog = [c for c in config.exogenas if c not in renombre]
    etiquetas = {rol: orig for rol, orig in roles.items() if orig}
    etiquetas.update({c: c for c in exog})

    df = df_original[list(renombre) + exog].rename(columns=renombre).copy()
    if "entidad" not in df.columns:
        df["entidad"] = "Serie única"
        etiquetas["entidad"] = "(una sola serie)"
    df["entidad"] = df["entidad"].astype(str).str.strip()

    # fechas
    df["fecha"], nota = parsear_fechas(df["fecha"])
    malas = df["fecha"].isna().sum()
    if malas:
        reporte.append(f"Se descartaron {malas:,} filas con fecha ilegible.".replace(",", "."))
        df = df[df["fecha"].notna()]
    reporte.append(f"Fechas interpretadas como {nota}.")

    # numéricos
    numericas = [c for c in df.columns if c not in ("fecha", "entidad")]
    for c in numericas:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    sin_obj = df["objetivo"].isna().sum()
    if sin_obj:
        reporte.append(f"{sin_obj:,} filas sin valor en la variable a pronosticar se tratarán como faltantes.".replace(",", "."))

    # alinear fecha al inicio del período
    fq = FRECUENCIAS[config.frecuencia]["pandas"]
    if config.frecuencia == "D":
        df["fecha"] = df["fecha"].dt.normalize()
    else:
        df["fecha"] = df["fecha"].dt.to_period(fq.split("-")[0] if config.frecuencia == "W" else fq[0]).dt.start_time

    # agregar duplicados (datos transaccionales o frecuencia más fina)
    n_antes = len(df)
    agg = {c: AGREGACION.get(c, "mean") for c in numericas}
    df = df.sort_values(["entidad", "fecha"]).groupby(["entidad", "fecha"], as_index=False).agg(agg)
    if len(df) < n_antes:
        reporte.append(
            f"Se agruparon {n_antes - len(df):,} filas repetidas por entidad y {FRECUENCIAS[config.frecuencia]['unidad']} "
            f"(la demanda se sumó; precio y costo se promediaron).".replace(",", "."))

    # regularizar: una fila por período, sin huecos
    partes, huecos_total = [], 0
    for ent, g in df.groupby("entidad", sort=True):
        rango = pd.date_range(g["fecha"].min(), g["fecha"].max(), freq=fq)
        g = g.set_index("fecha").reindex(rango)
        g.index.name = "fecha"
        huecos = int(g["objetivo"].isna().sum())
        huecos_total += huecos
        if config.relleno_objetivo == "cero":
            g["objetivo"] = g["objetivo"].fillna(0)
        else:
            g["objetivo"] = g["objetivo"].interpolate(limit_direction="both")
        for c in numericas:
            if c == "objetivo":
                continue
            if c in ("promocion", "quiebre"):
                g[c] = g[c].fillna(0)
            else:
                g[c] = g[c].ffill().bfill()
        g["entidad"] = ent
        partes.append(g.reset_index())
    df = pd.concat(partes, ignore_index=True)
    if huecos_total:
        metodo = "con cero (sin ventas ese período)" if config.relleno_objetivo == "cero" else "por interpolación"
        reporte.append(f"Se completaron {huecos_total:,} períodos faltantes {metodo}.".replace(",", "."))

    # negativos
    neg = int((df["objetivo"] < 0).sum())
    if neg and config.negativos_a_cero:
        df["objetivo"] = df["objetivo"].clip(lower=0)
        reporte.append(f"{neg:,} valores negativos (devoluciones) se llevaron a cero.".replace(",", "."))

    # exógenas constantes no aportan al modelo
    variables_modelo = ["objetivo"]
    for c in [r for r in ROLES_MODELO if r in df.columns] + exog:
        if df.groupby("entidad")[c].nunique().max() > 1:
            variables_modelo.append(c)
        else:
            reporte.append(f"'{etiquetas.get(c, c)}' no varía en el tiempo dentro de ninguna entidad y no se usará en el modelo.")
    columnas = ["fecha", "entidad"] + [c for c in df.columns if c not in ("fecha", "entidad")]
    df = df[columnas].sort_values(["entidad", "fecha"]).reset_index(drop=True)

    return DatasetPreparado(df=df, config=config, etiquetas=etiquetas, variables_modelo=variables_modelo,
                            reporte=reporte, n_filas_original=n0)


# ---------------------------------------------------------------- diagnóstico

def resumen_por_entidad(dp: DatasetPreparado) -> pd.DataFrame:
    g = dp.df.groupby("entidad")
    r = pd.DataFrame({
        "registros": g.size(),
        "desde": g["fecha"].min(),
        "hasta": g["fecha"].max(),
        "promedio": g["objetivo"].mean(),
        "cv": g["objetivo"].std() / g["objetivo"].mean().replace(0, np.nan),
        "pct_ceros": g["objetivo"].apply(lambda s: (s == 0).mean() * 100),
    }).reset_index()
    return r

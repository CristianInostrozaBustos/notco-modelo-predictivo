"""Reglas de entrenamiento (triggers) e indicadores disponibles.

Las reglas son funciones puras: reciben el dataset preparado y devuelven un plan
con la configuración elegida y el porqué de cada decisión, para poder mostrarlo
en el sitio y explicarlo en la defensa.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .datos import FRECUENCIAS, DatasetPreparado

# ---------------------------------------------------------------- umbrales

UMBRAL_HISTORIA_COMPLETA = 300      # registros por entidad
MAX_ENTIDADES_CABEZAS = 50          # sobre esto, embedding compartido en vez de una cabeza por entidad

# Por frecuencia: ventanas candidatas que prueba la búsqueda, validación (modo completo, modo escaso),
# horizonte de pronóstico por defecto y máximo.
PARAMETROS_FRECUENCIA = {
    "D": dict(ventanas=[7, 14, 30, 60, 90], validacion=(84, 28), h_defecto=30, h_max=180),
    "W": dict(ventanas=[4, 8, 13, 26], validacion=(13, 8), h_defecto=8, h_max=26),
    "M": dict(ventanas=[3, 6, 12], validacion=(12, 6), h_defecto=6, h_max=12),
    "Q": dict(ventanas=[2, 4, 8], validacion=(4, 2), h_defecto=4, h_max=8),
}

ARQUITECTURAS = {
    "completa": dict(lstm=(64, 32), dense_tronco=16, dense_cabeza=16, dropout=0.2,
                     descripcion="LSTM 64 → LSTM 32 → Dense 16, dropout 0,2 (la configuración validada en el caso de estudio)"),
    "escasa": dict(lstm=(32, 16), dense_tronco=8, dense_cabeza=8, dropout=0.3,
                   descripcion="LSTM 32 → LSTM 16 → Dense 8, dropout 0,3 (red más chica para no sobreajustar con poca historia)"),
}

ESQUEMAS = {
    "simple": "Una sola serie: red LSTM directa, sin cabezas por entidad.",
    "cabezas": "Tronco LSTM compartido + una cabeza de salida independiente por entidad.",
    "embedding": "Tronco LSTM compartido + embedding de entidad como entrada (escala a miles de series).",
}


def minimo_registros(frecuencia: str) -> int:
    """Mínimo para entrenar: dos bloques de validación (selección y prueba) + tres ventanas mínimas."""
    p = PARAMETROS_FRECUENCIA[frecuencia]
    return 2 * p["validacion"][1] + 3 * min(p["ventanas"])


@dataclass
class PlanEntrenamiento:
    frecuencia: str
    modo: str                     # "completa" / "escasa"
    esquema: str                  # "simple" / "cabezas" / "embedding"
    ventanas: list                # candidatas que probará la búsqueda
    validacion: int               # largo de cada bloque de validación (selección y prueba)
    horizonte_max: int
    horizonte_defecto: int
    arquitectura: dict
    entidades_incluidas: list
    entidades_excluidas: list
    historia_min: int = 0
    razones: list = field(default_factory=list)
    viable: bool = True


def planificar(dp: DatasetPreparado) -> PlanEntrenamiento:
    f = dp.config.frecuencia
    fi = FRECUENCIAS[f]
    p = PARAMETROS_FRECUENCIA[f]
    conteos = dp.df.groupby("entidad").size()
    minimo = minimo_registros(f)
    incluidas = sorted(conteos[conteos >= minimo].index.tolist())
    excluidas = sorted(conteos[conteos < minimo].index.tolist())
    razones = [f"Frecuencia {fi['nombre']}: un registro por {fi['unidad']} y entidad."]

    if excluidas:
        razones.append(
            f"{len(excluidas)} entidad(es) tienen menos de {minimo} {fi['unidad_pl']} de historia, el mínimo para "
            f"entrenar, y quedan fuera del modelo.")
    if not incluidas:
        return PlanEntrenamiento(f, "escasa", "simple", [], 0, 0, 0, ARQUITECTURAS["escasa"], [], excluidas,
                                 razones=razones + ["Ninguna entidad alcanza el mínimo de historia: no se puede entrenar."],
                                 viable=False)

    historia_min = int(conteos[incluidas].min())
    if historia_min >= UMBRAL_HISTORIA_COMPLETA:
        modo = "completa"
        razones.append(
            f"Todas las entidades incluidas tienen al menos {UMBRAL_HISTORIA_COMPLETA} registros "
            f"(la más corta tiene {historia_min}): se usa la red completa.")
    else:
        modo = "escasa"
        razones.append(
            f"La entidad con menos historia tiene {historia_min} registros, bajo el umbral de "
            f"{UMBRAL_HISTORIA_COMPLETA}: se usa la red para datos escasos.")

    validacion = p["validacion"][0 if modo == "completa" else 1]
    tope = max(p["validacion"][1], int(historia_min * 0.15))
    if validacion > tope:
        razones.append(f"Bloques de validación reducidos de {validacion} a {tope} {fi['unidad_pl']} para no quitarle "
                       "demasiada historia al entrenamiento.")
        validacion = tope

    disponible = historia_min - 2 * validacion
    ventanas = [v for v in p["ventanas"] if disponible >= 3 * v]
    if not ventanas:
        ventanas = [min(p["ventanas"])]
    razones.append(
        "Ventanas que probará la búsqueda automática: " + ", ".join(str(v) for v in ventanas) + f" {fi['unidad_pl']}. "
        "Se queda con la de menor error en el bloque de selección.")
    razones.append(f"El error final se mide sobre los últimos {validacion} {fi['unidad_pl']}, que no se usan ni para "
                   "entrenar ni para elegir la ventana.")

    n = len(incluidas)
    if n == 1:
        esquema = "simple"
        razones.append("Hay una sola serie: no se necesitan cabezas por entidad.")
    elif n <= MAX_ENTIDADES_CABEZAS:
        esquema = "cabezas"
        razones.append(f"{n} entidades (hasta {MAX_ENTIDADES_CABEZAS}): tronco compartido y una cabeza por entidad, "
                       "para que la calibración de una no contamine a las demás.")
    else:
        esquema = "embedding"
        razones.append(f"{n} entidades (más de {MAX_ENTIDADES_CABEZAS}): tronco compartido con embedding de entidad, "
                       "porque una cabeza por entidad no escala en memoria.")

    h_max = max(1, min(p["h_max"], historia_min // 2))
    h_def = min(p["h_defecto"], h_max)
    return PlanEntrenamiento(f, modo, esquema, ventanas, validacion, h_max, h_def, ARQUITECTURAS[modo],
                             incluidas, excluidas, historia_min, razones)


def clase_entidad(n_registros: int, frecuencia: str) -> str:
    if n_registros < minimo_registros(frecuencia):
        return "Insuficiente"
    return "Completa" if n_registros >= UMBRAL_HISTORIA_COMPLETA else "Escasa"


# ---------------------------------------------------------------- indicadores

@dataclass
class Indicador:
    nombre: str
    seccion: str
    estado: str          # "disponible" / "manual" / "no disponible"
    detalle: str


def indicadores(dp: DatasetPreparado) -> list[Indicador]:
    t = dp.tiene
    n_ent = dp.df["entidad"].nunique()
    exog = [v for v in dp.variables_modelo if v != "objetivo"]
    L = []

    def add(nombre, seccion, estado, detalle):
        L.append(Indicador(nombre, seccion, estado, detalle))

    # Exploración
    add("Demanda por entidad y estacionalidad", "Exploración", "disponible", "Solo requiere fecha y variable a pronosticar.")
    if t("precio") and n_ent > 1:
        add("Participación ABC en ingresos", "Exploración", "disponible", "Ingreso = demanda × precio.")
    elif n_ent > 1:
        add("Participación ABC en volumen", "Exploración", "disponible", "Sin precio, la clasificación ABC se hace por unidades.")
    else:
        add("Participación ABC", "Exploración", "no disponible", "Requiere más de una entidad.")
    if t("quiebre"):
        add("Tasa de quiebre de stock", "Exploración", "disponible", "Desde la columna de quiebre.")
    elif t("inventario"):
        add("Tasa de quiebre de stock", "Exploración", "disponible", "Se estima como períodos con inventario en cero.")
    else:
        add("Tasa de quiebre de stock", "Exploración", "no disponible", "Requiere columna de quiebre o de inventario.")
    if t("promocion"):
        add("Efecto de las promociones en la demanda", "Exploración", "disponible", "Compara períodos con y sin promoción.")
    else:
        add("Efecto de las promociones en la demanda", "Exploración", "no disponible", "Requiere columna de promoción (0/1).")
    riesgo = [x for x in ("lead_time", "quiebre", "inventario") if t(x)]
    if n_ent > 1 and riesgo:
        add("Ranking de riesgo de abastecimiento", "Exploración", "disponible",
            "Combina variabilidad de la demanda con " + ", ".join(dp.etiquetas[x] for x in riesgo) + ".")
    elif n_ent > 1:
        add("Ranking de riesgo de abastecimiento", "Exploración", "disponible", "Solo con la variabilidad de la demanda (sin lead time ni quiebres).")
    else:
        add("Ranking de riesgo de abastecimiento", "Exploración", "no disponible", "Requiere más de una entidad.")

    # Modelo
    for nombre, det in [
        ("Pronóstico probabilístico P10 · P50 · P90", "Tres cuantiles por período."),
        ("MAPE y cobertura del rango P10–P90", "Sobre el período de validación que el modelo no vio."),
        ("Comparación con método ingenuo", "Contra repetir el valor de la misma temporada anterior."),
        ("Diagnóstico de cuantiles", "Cruce de cuantiles, captura de variabilidad y ancho de banda."),
        ("Prueba de retardo", "Revisa si el pronóstico solo copia el valor anterior."),
        ("Fallas de cobertura", "Períodos en que la demanda real se salió del rango."),
    ]:
        add(nombre, "Modelo", "disponible", det)
    if t("promocion"):
        add("Error en períodos con y sin promoción", "Modelo", "disponible", "MAPE separado por tipo de período.")
    else:
        add("Error en períodos con y sin promoción", "Modelo", "no disponible", "Requiere columna de promoción.")

    # Política
    if t("lead_time"):
        add("Política ROP / SS / meta T", "Política", "disponible", "Lead time tomado del dataset.")
    else:
        add("Política ROP / SS / meta T", "Política", "manual", "Falta el lead time: se ingresa a mano.")
    if t("inventario"):
        add("Simulación de inventario mes a mes", "Política", "disponible", "Parte del último inventario registrado.")
    else:
        add("Simulación de inventario mes a mes", "Política", "manual", "Falta el inventario inicial: se ingresa a mano.")
    if exog:
        add("Escenarios what-if", "Política", "disponible",
            "Variables que se pueden estresar: " + ", ".join(dp.etiquetas.get(v, v) for v in exog) + ".")
    else:
        add("Escenarios what-if", "Política", "no disponible", "Requiere al menos una variable exógena (precio, promoción u otra).")
    if t("costo_unitario"):
        add("Comparación de costo entre proveedores", "Política", "manual",
            "Costo del proveedor actual desde el dataset; el alternativo se ingresa a mano.")
    else:
        add("Comparación de costo entre proveedores", "Política", "manual", "Ambos costos se ingresan a mano.")
    return L


def tabla_indicadores(lista: list[Indicador]) -> pd.DataFrame:
    return pd.DataFrame([vars(i) for i in lista])

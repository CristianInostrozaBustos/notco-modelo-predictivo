"""Del pronóstico a la decisión de abastecimiento.

Sistema de revisión periódica con nivel meta (T) y punto de reorden (ROP) como alarma,
calculado con la demanda y la incertidumbre del pronóstico (no con el histórico):

    SS  = Z · σ · √L
    ROP = d · L + SS
    T   = d · (L + P) + SS

d y σ se toman de los períodos futuros que cubre el pedido (L + P). σ sale del ancho del
rango P10–P90: en una normal, P90 − P10 = 2 · 1,2816 · σ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .datos import FRECUENCIAS

Z_NIVEL = {"80%": 0.842, "85%": 1.036, "90%": 1.282, "95%": 1.645, "97,5%": 1.960, "99%": 2.326}
Z_P10_P90 = 1.2816


@dataclass
class Parametros:
    lead_time_dias: float
    revision_dias: float
    nivel_servicio: str
    inventario_actual: float | None


@dataclass
class Decision:
    entidad: str
    d: float                 # demanda media por período en la cobertura L+P
    sigma: float             # desviación por período
    L: float                 # lead time en períodos
    P: float                 # revisión en períodos
    ss: float
    rop: float
    meta: float
    inventario: float | None
    estado: str              # "Pedir ahora" / "Riesgo de quiebre" / "Stock suficiente" / "Sin inventario"
    cantidad: float
    periodos_hasta_pedido: float | None
    cobertura_dias: float | None
    fecha_pedido: pd.Timestamp | None
    aviso: str = ""
    z: float = 1.282


def decidir(entidad: str, pron: pd.DataFrame, freq: str, par: Parametros) -> Decision:
    dias = FRECUENCIAS[freq]["dias"]
    L = max(par.lead_time_dias / dias, 1e-6)
    P = max(par.revision_dias / dias, 1e-6)
    cubre = int(np.ceil(L + P))
    ventana = pron.iloc[:max(1, min(cubre, len(pron)))]
    aviso = ""
    if cubre > len(pron):
        aviso = (f"El lead time más la revisión cubren {cubre} {FRECUENCIAS[freq]['unidad_pl']} y el pronóstico llega a "
                 f"{len(pron)}. Se usó el promedio disponible; alarga el horizonte para mayor precisión.")
    d = float(ventana["P50"].mean())
    sigma = float(((ventana["P90"] - ventana["P10"]) / (2 * Z_P10_P90)).clip(lower=0).mean())
    z = Z_NIVEL[par.nivel_servicio]
    ss = z * sigma * np.sqrt(L)
    rop = d * L + ss
    meta = d * (L + P) + ss

    I = par.inventario_actual
    if I is None or np.isnan(I):
        return Decision(entidad, d, sigma, L, P, ss, rop, meta, None, "Sin inventario", 0.0, None, None, None, aviso, z)

    cobertura = I / d * dias if d > 0 else np.inf
    if I <= ss:
        estado, cantidad, hasta, fecha = "Riesgo de quiebre", max(0.0, meta - I), 0.0, pron["fecha"].iloc[0]
    elif I <= rop:
        estado, cantidad, hasta, fecha = "Pedir ahora", max(0.0, meta - I), 0.0, pron["fecha"].iloc[0]
    else:
        proy = I - pron["P50"].cumsum().to_numpy()
        cruce = np.where(proy <= rop)[0]
        if len(cruce):
            k = int(cruce[0])
            hasta, fecha = float(k + 1), pron["fecha"].iloc[k]
            cantidad = max(0.0, meta - proy[k])
        else:
            hasta, fecha, cantidad = None, None, 0.0
        estado = "Stock suficiente"
    return Decision(entidad, d, sigma, L, P, ss, rop, meta, I, estado, cantidad, hasta, cobertura, fecha, aviso, z)


def politica_dinamica(pron: pd.DataFrame, L, P: float, z: float):
    """ROP, meta y SS recalculados en cada período con el pronóstico de los períodos que vienen.

    L puede ser un escalar o un arreglo (lead time en períodos del pedido emitido en cada período).
    Devuelve tres arreglos del largo del pronóstico.
    """
    n = len(pron)
    L = np.broadcast_to(np.asarray(L, float), (n,))
    p50 = pron["P50"].to_numpy(float)
    sig = ((pron["P90"] - pron["P10"]).to_numpy(float) / (2 * Z_P10_P90)).clip(min=0)
    rop, meta, ss = np.zeros(n), np.zeros(n), np.zeros(n)
    for t in range(n):
        w = max(1, int(np.ceil(L[t] + P)))
        sl = slice(t, min(n, t + w)) if t < n else slice(n - 1, n)
        d, sg = p50[sl].mean(), sig[sl].mean()
        ss[t] = z * sg * np.sqrt(L[t])
        rop[t] = d * L[t] + ss[t]
        meta[t] = d * (L[t] + P) + ss[t]
    return rop, meta, ss


def simular(pron: pd.DataFrame, dec: Decision) -> pd.DataFrame:
    """Inventario proyectado aplicando la política sobre el P50, recalculada período a período.

    Cada vez que la posición de inventario (disponible + en tránsito) cae bajo el ROP vigente se pide
    hasta la meta vigente; el pedido llega L períodos después. Venta perdida: el inventario no baja de cero.
    """
    if dec.inventario is None:
        return pd.DataFrame()
    from .escenarios import simular as simular_base
    rop, meta, ss = politica_dinamica(pron, dec.L, dec.P, dec.z)
    sim = simular_base(pron["fecha"].to_numpy(), pron["P50"].to_numpy(), dec.inventario, rop, meta, dec.L)
    sim["ss"] = ss
    sim["meta"] = meta
    return sim


def lead_time_dataset(serie_lead: pd.Series) -> float:
    """Lead time representativo: promedio de los últimos 90 registros."""
    return float(serie_lead.dropna().tail(90).mean())

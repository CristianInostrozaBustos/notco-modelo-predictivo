"""Escenarios what-if y simulación de inventario.

Un escenario es un conjunto de eventos que ocurren durante una ventana de tiempo dentro
del horizonte. Hay dos tipos de efecto:

- Por el modelo: cambios en variables que el modelo usa como entrada (precio, promoción u
  otras exógenas). El pronóstico se recalcula completo con esos valores, así que el efecto
  en la demanda es el que el modelo aprendió de la historia.
- Directos: cambios que no pasan por el modelo. Un shock de demanda (+/- %) se aplica sobre
  el pronóstico, y un retraso del proveedor alarga el lead time de los pedidos emitidos
  durante el evento.

Para cada entidad se comparan tres mundos sobre el mismo horizonte:

    base            demanda base,      política actual
    sin ajustar     demanda escenario, política actual      -> lo que pasaría si no reaccionas
    ajustada        demanda escenario, política recalculada -> lo que pasaría si reaccionas a tiempo
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .datos import FRECUENCIAS
from .politica import Z_NIVEL, Decision, Parametros, decidir, politica_dinamica

TIPOS = {
    "demanda": "Sube o baja la demanda",
    "retraso": "Retraso del proveedor",
    "precio": "Cambio de precio",
    "promocion": "Promoción",
    "exogena": "Cambio en otra variable",
}


@dataclass
class Evento:
    tipo: str              # clave de TIPOS
    valor: float           # % (como fracción) para demanda/precio/exógena; días para retraso; ignorado en promoción
    var: str | None = None  # variable exógena (solo tipo "exogena")
    elasticidad: float | None = None  # solo precio: si se da, el efecto se calcula con esta elasticidad y no con el modelo


@dataclass
class Escenario:
    eventos: list
    desde: int             # índice del primer período afectado dentro del horizonte
    duracion: int          # períodos

    @property
    def hasta(self):
        return self.desde + self.duracion

    def cambios_modelo(self) -> list:
        c = []
        for ev in self.eventos:
            if ev.tipo == "precio" and ev.elasticidad is None:
                c.append(dict(var="precio", desde=self.desde, hasta=self.hasta, tipo="pct", valor=ev.valor))
            elif ev.tipo == "promocion":
                c.append(dict(var="promocion", desde=self.desde, hasta=self.hasta, tipo="fijar", valor=1.0))
            elif ev.tipo == "exogena" and ev.var:
                c.append(dict(var=ev.var, desde=self.desde, hasta=self.hasta, tipo="pct", valor=ev.valor))
        return c

    def shock_demanda(self) -> float:
        k = sum(ev.valor for ev in self.eventos if ev.tipo == "demanda")
        k += sum(ev.elasticidad * ev.valor for ev in self.eventos if ev.tipo == "precio" and ev.elasticidad is not None)
        return k

    def retraso_dias(self) -> float:
        return sum(ev.valor for ev in self.eventos if ev.tipo == "retraso")

    def clave(self) -> tuple:
        return (tuple((e.tipo, round(e.valor, 6), e.var, e.elasticidad) for e in self.eventos), self.desde, self.duracion)


def aplicar_shock(pron: pd.DataFrame, esc: Escenario) -> pd.DataFrame:
    out = pron.copy()
    k = esc.shock_demanda()
    if k:
        sl = slice(esc.desde, esc.hasta)
        for q in ("P10", "P50", "P90"):
            out.loc[out.index[sl], q] = out[q].iloc[sl] * (1 + k)
    return out


def lead_time_por_periodo(lt_dias: float, esc: Escenario, n: int, freq: str) -> np.ndarray:
    """Lead time (en períodos) que tendría un pedido emitido en cada período."""
    dias = FRECUENCIAS[freq]["dias"]
    lt = np.full(n, lt_dias / dias)
    lt[esc.desde:esc.hasta] += esc.retraso_dias() / dias
    return lt


# ---------------------------------------------------------------- simulación

def simular(fechas, demanda, inventario0, rop, meta, lt_periodos) -> pd.DataFrame:
    """Simulación período a período con venta perdida (el inventario no baja de cero).

    rop, meta y lt_periodos pueden ser escalares o arreglos por período. Cuando la posición
    de inventario (disponible + en tránsito) llega al ROP se pide hasta la meta; el pedido
    llega después del lead time vigente al momento de pedir.
    """
    n = len(demanda)
    rop = np.broadcast_to(np.asarray(rop, float), (n,))
    meta = np.broadcast_to(np.asarray(meta, float), (n,))
    lt = np.broadcast_to(np.asarray(lt_periodos, float), (n,))
    I = float(inventario0)
    transito = []
    filas = []
    for t in range(n):
        llega = sum(q for k, q in transito if k == t)
        transito = [(k, q) for k, q in transito if k != t]
        I += llega
        d = float(demanda[t])
        atendida = min(I, d)
        perdida = d - atendida
        I -= atendida
        posicion = I + sum(q for _, q in transito)
        pedido = 0.0
        if posicion <= rop[t]:
            pedido = max(0.0, meta[t] - posicion)
            if pedido > 0:
                transito.append((t + max(1, int(np.ceil(lt[t]))), pedido))
        filas.append(dict(fecha=fechas[t], demanda=d, inventario=I, llegada=llega, pedido=pedido,
                          no_atendida=perdida, quiebre=perdida > 1e-9, rop=rop[t]))
    return pd.DataFrame(filas)


@dataclass
class Comparacion:
    entidad: str
    base: pd.DataFrame            # pronóstico base
    escenario: pd.DataFrame       # pronóstico con el escenario
    dec_base: Decision            # política estática (para cuando no hay inventario)
    dec_esc: Decision
    sim_base: pd.DataFrame | None
    sim_sin_ajuste: pd.DataFrame | None
    sim_ajustada: pd.DataFrame | None
    inicio_ajuste: int = 0
    resumen: dict = field(default_factory=dict)


def _anticipar(L, k):
    """Lead time que planifica quien sabe lo que viene: el máximo de los próximos k períodos."""
    n = len(L)
    return np.array([L[t:min(n, t + k)].max() for t in range(n)])


def comparar(entidad, pron_base, pron_esc_modelo, par: Parametros, esc: Escenario, freq) -> Comparacion:
    """Compara tres mundos sobre el mismo horizonte, con la política recalculada período a período:

    base         demanda sin evento, lead time normal, política con el pronóstico base
    sin ajustar  demanda y lead time del evento, pero la política sigue usando el pronóstico base
    ajustada     demanda y lead time del evento, y la política usa el pronóstico del escenario y
                 anticipa el mayor lead time (planifica con el lead time que tendrá el próximo pedido)
    """
    pron_esc = aplicar_shock(pron_esc_modelo, esc)
    n = len(pron_base)
    dias = FRECUENCIAS[freq]["dias"]
    z = Z_NIVEL[par.nivel_servicio]
    P_per = max(par.revision_dias / dias, 1e-6)

    dec_base = decidir(entidad, pron_base, freq, par)
    par_esc = Parametros(lead_time_dias=par.lead_time_dias + esc.retraso_dias(), revision_dias=par.revision_dias,
                         nivel_servicio=par.nivel_servicio, inventario_actual=par.inventario_actual)
    desde_evento = pron_esc.iloc[esc.desde:].reset_index(drop=True)
    dec_esc = decidir(entidad, desde_evento if len(desde_evento) else pron_esc, freq, par_esc)

    lt_base = lead_time_por_periodo(par.lead_time_dias, Escenario([], 0, 0), n, freq)
    lt_esc = lead_time_por_periodo(par.lead_time_dias, esc, n, freq)
    k = int(np.ceil(lt_esc.max())) + 1
    lt_plan = _anticipar(lt_esc, k)

    rop_b, meta_b, ss_b = politica_dinamica(pron_base, lt_base, P_per, z)
    rop_e, meta_e, ss_e = politica_dinamica(pron_esc, lt_plan, P_per, z)
    difiere = np.where(np.abs(rop_e - rop_b) > 0.01 * np.maximum(rop_b, 1e-9))[0]
    inicio_ajuste = int(difiere[0]) if len(difiere) else esc.desde

    t_ev = min(esc.desde, n - 1)
    resumen = {
        "rop_base": float(rop_b[t_ev]), "rop_esc": float(rop_e[t_ev]),
        "ss_base": float(ss_b[t_ev]), "ss_esc": float(ss_e[t_ev]),
        "meta_base": float(meta_b[t_ev]), "meta_esc": float(meta_e[t_ev]),
    }
    sims = (None, None, None)
    if par.inventario_actual is not None and not np.isnan(par.inventario_actual):
        I0 = par.inventario_actual
        f = pron_base["fecha"].to_numpy()
        s_base = simular(f, pron_base["P50"].to_numpy(), I0, rop_b, meta_b, lt_base)
        s_sin = simular(f, pron_esc["P50"].to_numpy(), I0, rop_b, meta_b, lt_esc)
        s_aj = simular(f, pron_esc["P50"].to_numpy(), I0, rop_e, meta_e, lt_esc)
        sims = (s_base, s_sin, s_aj)
        for nombre, s in (("base", s_base), ("sin_ajuste", s_sin), ("ajustada", s_aj)):
            resumen[f"quiebre_{nombre}"] = int(s["quiebre"].sum())
            resumen[f"perdida_{nombre}"] = float(s["no_atendida"].sum())
            resumen[f"inv_prom_{nombre}"] = float(s["inventario"].mean())
            resumen[f"pedidos_{nombre}"] = int((s["pedido"] > 0).sum())
        # lo que agrega el evento por sobre lo que ya pasaría sin él
        # se compara el total (no día a día): un cambio de demanda puede correr unos días los quiebres
        # que ya existían sin que eso sea un efecto del evento
        resumen["quiebre_extra_sin"] = max(0, resumen["quiebre_sin_ajuste"] - resumen["quiebre_base"])
        resumen["quiebre_extra_aj"] = max(0, resumen["quiebre_ajustada"] - resumen["quiebre_base"])
        resumen["perdida_extra_sin"] = max(0.0, resumen["perdida_sin_ajuste"] - resumen["perdida_base"])
        resumen["perdida_extra_aj"] = max(0.0, resumen["perdida_ajustada"] - resumen["perdida_base"])
        # diferencias mínimas (menos de 0,5% de la demanda del horizonte) son ruido de la simulación discreta:
        # los pedidos se corren un día y un quiebre que ya existía cae en otra fecha
        tolerancia = 0.005 * float(pron_base["P50"].sum())
        for k in ("sin", "aj"):
            if resumen[f"perdida_extra_{k}"] < tolerancia:
                resumen[f"quiebre_extra_{k}"], resumen[f"perdida_extra_{k}"] = 0, 0.0
    resumen["demanda_base"] = float(pron_base["P50"].sum())
    resumen["demanda_esc"] = float(pron_esc["P50"].sum())
    ev = slice(esc.desde, esc.hasta)
    resumen["demanda_base_evento"] = float(pron_base["P50"].iloc[ev].sum())
    resumen["demanda_esc_evento"] = float(pron_esc["P50"].iloc[ev].sum())
    resumen["dias_por_periodo"] = dias
    return Comparacion(entidad, pron_base, pron_esc, dec_base, dec_esc, *sims, inicio_ajuste=inicio_ajuste,
                       resumen=resumen)

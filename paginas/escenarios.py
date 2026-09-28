import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import estilo as E
import sesion as S
from motor import escenarios as X
from motor import politica as P
from motor.datos import FRECUENCIAS

dp, res = S.requiere_pronostico()
S.panel_dataset()
fi = dp.freq_info
freq = dp.config.frecuencia
H = S.horizonte()
nom = S.nombre_entidad(dp).capitalize()
entidades = list(res.series)
u, u_pl = fi["unidad"], fi["unidad_pl"]

E.encabezado(
    "Paso 4",
    "Escenarios: ¿qué pasa si…?",
    "Simula un evento y mira cómo cambia la demanda, qué le pasa a tu inventario si no reaccionas "
    "y qué tendrías que cambiar para evitar quedarte sin stock.",
)

import cuenta  # noqa: E402

registro = S.registro_actual()
guardados = registro.get("escenarios", []) if registro else []
if guardados:
    nombres = [g["nombre"] for g in guardados]
    elegido = st.pills("Tus escenarios guardados", nombres, key="esc_guardado_sel")
    if elegido and st.session_state.get("_esc_cargado") != elegido:
        g = guardados[nombres.index(elegido)]
        st.session_state["escenario"] = dict(clave=S.clave_dataset(dp), desde=g["desde"], duracion=g["duracion"],
                                             alcance=g["alcance"] if g["alcance"] in (["Todas"] + list(res.series)) else "Todas",
                                             eventos=[X.Evento(**ev) for ev in g["eventos"]])
        st.session_state["_esc_cargado"] = elegido

# ---------------------------------------------------------------- armar el escenario
disponibles = ["demanda", "retraso"]
if "precio" in res.exog_nombres:
    disponibles.append("precio")
if "promocion" in res.exog_nombres:
    disponibles.append("promocion")
otras = [v for v in res.exog_nombres if v not in ("precio", "promocion")]
if otras:
    disponibles.append("exogena")

primera_fecha = res.series[entidades[0]].fechas[-1] + pd.tseries.frequencies.to_offset(FRECUENCIAS[freq]["pandas"])
dur_def = {"D": 30, "W": 6, "M": 3, "Q": 2}[freq]
dur_max = max(1, min(res.plan.horizonte_max - 1, {"D": 120, "W": 26, "M": 12, "Q": 4}[freq]))

with st.container(border=True):
    st.markdown("**¿Qué ocurre?** · puedes combinar varios eventos")
    tipos = st.pills("Eventos", disponibles, format_func=lambda t: X.TIPOS[t], selection_mode="multi",
                     default=["retraso"], label_visibility="collapsed", key="esc_tipos")
    eventos = []
    if tipos:
        cols = st.columns(len(tipos))
        for col, t in zip(cols, tipos):
            with col:
                if t == "demanda":
                    v = st.slider("Cambio en la demanda (%)", -60, 150, 25, 5, key="esc_dem",
                                  help="Por ejemplo, un cliente nuevo (+) o un competidor que entra (−).")
                    eventos.append(X.Evento("demanda", v / 100))
                elif t == "retraso":
                    v = st.number_input("Días adicionales de retraso", 1, 180, 10, key="esc_ret",
                                        help="Los pedidos que emitas durante el evento llegan con este retraso extra.")
                    eventos.append(X.Evento("retraso", float(v)))
                elif t == "precio":
                    v = st.slider("Cambio de precio (%)", -50, 50, 15, 5, key="esc_pre")
                    modo = st.radio("Cómo estimar el efecto en la demanda", ["Elasticidad", "Lo que aprendió el modelo"],
                                    horizontal=True, key="esc_pre_modo",
                                    help="Si en tu historial el precio casi nunca cambió por sí solo, el modelo no alcanza "
                                         "a aprender cuánto reacciona la demanda. En ese caso conviene una elasticidad conocida.")
                    if modo == "Elasticidad":
                        el = st.number_input("Elasticidad precio-demanda", -3.0, 0.0, -0.3, 0.05, format="%.2f",
                                             key="esc_pre_el",
                                             help="Cambio % de la demanda por cada 1% de cambio en el precio. −0,3 = si el "
                                                  "precio sube 10%, la demanda baja 3%.")
                        eventos.append(X.Evento("precio", v / 100, elasticidad=float(el)))
                    else:
                        eventos.append(X.Evento("precio", v / 100))
                elif t == "promocion":
                    st.markdown("**Promoción**")
                    st.caption("Se activa una promoción durante todo el evento; el modelo estima su efecto.")
                    eventos.append(X.Evento("promocion", 1.0))
                elif t == "exogena":
                    var = st.selectbox("Variable", otras, format_func=lambda v: dp.etiquetas.get(v, v), key="esc_var")
                    v = st.slider("Cambio (%)", -60, 150, 30, 5, key="esc_exo")
                    eventos.append(X.Evento("exogena", v / 100, var))

    st.markdown("**¿Cuándo y a quién?**")
    c1, c2, c3 = st.columns(3)
    alcance = c1.selectbox("Afecta a", (["Todas"] if len(entidades) > 1 else []) + entidades, key="esc_alcance",
                           format_func=lambda x: f"Todas las {S.nombre_entidad(dp, True)}" if x == "Todas" else x)
    adelanto = {"D": pd.Timedelta(days=14), "W": pd.Timedelta(weeks=2), "M": pd.DateOffset(months=1), "Q": pd.DateOffset(months=3)}[freq]
    fecha_ini = c2.date_input("Empieza", (primera_fecha + adelanto).date(), min_value=primera_fecha.date(),
                              format="DD/MM/YYYY", key="esc_ini")
    duracion = c3.slider(f"Dura ({u_pl})", 1, dur_max, min(dur_def, dur_max), key="esc_dur")
    simular = st.button("Simular escenario", type="primary", icon=":material/play_arrow:", disabled=not eventos)

if simular:
    st.session_state.pop("_esc_cargado", None)
    offset = pd.tseries.frequencies.to_offset(FRECUENCIAS[freq]["pandas"])
    desde = len(pd.date_range(primera_fecha, pd.Timestamp(fecha_ini), freq=offset)) - 1
    st.session_state["escenario"] = dict(clave=S.clave_dataset(dp), eventos=eventos, desde=max(0, desde),
                                         duracion=duracion, alcance=alcance)

guardado = st.session_state.get("escenario")
if not guardado or guardado["clave"] != S.clave_dataset(dp):
    st.caption(":material/info: Elige uno o más eventos y presiona **Simular escenario**.")
    st.stop()

esc = X.Escenario(guardado["eventos"], guardado["desde"], guardado["duracion"])
afectadas = entidades if guardado["alcance"] == "Todas" else [guardado["alcance"]]

# ---------------------------------------------------------------- cálculo
tabla_pol, nivel, revision = S.politica_actual(dp, entidades)
tabla_pol = tabla_pol.set_index("entidad")
lt_max = float(tabla_pol.loc[afectadas, "lead_time"].max()) + esc.retraso_dias()
necesarios = int(np.ceil((lt_max + revision) / fi["dias"]))
H_an = int(min(res.plan.horizonte_max, max(H, esc.hasta + 2 * necesarios)))
if esc.desde >= H_an:
    st.error(f"El evento empieza después del máximo que se puede pronosticar ({res.plan.horizonte_max} {u_pl}). "
             "Elige una fecha más cercana.")
    st.stop()

base = S.pronostico(H_an)
con_evento = S.pronostico_escenario(H_an, esc.cambios_modelo(), afectadas)
comps = {}
for e in afectadas:
    fila = tabla_pol.loc[e]
    inv = fila["inventario"]
    par = P.Parametros(lead_time_dias=float(fila["lead_time"]), revision_dias=float(revision), nivel_servicio=nivel,
                       inventario_actual=None if pd.isna(inv) else float(inv))
    comps[e] = X.comparar(e, base[e], con_evento[e], par, esc, freq)

hay_inv = all(c.sim_base is not None for c in comps.values())
f_ini = base[afectadas[0]]["fecha"].iloc[min(esc.desde, H_an - 1)]
f_fin = base[afectadas[0]]["fecha"].iloc[min(esc.hasta, H_an) - 1]
descripcion = " + ".join(
    {"demanda": f"demanda {ev.valor:+.0%}", "retraso": f"retraso de {ev.valor:.0f} días",
     "precio": f"precio {ev.valor:+.0%}", "promocion": "promoción",
     "exogena": f"{dp.etiquetas.get(ev.var, ev.var)} {ev.valor:+.0%}"}[ev.tipo].replace(".", ",")
    for ev in esc.eventos)
st.markdown(f"## Resultado · {descripcion}")
st.caption(f"Del {f_ini:%d/%m/%Y} al {f_fin:%d/%m/%Y} · "
           + (f"todas las {S.nombre_entidad(dp, True)}" if len(afectadas) > 1 else afectadas[0])
           + f" · análisis sobre {H_an} {u_pl} · nivel de servicio {nivel}")

if not hay_inv:
    E.nota("Para simular el inventario necesitas el inventario actual. Complétalo en <b>3. Decisiones</b> "
           "(sección Lead time e inventario) y vuelve aquí. Mientras tanto ves el efecto en la demanda y en la política.")

# ---------------------------------------------------------------- resumen de todas
if len(afectadas) > 1:
    filas = []
    for e, c in comps.items():
        r = c.resumen
        filas.append({
            nom: e,
            "Demanda en el evento": (r["demanda_esc_evento"] / r["demanda_base_evento"] - 1) * 100 if r["demanda_base_evento"] else np.nan,
            "Punto de reorden sin evento": r["rop_base"],
            "Punto de reorden con evento": r["rop_esc"],
            f"{u_pl.capitalize()} extra sin stock, sin ajustar": r.get("quiebre_extra_sin", np.nan),
            "Unidades perdidas, sin ajustar": r.get("perdida_extra_sin", np.nan),
            f"{u_pl.capitalize()} extra sin stock, ajustando": r.get("quiebre_extra_aj", np.nan),
        })
    tabla = pd.DataFrame(filas)
    k1, k2, k3, k4 = st.columns(4)
    tot_b = sum(c.resumen["demanda_base_evento"] for c in comps.values())
    tot_e = sum(c.resumen["demanda_esc_evento"] for c in comps.values())
    var_tot = (tot_e / tot_b - 1) * 100 if tot_b else 0
    k1.metric("Demanda durante el evento", E.num(tot_e), delta=E.pct(var_tot) if abs(var_tot) >= 0.05 else None,
              delta_color="off")
    if hay_inv:
        afect = int((tabla[f"{u_pl.capitalize()} extra sin stock, sin ajustar"] > 0).sum())
        k2.metric("Con quiebre si no ajustas", f"{afect} de {len(afectadas)}",
                  help=f"{S.nombre_entidad(dp, True).capitalize()} que se quedan sin stock en algún momento.")
        k3.metric("Sin atender, sin ajustar", E.num(tabla["Unidades perdidas, sin ajustar"].sum()), help="Unidades de demanda perdidas.")
        k4.metric("Sin atender, ajustando", E.num(sum(c.resumen["perdida_extra_aj"] for c in comps.values())))
    st.dataframe(tabla.sort_values(f"{u_pl.capitalize()} extra sin stock, sin ajustar", ascending=False) if hay_inv else tabla,
                 width="stretch", hide_index=True, column_config={
                     "Demanda en el evento": st.column_config.NumberColumn(format="%+.1f%%"),
                     **{c: st.column_config.NumberColumn(format="%.0f") for c in tabla.columns[2:]},
                 })
    st.write("")
    st.markdown("### Detalle")
    ent = S.selector_entidad(afectadas, dp)
else:
    ent = afectadas[0]

# ---------------------------------------------------------------- detalle de una entidad
c = comps[ent]
r = c.resumen
d0, d1 = r["demanda_base_evento"], r["demanda_esc_evento"]
m1, m2, m3, m4 = st.columns(4)
var_d = (d1 / d0 - 1) * 100 if d0 else 0
m1.metric("Demanda durante el evento", E.num(d1), delta=f"{E.pct(var_d)} vs. sin evento" if abs(var_d) >= 0.05 else None,
          delta_color="off")
rop_b, rop_e = r["rop_base"], r["rop_esc"]
m2.metric("Nuevo punto de reorden", E.num(rop_e), delta=f"antes {E.num(rop_b)}", delta_color="off", delta_arrow="off",
          help="Al inicio del evento. La política se recalcula cada período con el pronóstico.")
m3.metric("Nuevo stock de seguridad", E.num(r["ss_esc"]), delta=f"antes {E.num(r['ss_base'])}",
          delta_color="off", delta_arrow="off")
if hay_inv:
    q = r["quiebre_extra_sin"]
    m4.metric(f"{u_pl.capitalize()} sin stock por el evento", q,
              delta=f"{r['quiebre_extra_aj']} ajustando a tiempo", delta_color="off", delta_arrow="off",
              help="Días (o períodos) sin stock que agrega el evento, por sobre los que ya ocurrirían sin él.")

if hay_inv:
    f_aj = c.base["fecha"].iloc[c.inicio_ajuste]
    qs, qa = r["quiebre_extra_sin"], r["quiebre_extra_aj"]
    if qs > 0:
        texto = (f"Si no haces nada, el evento deja a <b>{ent}</b> sin stock <b>{qs} {u_pl if qs != 1 else u}</b> más "
                 f"(unas <b>{E.num(r['perdida_extra_sin'])} unidades</b> sin atender). ")
        if qa == 0:
            texto += (f"Si desde el <b>{f_aj:%d/%m/%Y}</b> planificas con el escenario (punto de reorden "
                      f"<b>{E.num(rop_e)}</b> al inicio del evento), no te quedas sin stock.")
        elif qa < qs:
            texto += (f"Planificando con el escenario desde el <b>{f_aj:%d/%m/%Y}</b> los quiebres bajan a "
                      f"<b>{qa}</b> ({E.num(r['perdida_extra_aj'])} unidades). El evento empieza demasiado pronto para "
                      "evitarlos del todo: conviene un pedido urgente o un proveedor alternativo.")
        else:
            texto += "Ni ajustando la política se alcanza a evitar: conviene un pedido urgente o un proveedor alternativo."
    else:
        texto = (f"El inventario de <b>{ent}</b> aguanta este escenario sin quiebres. Durante el evento el punto de "
                 f"reorden recomendado es <b>{E.num(rop_e)}</b> (sin el evento sería {E.num(rop_b)}).")
    E.nota(texto)
st.write("")

x0, x1 = f_ini, f_fin + pd.tseries.frequencies.to_offset(FRECUENCIAS[freq]["pandas"])


def sombrear(fig):
    fig.add_vrect(x0=x0, x1=x1, fillcolor=E.AMARILLO, opacity=0.12, line_width=0, annotation_text="Evento",
                  annotation_position="top left", annotation_font_color=E.TINTA_2)


col_a, col_b = st.columns(2, gap="medium")
with col_a.container(border=True):
    fig = E.fig_banda(c.escenario["fecha"], c.escenario["P10"], c.escenario["P50"], c.escenario["P90"],
                      nombre_banda="Rango con el evento", nombre_p50="Con el evento")
    fig.add_trace(go.Scatter(x=c.base["fecha"], y=c.base["P50"], name="Sin el evento", mode="lines",
                             line=dict(color=E.TINTA_MUTED, width=1.6, dash="dot"), hovertemplate="%{y:,.0f}"))
    sombrear(fig)
    fig.update_layout(title=f"Demanda pronosticada · {ent}", yaxis_title=dp.etiquetas["objetivo"], height=420,
                      margin=dict(t=110))
    E.grafico(fig, key="fig_esc_dem")
    ev_precio = [ev for ev in esc.eventos if ev.tipo == "precio"]
    otros_modelo = [ev for ev in esc.eventos if ev.tipo in ("promocion", "exogena")]
    if ev_precio and ev_precio[0].elasticidad is None and len(esc.eventos) == 1 and d0:
        el_modelo = ((d1 / d0) - 1) / ev_precio[0].valor
        st.caption(f"Efecto calculado por el modelo: equivale a una elasticidad de {E.num(el_modelo, 2)}. Si parece "
                   "demasiado baja, tu historial tiene pocos cambios de precio: prueba con una elasticidad fija.")
    elif ev_precio and ev_precio[0].elasticidad is not None:
        st.caption(f"Efecto del precio calculado con elasticidad {E.num(ev_precio[0].elasticidad, 2)}.")
    if otros_modelo:
        st.caption("El efecto de la promoción u otras variables lo calcula el modelo con lo que aprendió de tu historial.")

if hay_inv:
    with col_b.container(border=True):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=c.sim_base["fecha"], y=c.sim_base["inventario"], name="Sin el evento", mode="lines",
                                 line=dict(color=E.TINTA_MUTED, width=1.4, dash="dot"), hovertemplate="%{y:,.0f}"))
        fig.add_trace(go.Scatter(x=c.sim_sin_ajuste["fecha"], y=c.sim_sin_ajuste["inventario"], mode="lines",
                                 name="Con el evento, sin ajustar", line=dict(color=E.NARANJO, width=2.2),
                                 hovertemplate="%{y:,.0f}"))
        fig.add_trace(go.Scatter(x=c.sim_ajustada["fecha"], y=c.sim_ajustada["inventario"], mode="lines",
                                 name="Con el evento, ajustando", line=dict(color=E.AZUL, width=2.2),
                                 hovertemplate="%{y:,.0f}"))
        qb = c.sim_sin_ajuste[c.sim_sin_ajuste["quiebre"]]
        if len(qb):
            fig.add_trace(go.Scatter(x=qb["fecha"], y=qb["inventario"], mode="markers", name="Sin stock",
                                     marker=dict(color=E.ROJO, size=8, symbol="x"),
                                     customdata=qb["no_atendida"], hovertemplate="faltan %{customdata:,.0f} u."))
        sombrear(fig)
        fig.update_layout(title=f"Inventario proyectado · {ent}", yaxis_title="Unidades", height=420,
                          margin=dict(t=110))
        E.grafico(fig, key="fig_esc_inv")
        st.caption("Simulación con venta perdida: cuando el inventario llega a cero, la demanda de ese día no se atiende.")
else:
    with col_b.container(border=True):
        st.markdown("**Política durante el evento**")
        t = pd.DataFrame({
            "": ["Stock de seguridad", "Punto de reorden", "Meta (T)", "Lead time (días)"],
            "Actual": [c.dec_base.ss, c.dec_base.rop, c.dec_base.meta, c.dec_base.L * fi["dias"]],
            "Con el evento": [c.dec_esc.ss, c.dec_esc.rop, c.dec_esc.meta, c.dec_esc.L * fi["dias"]],
        })
        st.dataframe(t, hide_index=True, width="stretch",
                     column_config={k: st.column_config.NumberColumn(format="%.0f") for k in ("Actual", "Con el evento")})

# ---------------------------------------------------------------- guardar
if registro:
    with st.container(border=True):
        g1, g2 = st.columns([3, 1], vertical_alignment="bottom")
        nombre_esc = g1.text_input("Guardar este escenario como", value=descripcion.capitalize()[:60], key="esc_nombre")
        if g2.button("Guardar escenario", icon=":material/bookmark_add:", width="stretch"):
            nuevo = dict(nombre=nombre_esc.strip() or descripcion, desde=esc.desde, duracion=esc.duracion,
                         alcance=guardado["alcance"],
                         eventos=[dict(tipo=ev.tipo, valor=ev.valor, var=ev.var, elasticidad=ev.elasticidad)
                                  for ev in esc.eventos])
            lista = [x for x in registro.get("escenarios", []) if x["nombre"] != nuevo["nombre"]] + [nuevo]
            registro["escenarios"] = lista
            S.actualizar_registro(escenarios=lista)
            st.toast(f"Escenario “{nuevo['nombre']}” guardado en Mis pronósticos.", icon=":material/bookmark_added:")
elif cuenta.login_disponible() and not cuenta.usuario():
    st.caption(":material/lock_open: Inicia sesión para guardar tus escenarios y volver a ellos.")

# ---------------------------------------------------------------- descarga
hojas = {}
if len(afectadas) > 1:
    hojas["Resumen"] = tabla.round(1)
det = c.base[["fecha", "P50"]].rename(columns={"P50": "Demanda sin evento"})
det["Demanda con evento"] = c.escenario["P50"].to_numpy()
if hay_inv:
    det["Inventario sin evento"] = c.sim_base["inventario"].to_numpy()
    det["Inventario con evento, sin ajustar"] = c.sim_sin_ajuste["inventario"].to_numpy()
    det["No atendida, sin ajustar"] = c.sim_sin_ajuste["no_atendida"].to_numpy()
    det["Inventario con evento, ajustando"] = c.sim_ajustada["inventario"].to_numpy()
    det["Pedido, ajustando"] = c.sim_ajustada["pedido"].to_numpy()
det["fecha"] = det["fecha"].dt.date
hojas[f"Detalle {ent}"[:31]] = det.rename(columns={"fecha": "Fecha"}).round(1)
st.download_button("Descargar escenario (Excel)", E.excel_bytes(hojas), file_name="escenario.xlsx",
                   icon=":material/download:",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

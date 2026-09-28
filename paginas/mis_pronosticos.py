from datetime import datetime

import streamlit as st

import cuenta
import estilo as E
import sesion as S
from motor import almacen as A
from motor import repositorio as Rp
from motor.datos import FRECUENCIAS

S.panel_dataset()
E.encabezado("Tu cuenta", "Mis pronósticos",
             "Tus pronósticos guardados, con sus datos, política y escenarios. Ábrelos sin volver a subir el archivo.")

u = cuenta.usuario()
if not u:
    if cuenta.login_disponible():
        with st.container(border=True):
            st.markdown("**Inicia sesión para guardar tus pronósticos**")
            st.caption("Sin sesión puedes usar todo el sitio, pero nada queda guardado a tu nombre. Con sesión, cada "
                       "pronóstico que generes queda aquí junto con tu archivo, la política y los escenarios.")
            if st.button("Continuar con Google" if cuenta.login_google_disponible() else "Iniciar sesión (simulado)",
                         icon=":material/login:", type="primary", key="login_mis"):
                cuenta.iniciar_sesion()
    else:
        st.info("El inicio de sesión todavía no está configurado en este sitio.", icon=":material/info:")
    st.stop()

repo = S.repositorio()
try:
    lista = repo.listar(u["correo"])
except Exception as e:  # noqa: BLE001
    st.error(f"No pudimos leer tus pronósticos: {e}")
    st.stop()

# pronóstico abierto que todavía no está guardado (por ejemplo, se generó antes de iniciar sesión)
dp_act, res_act = S.resultado()
if res_act is not None and not S.registro_actual():
    with st.container(border=True):
        c1, c2 = st.columns([3, 1], vertical_alignment="center")
        c1.markdown(f"**Tienes un pronóstico abierto sin guardar** · {st.session_state.get('nombre_dataset', '')}")
        if c2.button("Guardarlo", icon=":material/save:", width="stretch"):
            err = S.guardar_pronostico_actual()
            st.error(err) if err else st.rerun()

if not lista:
    st.info("Todavía no tienes pronósticos guardados. Genera uno en **1. Tus datos** y quedará aquí automáticamente.",
            icon=":material/inbox:")
    st.page_link("paginas/datos.py", label="Ir a Tus datos", icon=":material/arrow_forward:")
    st.stop()


def fecha_corta(valor):
    try:
        return datetime.fromisoformat(str(valor).replace("Z", "+00:00")).strftime("%d/%m/%Y %H:%M")
    except ValueError:
        return str(valor)[:16]


activo = (S.registro_actual() or {}).get("id")
st.caption(f"{len(lista)} pronóstico{'s' if len(lista) != 1 else ''} guardado{'s' if len(lista) != 1 else ''}")
for reg in lista:
    fi = FRECUENCIAS.get(reg.get("frecuencia") or "D", FRECUENCIAS["D"])
    with st.container(border=True):
        c1, c2 = st.columns([2.2, 1.8], vertical_alignment="center")
        with c1:
            insignias = E.insignia("Abierto ahora", "ok") if reg["id"] == activo else ""
            n_esc = len(reg.get("escenarios") or [])
            st.markdown(f"**{reg['nombre']}** &nbsp; {insignias}", unsafe_allow_html=True)
            detalle = [f"{E.num(reg.get('n_entidades') or 0)} entidades", f"datos {fi['nombre']}s",
                       f"horizonte {reg.get('horizonte')} {fi['unidad_pl']}",
                       f"error {E.pct(reg['error_pct'])}" if reg.get("error_pct") is not None else None,
                       f"{n_esc} escenario{'s' if n_esc != 1 else ''}" if n_esc else None]
            st.caption(" · ".join(d for d in detalle if d) + f"  \nArchivo {reg.get('archivo_nombre')} · "
                       f"actualizado {fecha_corta(reg.get('actualizado'))}")
        with c2:
            b1, b2, b3 = st.columns(3)
            if b1.button("Abrir", key=f"abrir_{reg['id']}", type="primary", width="stretch"):
                barra = st.progress(0.0, text="Abriendo…")
                try:
                    S.abrir_pronostico(reg, lambda f, t: barra.progress(min(f, 1.0), text=t))
                    barra.empty()
                    st.switch_page("paginas/pronostico.py")
                except Exception as e:  # noqa: BLE001
                    barra.empty()
                    st.error(f"No se pudo abrir: {e}")
            with b2.popover("Renombrar", width="stretch"):
                nuevo = st.text_input("Nuevo nombre", reg["nombre"], key=f"nom_{reg['id']}")
                if st.button("Guardar", key=f"ren_{reg['id']}"):
                    repo.actualizar(u["correo"], reg["id"], {"nombre": nuevo.strip() or reg["nombre"]})
                    if reg["id"] == activo:
                        st.session_state["registro"]["nombre"] = nuevo.strip()
                    st.rerun()
            with b3.popover("Borrar", width="stretch"):
                st.caption("Se borra el pronóstico, su archivo de datos y sus escenarios. No se puede deshacer.")
                if st.button("Sí, borrar", key=f"del_{reg['id']}", type="primary"):
                    repo.borrar(u["correo"], reg["id"])
                    try:
                        S.almacen_persistente().borrar(A.ruta_datos(Rp.id_usuario(u["correo"]), reg["id"]))
                    except Exception:  # noqa: BLE001
                        pass
                    if reg["id"] == activo:
                        st.session_state.pop("registro", None)
                    st.rerun()

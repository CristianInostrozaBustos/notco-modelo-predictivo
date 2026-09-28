"""Sesión de usuario: login con Google (nativo de Streamlit) y modo abierto sin login.

- Sin la sección [auth] en los secrets, el sitio funciona solo en modo abierto.
- Con [auth] configurado aparece "Continuar con Google".
- Para probar sin Google existe un usuario simulado ([desarrollo] usuario_simulado = "correo").
  Nunca lo configures en el sitio publicado.
"""

import streamlit as st


def _secreto(seccion):
    try:
        return dict(st.secrets[seccion]) if seccion in st.secrets else None
    except Exception:  # noqa: BLE001  (no hay archivo de secrets)
        return None


def login_google_disponible() -> bool:
    auth = _secreto("auth")
    return bool(auth and (auth.get("client_id") or any(isinstance(v, dict) for v in auth.values())))


def usuario_simulado():
    dev = _secreto("desarrollo") or {}
    return dev.get("usuario_simulado")


def login_disponible() -> bool:
    return login_google_disponible() or bool(usuario_simulado())


def usuario():
    """dict(correo, nombre) del usuario con sesión iniciada, o None en modo abierto."""
    if usuario_simulado() and st.session_state.get("_sesion_simulada"):
        correo = usuario_simulado()
        return {"correo": correo.lower(), "nombre": correo.split("@")[0].capitalize()}
    if login_google_disponible():
        try:
            if st.user.is_logged_in:
                correo = (st.user.get("email") or "").lower()
                if correo:
                    return {"correo": correo, "nombre": st.user.get("name") or correo.split("@")[0]}
        except Exception:  # noqa: BLE001
            return None
    return None


def iniciar_sesion():
    if login_google_disponible():
        st.login()
    elif usuario_simulado():
        st.session_state["_sesion_simulada"] = True
        st.rerun()


def cerrar_sesion():
    for k in ("_sesion_simulada", "registro", "resultado", "dp", "df_raw", "nombre_dataset", "archivo_bytes",
              "politica", "escenario"):
        st.session_state.pop(k, None)
    if login_google_disponible() and not usuario_simulado():
        st.logout()
    st.rerun()


def caja_cuenta():
    """Bloque de cuenta en la barra lateral (en todas las páginas)."""
    if not login_disponible():
        return
    u = usuario()
    with st.sidebar:
        st.write("")
        if u:
            inicial = (u["nombre"] or "?")[0].upper()
            st.markdown(
                f"""<div style="display:flex;align-items:center;gap:10px;margin:12px 0 14px 8px">
                  <div style="width:32px;height:32px;border-radius:50%;background:#1c5cab;color:#fff;display:flex;
                    align-items:center;justify-content:center;font-weight:700">{inicial}</div>
                  <div style="line-height:1.2"><div style="font-weight:600;font-size:.9rem">{u['nombre']}</div>
                  <div style="font-size:.75rem;color:#898781;word-break:break-all">{u['correo']}</div></div>
                </div>""",
                unsafe_allow_html=True,
            )
            if st.button("Cerrar sesión", icon=":material/logout:", key="btn_logout", width="stretch"):
                cerrar_sesion()
        else:
            etiqueta = "Continuar con Google" if login_google_disponible() else "Iniciar sesión (simulado)"
            if st.button(etiqueta, icon=":material/login:", key="btn_login", type="primary", width="stretch"):
                iniciar_sesion()
            st.caption("Opcional: inicia sesión para guardar tus pronósticos y volver a ellos.")

"""Almacenamiento de archivos: modelos entrenados y archivos de datos de los usuarios.

Todo vive en un solo lugar (un bucket de Supabase Storage o una carpeta local) organizado por rutas:

    modelos/<version>_<huella>.zip        modelos entrenados (compartidos: misma huella = mismos datos)
    datos/<usuario>/<id_pronostico>.bin    archivo original que subió un usuario con sesión iniciada

Cada modelo se guarda como un .zip con:
    modelo.keras   la red entrenada
    meta.pkl       todo lo demás del ResultadoModelo (escalador, series, métricas, plan...)
    version.txt    versión del motor; si no coincide, el modelo se ignora y se vuelve a entrenar

Configuración en los secrets del sitio (sin ella se usa la carpeta local almacen_local/):

    [almacen]
    tipo = "supabase"
    url = "https://xxxx.supabase.co"
    key = "service_role key"
    bucket = "motor"
"""

from __future__ import annotations

import io
import os
import pickle
import tempfile
import time
import zipfile

import requests

VERSION_MOTOR = "m1"          # subir cuando cambie la arquitectura o el formato guardado
MAX_MODELOS_LOCALES = 40


def ruta_modelo(clave: str) -> str:
    return f"modelos/{VERSION_MOTOR}_{clave}.zip"


def ruta_datos(usuario_id: str, id_pronostico: str) -> str:
    return f"datos/{usuario_id}/{id_pronostico}.bin"


# ---------------------------------------------------------------- lugares de guardado

class CarpetaLocal:
    nombre = "carpeta local"

    def __init__(self, ruta="almacen_local"):
        self.raiz = ruta
        os.makedirs(ruta, exist_ok=True)

    def _p(self, ruta):
        p = os.path.normpath(os.path.join(self.raiz, ruta))
        if not p.startswith(os.path.normpath(self.raiz)):
            raise ValueError("ruta fuera del almacén")
        return p

    def existe(self, ruta):
        return os.path.exists(self._p(ruta))

    def leer(self, ruta):
        p = self._p(ruta)
        if not os.path.exists(p):
            return None
        with open(p, "rb") as fh:
            return fh.read()

    def escribir(self, ruta, contenido):
        p = self._p(ruta)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p + ".tmp", "wb") as fh:
            fh.write(contenido)
        os.replace(p + ".tmp", p)
        if ruta.startswith("modelos/"):
            self._podar()

    def borrar(self, ruta):
        p = self._p(ruta)
        if os.path.exists(p):
            os.remove(p)

    def listar(self, prefijo="modelos/"):
        d = self._p(prefijo)
        if not os.path.isdir(d):
            return []
        out = []
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if os.path.isfile(p):
                out.append(dict(archivo=prefijo + f, bytes=os.path.getsize(p), modificado=time.ctime(os.path.getmtime(p))))
        return out

    def _podar(self):
        d = self._p("modelos/")
        archivos = sorted((os.path.join(d, f) for f in os.listdir(d) if f.endswith(".zip")), key=os.path.getmtime)
        for p in archivos[:-MAX_MODELOS_LOCALES]:
            os.remove(p)


class SupabaseStorage:
    nombre = "Supabase Storage"

    def __init__(self, url, key, bucket="motor", timeout=60):
        self.base = url.rstrip("/") + "/storage/v1"
        self.bucket = bucket
        self.h = {"Authorization": f"Bearer {key}", "apikey": key}
        self.timeout = timeout

    def _url(self, ruta):
        return f"{self.base}/object/{self.bucket}/{ruta}"

    def existe(self, ruta):
        return requests.head(self._url(ruta), headers=self.h, timeout=self.timeout).status_code == 200

    def leer(self, ruta):
        r = requests.get(self._url(ruta), headers=self.h, timeout=self.timeout)
        return r.content if r.status_code == 200 else None

    def escribir(self, ruta, contenido):
        h = dict(self.h, **{"x-upsert": "true", "Content-Type": "application/octet-stream"})
        r = requests.post(self._url(ruta), headers=h, data=contenido, timeout=self.timeout)
        r.raise_for_status()

    def borrar(self, ruta):
        r = requests.delete(f"{self.base}/object/{self.bucket}", headers=self.h, json={"prefixes": [ruta]},
                            timeout=self.timeout)
        r.raise_for_status()

    def listar(self, prefijo="modelos/"):
        r = requests.post(f"{self.base}/object/list/{self.bucket}", headers=self.h,
                          json={"prefix": prefijo, "limit": 1000}, timeout=self.timeout)
        r.raise_for_status()
        return [dict(archivo=prefijo + o["name"], bytes=(o.get("metadata") or {}).get("size"), modificado=o.get("updated_at"))
                for o in r.json()]


def crear(config: dict | None):
    config = config or {}
    if config.get("tipo") == "supabase":
        return SupabaseStorage(config["url"], config["key"], config.get("bucket", "motor"))
    return CarpetaLocal(config.get("ruta", "almacen_local"))


# ---------------------------------------------------------------- modelos

def a_bytes(res) -> bytes:
    modelo = res.modelo
    fn = getattr(modelo, "_fn_inferencia", None)
    if fn is not None:
        del modelo._fn_inferencia
    try:
        with tempfile.TemporaryDirectory() as d:
            ruta = os.path.join(d, "modelo.keras")
            modelo.save(ruta)
            with open(ruta, "rb") as fh:
                bytes_modelo = fh.read()
    finally:
        if fn is not None:
            modelo._fn_inferencia = fn
    res.modelo = None
    try:
        meta = pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        res.modelo = modelo
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("modelo.keras", bytes_modelo)
        z.writestr("meta.pkl", meta)
        z.writestr("version.txt", VERSION_MOTOR)
    return buf.getvalue()


def desde_bytes(contenido: bytes):
    from tensorflow import keras
    from . import modelo as M  # registra la capa SeleccionarPorEntidad

    with zipfile.ZipFile(io.BytesIO(contenido)) as z:
        if z.read("version.txt").decode() != VERSION_MOTOR:
            return None
        meta = z.read("meta.pkl")
        bytes_modelo = z.read("modelo.keras")
    res = pickle.loads(meta)  # solo se leen archivos que escribió el propio sitio
    with tempfile.TemporaryDirectory() as d:
        ruta = os.path.join(d, "modelo.keras")
        with open(ruta, "wb") as fh:
            fh.write(bytes_modelo)
        res.modelo = keras.models.load_model(ruta, compile=False, safe_mode=False,
                                             custom_objects={"SeleccionarPorEntidad": M.SeleccionarPorEntidad})
    return res


def guardar(almacen, clave, res) -> str | None:
    """Devuelve None si salió bien, o el texto del error (guardar nunca debe botar el sitio)."""
    try:
        almacen.escribir(ruta_modelo(clave), a_bytes(res))
        return None
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def cargar(almacen, clave):
    """Devuelve (res, error). res es None si no existe o no se pudo leer."""
    try:
        contenido = almacen.leer(ruta_modelo(clave))
        if contenido is None:
            return None, None
        return desde_bytes(contenido), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def existe_modelo(almacen, clave) -> bool:
    try:
        return almacen.existe(ruta_modelo(clave))
    except Exception:  # noqa: BLE001
        return False

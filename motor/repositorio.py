"""Registro de "Mis pronósticos": qué pronósticos tiene guardados cada usuario.

Dos implementaciones con la misma interfaz:
- SupabaseRepositorio: tabla `pronosticos` en el Postgres de Supabase (API REST).
- RepositorioLocal: SQLite en un archivo, para correr en local o probar sin Supabase.

Seguridad: el sitio usa la service_role key, que tiene acceso completo, así que TODAS las
operaciones filtran por el usuario con sesión iniciada. En Supabase la tabla tiene RLS activado
y sin políticas públicas: con la clave pública (anon) nadie puede leer ni escribir nada.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone

import requests

CAMPOS_JSON = ("config", "politica", "escenarios")
CAMPOS = ("id", "usuario", "nombre", "creado", "actualizado", "archivo_nombre", "clave_modelo", "frecuencia",
          "n_entidades", "horizonte", "error_pct", "config", "politica", "escenarios")

SQL_SUPABASE = """
-- Motor Predictivo de Abastecimiento: tabla de pronósticos guardados por usuario
create table if not exists public.pronosticos (
    id              uuid primary key default gen_random_uuid(),
    usuario         text not null,              -- correo del usuario (login con Google)
    nombre          text not null,
    creado          timestamptz not null default now(),
    actualizado     timestamptz not null default now(),
    archivo_nombre  text,
    clave_modelo    text not null,              -- huella de los datos (modelo en Storage)
    frecuencia      text,
    n_entidades     integer,
    horizonte       integer,
    error_pct       double precision,
    config          jsonb not null default '{}'::jsonb,   -- columnas y opciones con que se prepararon los datos
    politica        jsonb,                                  -- nivel de servicio, revisión, lead time e inventario
    escenarios      jsonb not null default '[]'::jsonb      -- escenarios guardados
);
create index if not exists pronosticos_usuario_idx on public.pronosticos (usuario, creado desc);

-- Solo el servidor del sitio (service_role) accede. Sin políticas, la clave pública no puede leer nada.
alter table public.pronosticos enable row level security;
""".strip()


def id_usuario(correo: str) -> str:
    """Identificador estable y no reversible para las rutas de archivos."""
    return hashlib.sha256(correo.strip().lower().encode()).hexdigest()[:20]


def _ahora():
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------- local (SQLite)

class RepositorioLocal:
    nombre = "base local (SQLite)"

    def __init__(self, ruta="almacen_local/pronosticos.db"):
        os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
        self.ruta = ruta
        with self._con() as c:
            c.execute("""create table if not exists pronosticos (
                id text primary key, usuario text not null, nombre text not null, creado text, actualizado text,
                archivo_nombre text, clave_modelo text, frecuencia text, n_entidades integer, horizonte integer,
                error_pct real, config text, politica text, escenarios text)""")

    def _con(self):
        return sqlite3.connect(self.ruta)

    @staticmethod
    def _fila(r):
        d = dict(zip(CAMPOS, r))
        for k in CAMPOS_JSON:
            d[k] = json.loads(d[k]) if d[k] else ([] if k == "escenarios" else None)
        return d

    def listar(self, usuario):
        with self._con() as c:
            filas = c.execute(f"select {','.join(CAMPOS)} from pronosticos where usuario=? order by creado desc",
                              (usuario,)).fetchall()
        return [self._fila(r) for r in filas]

    def obtener(self, usuario, id_):
        with self._con() as c:
            r = c.execute(f"select {','.join(CAMPOS)} from pronosticos where usuario=? and id=?", (usuario, id_)).fetchone()
        return self._fila(r) if r else None

    def crear(self, usuario, datos):
        d = {k: datos.get(k) for k in CAMPOS}
        d.update(id=str(uuid.uuid4()), usuario=usuario, creado=_ahora(), actualizado=_ahora())
        d["escenarios"] = d["escenarios"] or []
        valores = [json.dumps(d[k], ensure_ascii=False) if k in CAMPOS_JSON else d[k] for k in CAMPOS]
        with self._con() as c:
            c.execute(f"insert into pronosticos ({','.join(CAMPOS)}) values ({','.join('?' * len(CAMPOS))})", valores)
        return d

    def actualizar(self, usuario, id_, cambios):
        cambios = dict(cambios, actualizado=_ahora())
        sets = ", ".join(f"{k}=?" for k in cambios)
        valores = [json.dumps(v, ensure_ascii=False) if k in CAMPOS_JSON else v for k, v in cambios.items()]
        with self._con() as c:
            c.execute(f"update pronosticos set {sets} where usuario=? and id=?", valores + [usuario, id_])

    def borrar(self, usuario, id_):
        with self._con() as c:
            c.execute("delete from pronosticos where usuario=? and id=?", (usuario, id_))


# ---------------------------------------------------------------- Supabase (PostgREST)

class SupabaseRepositorio:
    nombre = "Supabase (Postgres)"

    def __init__(self, url, key, timeout=30):
        self.base = url.rstrip("/") + "/rest/v1/pronosticos"
        self.h = {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self.timeout = timeout

    def _q(self, usuario, id_=None):
        p = {"usuario": f"eq.{usuario}"}
        if id_:
            p["id"] = f"eq.{id_}"
        return p

    def listar(self, usuario):
        p = dict(self._q(usuario), select="*", order="creado.desc")
        r = requests.get(self.base, headers=self.h, params=p, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def obtener(self, usuario, id_):
        r = requests.get(self.base, headers=self.h, params=dict(self._q(usuario, id_), select="*"), timeout=self.timeout)
        r.raise_for_status()
        filas = r.json()
        return filas[0] if filas else None

    def crear(self, usuario, datos):
        d = {k: v for k, v in datos.items() if k in CAMPOS and k not in ("id", "creado", "actualizado") and v is not None}
        d["usuario"] = usuario
        r = requests.post(self.base, headers=dict(self.h, Prefer="return=representation"), json=d, timeout=self.timeout)
        r.raise_for_status()
        return r.json()[0]

    def actualizar(self, usuario, id_, cambios):
        cambios = dict(cambios, actualizado=_ahora())
        r = requests.patch(self.base, headers=self.h, params=self._q(usuario, id_), json=cambios, timeout=self.timeout)
        r.raise_for_status()

    def borrar(self, usuario, id_):
        r = requests.delete(self.base, headers=self.h, params=self._q(usuario, id_), timeout=self.timeout)
        r.raise_for_status()


def crear(config: dict | None):
    config = config or {}
    if config.get("tipo") == "supabase":
        return SupabaseRepositorio(config["url"], config["key"])
    return RepositorioLocal(os.path.join(config.get("ruta", "almacen_local"), "pronosticos.db"))

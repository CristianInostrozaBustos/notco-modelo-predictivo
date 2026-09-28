import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from motor import datos as D, reglas as R, modelo as Mo, politica as P

def correr(ruta):
    df = D.leer_archivo(ruta, open(ruta, "rb").read())
    det = D.detectar_roles(df)
    f, _ = D.parsear_fechas(df[det.roles["fecha"]])
    fr, _ = D.detectar_frecuencia(f, df[det.roles["entidad"]] if det.roles["entidad"] else None)
    dp = D.preparar(df, D.Configuracion(roles=det.roles, exogenas=det.exogenas, frecuencia=fr))
    plan = R.planificar(dp)
    res = Mo.entrenar_motor(dp, plan)
    print("=" * 80)
    print(os.path.basename(ruta), plan.modo, plan.esquema, "| validación", plan.validacion, "| ventana elegida:", res.ventana,
          "| épocas", res.epocas, "| seg:", res.segundos)
    print(res.busqueda.round(2).to_string(index=False))
    print(res.metricas_entidad.round(2).to_string(index=False))
    fut = Mo.pronosticar(res, plan.horizonte_defecto, fr)
    e0 = next(iter(fut)); pr = fut[e0]
    print("  pronóstico", e0, len(pr), "períodos, P50 medio", round(pr.P50.mean(), 1))
    return dp, res, fut

if __name__ == "__main__":
    for r in sys.argv[1:]:
        correr(r)

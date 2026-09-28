import sys, glob, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from motor import datos as D, reglas as R

archivos = sorted(glob.glob("ejemplos/*.csv")) + ["tests/control_notco.csv"]
for a in archivos:
    df = D.leer_archivo(a, open(a, "rb").read())
    det = D.detectar_roles(df)
    ent = df[det.roles["entidad"]] if det.roles["entidad"] else None
    fechas, nota = D.parsear_fechas(df[det.roles["fecha"]])
    freq, info = D.detectar_frecuencia(fechas, ent)
    cfg = D.Configuracion(roles=det.roles, exogenas=det.exogenas, frecuencia=freq)
    dp = D.preparar(df, cfg)
    plan = R.planificar(dp)
    print("="*90); print(os.path.basename(a), df.shape)
    print(" roles:", {k: v for k, v in det.roles.items() if v}, "| exog:", det.exogenas)
    print(" advertencias:", det.advertencias)
    print(" frecuencia:", freq, info, "| fechas:", nota)
    print(" preparado:", dp.df.shape, "| vars modelo:", dp.variables_modelo)
    for r in dp.reporte: print("   -", r)
    print(" plan:", plan.modo, plan.esquema, "ventanas", plan.ventanas, "validación", plan.validacion, "h_max", plan.horizonte_max, "incl", len(plan.entidades_incluidas), "excl", len(plan.entidades_excluidas))
    ind = R.indicadores(dp)
    print(" indicadores:", {e: sum(i.estado == e for i in ind) for e in ("disponible", "manual", "no disponible")})

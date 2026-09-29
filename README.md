# Sitio de visualización — Modelo predictivo NotCo

Sitio web interactivo que carga el modelo LSTM global (`modelo_global.keras`) en vivo
y permite consultar cualquiera de los 5 SKUs de NotCo sin reentrenar nada.

## Contenido

- `app.py` — aplicación Streamlit completa (3 secciones).
- `modelo_global.keras` — modelo entrenado (tronco compartido + cabeza por SKU, Dense(16), semilla 42).
- `dataset_notco_5sku_5anos.csv` — dataset simulado de los 5 SKUs.
- `requirements.txt` — dependencias.

## Secciones del sitio

1. **Pronóstico y política ROP/SS** — validación del modelo (real vs. P10/P50/P90) sobre
   los últimos 84 días de cada SKU, más el cálculo de la política de revisión periódica
   (Meta T, Stock de Seguridad, ROP diario de alarma).
2. **Escenario what-if** — simula un evento de estrés de abastecimiento (parámetros
   ajustables: fechas, sobrecosto, elasticidad) sobre el insumo crítico del SKU
   seleccionado, con pronóstico recursivo día a día y comparación de costos entre el
   proveedor primario (afectado) y el secundario.
3. **Simulación de inventario** — aplica la política híbrida (revisión mensual +
   monitoreo diario con gatillo de compra de emergencia) sobre el escenario ejecutado
   en la sección 2.

## Cómo correrlo localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

Se abre automáticamente en `http://localhost:8501`.

## Cómo desplegarlo gratis (para que cualquiera lo vea sin instalar nada)

**Opción recomendada — Streamlit Community Cloud:**
1. Sube esta carpeta completa a un repositorio de GitHub (los 4 archivos).
2. Entra a https://streamlit.io/cloud, conecta tu cuenta de GitHub.
3. Selecciona el repositorio y el archivo `app.py`. Despliega — queda con una URL pública
   gratis (tipo `tuproyecto.streamlit.app`).

**Nota sobre el tamaño del repositorio:** `modelo_global.keras` pesa ~560 KB y el dataset
~950 KB — están dentro de los límites gratuitos de GitHub y Streamlit Cloud sin problema.

## Nota técnica importante

Este notebook original usaba una función `calcular_rop_diario` en el bloque de
simulación híbrida (Bloque 4 / OE3) que **nunca quedó definida** en el archivo `.ipynb`
subido — solo se usaba. Está incluida y funcionando en este `app.py`; si sigues editando
el notebook de Colab, agrega esta definición antes de usarla:

```python
def calcular_rop_diario(demanda, sigma, lead_time, z):
    ss = z * sigma * np.sqrt(lead_time)
    rop = demanda * lead_time + ss
    return round(rop), round(ss)
```

## Extensión futura declarada (no implementada)

La arquitectura fue diseñada considerando su extensibilidad hacia un motor genérico que
acepte datasets de otras organizaciones/productos. Esto queda fuera del alcance actual
(requeriría detección automática de esquema de columnas, entrenamiento en vivo desde la
interfaz, y generalización del escenario what-if a insumos/eventos arbitrarios).

# Lab02_MyST_Equipo7

**Integrantes:** Gian Carlo Campos Sayavedra (GCCS11) y Juan Pablo Barba González (JPBAG0)

**Nivel de alcance:** B (2 integrantes)

## Descripción

Estrategia sistemática sobre BTCUSDT con barras de 5 minutos (archivos publicados en Canvas: entrenamiento de 2022-06 a 2023-12 y prueba de 2023-12 a 2024-06). Combina tres indicadores de tres familias (Donchian como tendencia, ROC como momento y Keltner como volatilidad) con una regla de confirmación de 2 de 3, se evalúa en un motor de backtesting orientado a eventos con comisión de 0.125% por operación y ejecución en la apertura de la barra siguiente, y optimiza sus hiperparámetros con Optuna (TPE) maximizando el Calmar mediante walk-forward semanal (1 mes de entrenamiento y 1 semana de prueba). Como nivel B, detecta regímenes de mercado (reversión o rango, tendencia y crisis) con reglas sobre la volatilidad y el R² de una ventana de una semana, y compara un único conjunto de parámetros contra un conjunto por régimen.

## Regla de confirmación y protocolo

La regla de confirmación es: sea L el número de indicadores con señal +1 y S el número con señal -1; la señal es +1 si L >= 2, -1 si S >= 2 y 0 en otro caso.

Los parámetros finales se optimizaron una sola vez sobre todo el entrenamiento, se guardaron en `docs/theta_frozen.json` y se commitearon (commit `43aa031`) **antes** de abrir el archivo de prueba, que se evaluó una sola vez. El orden de los commits demuestra la secuencia, no la intención: el CSV de prueba estaba en el repositorio desde el inicio.

## Resultados principales

- Walk-forward semanal sobre entrenamiento (65 semanas fuera de muestra): un solo conjunto de parámetros con filtro de calidad +5.1% (Sharpe 0.31); un conjunto por régimen -35.1%; comprar y mantener +90.1%.
- Prueba (33 días con datos, parámetros congelados): un solo conjunto +0.7% con 8 operaciones; uno por régimen +1.0% con 9 operaciones; comprar y mantener +16.4%. Con tan pocas operaciones no hay evidencia de ventaja.
- No hay diferencia significativa de retorno por operación entre regímenes (p = 0.93).
- El detalle y la discusión están en `docs/reporte.pdf`.

## Instalación

Probado con Python 3.14.7.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Los datos crudos congelados están en `data/` (`btc_project_train.csv` y `btc_project_test.csv`, tal como se descargaron de Canvas).

## Reproducir los resultados

```bash
python main.py
```

Corre todo el proyecto (entre 9 y 21 minutos según la máquina): audita los datos, hace el walk-forward semanal, recalcula los parámetros sobre todo el entrenamiento, los evalúa en prueba, calcula la robustez y el análisis de régimen, y deja las tablas en `docs/tables`, las figuras en `docs/figures` y el registro en la consola. Verifica además que los parámetros recalculados coinciden con los congelados en `docs/theta_frozen.json` (no los sobrescribe). Para comprobar solamente que todo corre (alrededor de 1 minuto; usa 8 pruebas por ventana, escribe en una carpeta temporal y no toca `docs/`):

```bash
python main.py --quick
```

Pruebas:

```bash
pytest
```

## Semilla aleatoria

Fijada en `42` (constante `SEED` en `src/optimize.py`): la usan el muestreador TPE de Optuna, el bootstrap y el silhouette. `main.py` además fija las semillas globales de `random` y `numpy`.

## Estructura

```
main.py            punto de entrada
data/              datos crudos congelados
src/
  data.py          carga, limpieza, auditoría y tramos continuos
  signals.py       indicadores y regla de confirmación 2 de 3
  backtest.py      motor orientado a eventos con costos
  metrics.py       Sharpe, Sortino, Calmar, drawdown, win rate, retornos periódicos
  optimize.py      optimización, walk-forward, congelado de parámetros y robustez
  regimes.py       detección de régimen
  plots.py         figuras
tests/             pruebas con pytest
notebooks/         análisis y figuras (sin lógica nueva)
docs/              reporte, presentación, parámetros congelados, bitácora, tablas y figuras
```

## Notas

- La exploración de otros métodos de régimen (K-means y un HMM con probabilidades filtradas) se retiró del código al final del proyecto; se conserva en la etiqueta de git `exploracion-regimenes` y está resumida en `docs/EXPERIMENTS.md`.
- Los datos tienen huecos (una ventana diaria sin cotización, un hueco de 2 días en entrenamiento y uno de 122 días en prueba). No se rellenó ningún precio: la serie se corta en tramos continuos. El tratamiento está descrito en el reporte.

## Uso de asistencia de IA

[REVISAR Y AJUSTAR ANTES DE ENTREGAR: debe describir con exactitud lo que hicieron ustedes dos.]

Se usó Claude (Anthropic) como asistente a lo largo del proyecto, en estas partes:

- Discusión de decisiones de diseño: elección de indicadores por correlación entre señales, política de salida, método de régimen y protocolo para congelar parámetros antes de evaluar en prueba.
- Redacción inicial del código de `src/`, de `main.py` y de las pruebas (incluidas dos pruebas cuyo equity se calculó a mano), que los autores revisaron, ejecutaron en su equipo y modificaron.
- Diagnósticos de los resultados y borradores del reporte, de este README, del notebook y de la presentación.

Los autores ejecutaron todo el código, verificaron los resultados y son responsables de cada línea del repositorio.

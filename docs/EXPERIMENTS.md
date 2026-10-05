# Bitácora de experimentos

Todo lo siguiente usa solo `btc_project_train.csv`. El archivo de test no se ha usado.

## Ejecuciones

| # | Qué se corrió | Configuraciones evaluadas | Tiempo | Resultado |
|---|---|---|---|---|
| A | m=5, r=2 con signal_exit_after en {0, 48, 288, nunca} | 4 | segundos | 6,169 a 2,390 operaciones, retorno entre -100% y -99.3% |
| B | m en {10, 20, 40}, sin salida por señal | 3 | segundos | retorno -83%, -49%, -19%; bruto por operación 0.07%, 0.14%, 0.16% |
| 1 | Walk-forward, 65 ventanas x 150 pruebas TPE, un solo theta, sin gate | 9,750 | 161 s | OOS -98.1%, 2,142 operaciones, Sharpe -12.5 |
| 2 | Mismas optimizaciones de la ejecución 1, operando solo si el mejor Calmar de entrenamiento es mayor a 0 | 9,750 (las mismas) | 162 s | OOS +5.1%, 354 operaciones, Sharpe 0.31, 51 de 65 semanas operadas |
| 3 | Walk-forward con un theta por régimen (reglas), 150 pruebas por régimen y ventana, gate 0 | 21,450 | 294 s | OOS -35.1%, 430 operaciones, Sharpe -1.55, drawdown -39.4% |

Referencia en las mismas semanas: comprar y mantener +90.1%, Sharpe 1.51, drawdown -37.9%.

## Análisis del régimen (no evalúa la estrategia)
- Variables: volatilidad, R^2 de tendencia y autocorrelación horaria. La autocorrelación se descartó de la clasificación porque con ella el silhouette de K-means cae de 0.47 a 0.29, las rachas se acortan y el cluster de crisis pasa de 9% a 25% del tiempo.
- Métodos comparados en train: reglas (silhouette 0.45), K-means de 2 variables (0.47) y HMM filtrado (0.25). Se eligieron las reglas.
- Detalles del hallazgo: en datos de 5 minutos la autocorrelación de rezago 1 es positiva (0.09) y se desvanece desde el rezago 2, un efecto de cómo se construyen las barras.

## Decisiones y cuándo se tomaron
- Antes de ver resultados de la estrategia: ranuras de búsqueda del SPEC, mínimo de operaciones por ventana, variables y método de régimen (reglas, umbrales cuantil 90 de volatilidad y R^2 de 0.5), actualización horaria, cierre de la posición al cambiar el régimen, no operar un régimen sin datos suficientes.
- Después de ver resultados: ampliar el rango de m a 10-60 y de max_hold a 10 días y agregar signal_exit_after (tras A y B); agregar el gate de Calmar mayor a 0 (tras la ejecución 1). El efecto del gate medido en train es optimista.

## Limitaciones conocidas del procedimiento
- Cada semana de prueba cierra a la fuerza lo que tenga abierto (56 de 430 salidas en la ejecución 3), lo que castiga el holding largo.
- Con ventanas de un mes la crisis tiene datos suficientes (288 barras o más) en solo 13 de 65 ventanas (14 tienen alguna barra de crisis).

## Hecho después de esta bitácora
- Prueba de diferencias entre regímenes (Welch y bootstrap), sensibilidad de ±20% y curva de costos: ver docs/tables.
- Parámetros congelados en el commit 43aa031 y evaluados una sola vez en prueba: docs/tables/metricas_test.csv.

## Código de la exploración de régimen
El código de K-means y del HMM con probabilidades filtradas, usado para comparar métodos de régimen, se retiró del repositorio al final del proyecto. Se conserva en la etiqueta de git `exploracion-regimenes`.

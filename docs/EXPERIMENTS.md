# Bitácora de experimentos

Todo lo siguiente usa solo `btc_project_train.csv`. El archivo de test no se ha usado.

| # | Qué se corrió | Configuraciones evaluadas | Tiempo | Resultado |
|---|---|---|---|---|
| A | m=5, r=2 con signal_exit_after en {0, 48, 288, nunca} | 4 | segundos | 6,169 a 2,390 operaciones, retorno entre -100% y -99.3% |
| B | m en {10, 20, 40}, sin salida por señal | 3 | segundos | retorno -83%, -49%, -19%; bruto por operación 0.07%, 0.14%, 0.16% |
| 1 | Walk-forward, 65 ventanas x 150 pruebas TPE, sin gate | 9,750 | 161 s | OOS -98.1%, 2,142 operaciones, Sharpe -12.5 |
| 2 | Mismo walk-forward con gate (mejor Calmar de entrenamiento > 0) | 9,750 | 162 s | OOS +5.1%, 354 operaciones, Sharpe 0.31, 51 de 65 semanas operadas |

## Decisiones tomadas después de ver resultados
- Tras A y B: se amplió el rango de m a 10-60 y el de max_hold hasta 10 días, y se agregó el parámetro signal_exit_after.
- Tras la ejecución 1: se agregó el gate (no operar si el mejor Calmar de entrenamiento es menor o igual a 0). Fue motivado por los resultados, así que su efecto medido en train es optimista.

## Pendiente
- Evaluación final sobre test con parámetros congelados (aún no ejecutada).
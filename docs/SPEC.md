# SPEC: estrategia del Lab02 (Equipo 7, nivel B)

## 1. Universo y frecuencia
- Activo: BTCUSDT, barras de 5 minutos, UTC. Datos crudos congelados en `data/`.
- Train: 2022-06-01 a 2023-12-31. Test: 2023-12-31 a 2024-06-03.
- Limpieza (`src/data.py`): se descartan barras con OHLC vacío (3,745 en train, 17 en test) y barras fuera de la rejilla de 5 min (573 y 3, todas planas y sin volumen). No se rellena nada. Se quita del test la barra compartida con el train. Train queda en 162,132 barras y test en 9,366.
- Tramos continuos: un hueco mayor a 6 h abre un tramo nuevo. Train tiene 4 tramos y test 2 (un día de diciembre y 31.5 días de mayo-junio, con un salto de 122 días entre ellos). Los indicadores se calculan dentro de cada tramo y ninguna posición cruza un corte.
- Validación: no hay conjunto aparte. Las semanas de prueba del walk-forward dentro de train hacen de validación, y el test se usa una sola vez al final con los parámetros congelados. (Pendiente: definir si el test se evalúa con parámetros fijos o repitiendo el walk-forward.)

## 2. Features
Tres indicadores de tres familias. Selección: matriz de correlación de las señales con parámetros por defecto en train (sin mirar retornos); este trío tiene la menor correlación media entre pares (0.36).

| Indicador | Familia | Señal x_j en {-1, 0, +1} | Parámetros |
|---|---|---|---|
| Donchian | Tendencia | Dirección de la última ruptura del canal de n barras previas | n_donchian |
| ROC | Momento | Signo del retorno de las últimas n barras | n_roc |
| Keltner | Volatilidad | +1 sobre EMA + k*ATR, -1 bajo EMA - k*ATR, 0 dentro | n_ema, n_atr, k |

No se usa volumen: los datos lo traen vacío o poco confiable. Los tres usan solo información hasta el cierre de la barra t.

## 3. Regla de entrada
Sea L_t = #{j : x_j,t = +1} y S_t = #{j : x_j,t = -1}.

s_t = +1 si L_t >= 2; -1 si S_t >= 2; 0 en otro caso.

- Se opera en la apertura de t+1 con la señal de t.
- Filtro de viabilidad por costo: solo se abre si la distancia al target, como fracción del precio, es al menos 0.5% (el doble del costo ida y vuelta).
- Entrada por nivel: mientras la señal siga activa, se reentra tras cada salida.

## 4. Regla de salida
- Stop-loss: P_in - lado * m * ATR. Take-profit: P_in + lado * r * m * ATR.
- Señal opuesta: cierra la posición solo tras `signal_exit_after` barras (0 la cierra de inmediato; un valor muy grande equivale a no salir por señal).
- Holding máximo: `max_hold` barras.
- Cierre forzado al cierre de la última barra de cada tramo.
- Si la barra abre más allá de un nivel, se llena a la apertura (gap).

## 5. Sizing
q = min( rho * V / (m * ATR), V / (P * (1 + comisión)) )

- rho (`risk_frac`) es el presupuesto de riesgo por operación y V el equity.
- El segundo término impide apalancamiento. Con stops estrechos este tope es el que manda y el riesgo real es menor al presupuestado.

## 6. Costos
- Comisión: 0.125% por operación, en la entrada y en la salida (parámetro fijo del lab). El ida y vuelta es 0.25% del nocional.
- Slippage: 0 en el caso base. Se explora como sensibilidad (0 a 50 bps de ida y vuelta) y con el spread estimado por el modelo de Roll.
- Borrow fee: 0 en el caso base. (Pendiente: justificar y, si procede, barrer.)

## 7. Convenciones
- Señal al cierre de t, ejecución en la apertura de t+1.
- Si stop y target caen en la misma barra, gana el stop.
- Una sola posición abierta a la vez.
- La señal de la barra anterior no se usa si esa barra pertenece a otro tramo.

## 8. Break-even win rate
Con stop s (%), target t = r*s y costo ida y vuelta c = 0.25%:

p* = (s + c) / (s + t) = (s + c) / ((1 + r) * s)

Con r = 2 y la mediana del ATR(14) en train (0.074% del precio):

| m | stop (%) | p* |
|---|---|---|
| 10 | 0.74 | 44.6% |
| 20 | 1.48 | 39.0% |
| 40 | 2.96 | 36.1% |
| 60 | 4.45 | 35.2% |

Umbral que la estrategia debe superar: p* depende de (m, r, ATR) y tiende a 1/(1+r) = 33.3% cuando m crece. Solo aplica a salidas por stop o target; las salidas por holding máximo o por señal lo hacen aproximado.

## 9. Espacio de búsqueda (optimización)
| Parámetro | Rango | Justificación |
|---|---|---|
| m | 10 a 60 | Con m menor a 10 el costo se come el stop |
| r | 1.5 a 4 | Rango usual, sin evidencia propia |
| max_hold | 1 a 10 días | Holdings largos reducen el efecto del costo |
| signal_exit_after | 0, 1 día, nunca | Cubre las tres políticas exploradas |
| Ventanas de indicadores | 12 a 864 barras | Señales más lentas, menos operaciones |
| k (Keltner) | 1 a 3 | Ancho de banda estándar |
| rho | 0.25% a 3% | El lab pide optimizar el tamaño |

Restricción: mínimo de operaciones por ventana. Función objetivo: Calmar.

## 10. Configuraciones exploradas en train antes de optimizar
7 en total: m=5 con signal_exit_after en {0, 48, 288, sin salida por señal} y m en {10, 20, 40} sin salida por señal. Observación: con parámetros por defecto la estrategia pierde en train por el peso de los costos (más de 6,000 operaciones con m=5). Esto cuenta como selección sobre train y se declara en el reporte.
# Validación, selección de hiperparámetros y generalización en test

**Alcance:** revisión del código de la evaluación final ([`final_test.py`](../src/ecog_tpe/final_test.py)) y de las etapas que la preceden (búsqueda y verificación), con la diferencia entre las tres cifras que produce el proyecto. Complementa §4.2, §5.2, §6 y §10 de la [metodología](metodologia.md).

**Regla corta:** una cifra que se usó para decidir algo deja de ser una estimación del rendimiento. La validación y la selección son *decisiones*; el test es la *medida*.

---

## 1. Las tres cifras en una tabla

| | Rendimiento de validación | Selección de hiperparámetros | Generalización en test |
|---|---|---|---|
| **Pregunta** | ¿Qué tan bien predice este entrenamiento un bloque de desarrollo cuyos datos no usó para ajustar pesos? | ¿Qué configuración (arquitectura, k, L, f, LR, L2) nos quedamos? | ¿Qué tan bien predice la configuración elegida datos que no intervinieron en ninguna decisión? |
| **Datos** | El bloque validado del fold (B1…B5): 7750 columnas = 77.5 s | Los cinco folds del desarrollo (los mismos 400 s, una y otra vez) | `test_data` / `test_dg`: 19 750 columnas = 197.5 s |
| **Cifra** | `val_pearson_macro` por época; `best_val_pearson_macro` = su máximo | J_s (`objective_cv_best`) y `verification_cv_best` | `test_pearson_macro` |
| **¿Decide algo?** | Sí: qué época se conserva (early stopping) | Sí: qué trials son finalistas y cuál es la elegida | **No.** Después del test no queda nada por decidir |
| **¿Estima el rendimiento en datos nuevos sin sesgo?** | No: es un máximo sobre épocas | No: es un máximo sobre trials, siempre en los mismos folds | Sí, porque todo quedó fijado antes de abrirlo |

---

## 2. Qué es cada cifra en el código

### 2.1. Rendimiento de validación: dentro de un entrenamiento

Al final de cada época, [`train_fold`](../src/ecog_tpe/training.py#L209) predice las 119 ventanas del bloque validado, reconstruye la trayectoria con overlap-add y promedia el Pearson de los cinco dedos: `val_pearson_macro`. Si es un nuevo máximo, [`EarlyStopping.update`](../src/ecog_tpe/training.py#L210) copia los pesos; al terminar se [restauran](../src/ecog_tpe/training.py#L234) los de la mejor época.

El score del fold es por tanto

$$q = \max_{1 \le e \le T} r_e ,$$

y eso tiene dos consecuencias:

- **El bloque validado no es ciego.** No aporta gradientes, pero elige la época. Con 25–50 puntos de una curva ruidosa, el máximo queda por encima del nivel típico de la curva solo por ruido.
- **Es una estimación interna legítima**, útil para comparar trials con la misma regla. Lo incorrecto es leerla como rendimiento final.

### 2.2. Selección de hiperparámetros: entre entrenamientos

Hay tres filtros sucesivos, todos sobre el desarrollo:

1. **TPE.** Cada trial vale J_s(θ) = media de los cinco q ([`search.py:268`](../src/ecog_tpe/search.py#L268)). Tras los 10 warmups, TPE usa los J_s anteriores para proponer los siguientes θ: el objetivo dirige la búsqueda.
2. **Finalistas.** Mejor J_s dentro de A5b y dentro de A5 entre los 32 trials COMPLETE ([`verification.py:50`](../src/ecog_tpe/verification.py#L50)).
3. **Elegida.** Cada finalista se reentrena con dos semillas nuevas en los mismos cinco folds; `verification_cv_best` es la media de las semillas de la media de los folds de q. La elegida es la de mayor valor ([`final_test.py:208`](../src/ecog_tpe/final_test.py#L208), mismo criterio que [`reporting.py:216`](../src/ecog_tpe/reporting.py#L216)).

**Por qué el valor del ganador es optimista.** Al tomar el máximo de muchas estimaciones ruidosas, el elegido tiende a ser el que tuvo ruido favorable: E[máx X_i] ≥ máx E[X_i]. Con 32 trials, ese sesgo puede ser del mismo orden que las diferencias reales entre trials.

**Qué corrige la verificación y qué no.** El finalista se eligió con la semilla 42; la verificación usa 1042 y 2042, así que elimina la parte del sesgo que dependía de *esa semilla*. No elimina la parte que depende de *esos datos*: los cinco bloques son los mismos. Por eso la verificación mide estabilidad del entrenamiento, no generalización. Además, elegir entre los dos finalistas con `verification_cv_best` es otra selección (pequeña, con dos candidatos), así que el valor de la elegida conserva algo de optimismo.

### 2.3. Generalización en test

[`run_final_test`](../src/ecog_tpe/final_test.py#L366) hace, por sujeto:

1. **Congela el plan** ([`freeze_plan`](../src/ecog_tpe/final_test.py#L186)): escribe `final_test/<S>/plan.json` con los dos finalistas, su `verification_cv_best`, la elegida, las semillas (3042, 4042, 5042) y el máximo de épocas. Esto ocurre antes de leer el test.
2. **Prepara los datos** ([`SubjectTestData`](../src/ecog_tpe/final_test.py#L126)): calcula las características del fold B5, con clip y escaladores ajustados solo en B1–B4, y después carga el test y lo *transforma* con esos parámetros ([`eval_segment`](../src/ecog_tpe/final_test.py#L106)).
3. **Entrena y predice** ([`run_test_model`](../src/ecog_tpe/final_test.py#L250)): entrena desde cero con B1–B4, para con B5 (máximo 70 épocas), restaura el mejor checkpoint y predice el test **una sola vez** ([líneas 296–299](../src/ecog_tpe/final_test.py#L296-L299)).

Son 18 modelos (3 sujetos × 2 finalistas × 3 semillas). El resultado principal es la **elegida** de cada sujeto, media de sus tres semillas, y después la media de los tres sujetos.

Es la única cifra sin sesgo de selección porque nada de lo que la produce vio el test: ni la arquitectura, ni los hiperparámetros, ni la época del checkpoint, ni el clip, ni los escaladores.

Ojo: los modelos de test son **entrenamientos nuevos**, no los pesos de ningún trial ni de la verificación (esos no se guardan). Lo que se evalúa es la configuración más su receta de entrenamiento, no una red concreta de la búsqueda.

---

## 3. Qué datos intervienen en cada decisión

```text
Desarrollo: train_data / train_dg (400 s)                    Test: test_data / test_dg (200 s)
 B1   B2   B3   B4   B5
 │  5 folds: 4 bloques entrenan, 1 valida
 ├─ cada época:   val_pearson ─────────────► early stopping (elige la época)
 ├─ cada trial:   J_s = media de 5 máximos ─► TPE (propone el siguiente θ)
 ├─ 32 trials ──────────────────────────────► finalistas (mejor A5b, mejor A5)
 ├─ 2 semillas × 5 folds ───────────────────► elegida → plan.json (congelado)
 │
 └─ modelos de test: B1–B4 entrenan, B5 para ─► mejor checkpoint ─► una predicción ─► r_test
                                                                      (de aquí no sale ninguna flecha)
```

| Decisión | Datos con que se toma | Código |
|---|---|---|
| Canales excluidos | Lista fija (EDA y ablaciones) | [`configs/tpe.yaml`](../configs/tpe.yaml) |
| Clip robusto | Bloques de train del fold (B1–B4 en test) | [`preprocessing.py:237`](../src/ecog_tpe/preprocessing.py#L237) |
| RobustScaler y MinMax | Columnas de train del fold | [`preprocessing.py:312`](../src/ecog_tpe/preprocessing.py#L312), [`:323`](../src/ecog_tpe/preprocessing.py#L323) |
| Época del checkpoint | Bloque validado (B5 en test) | [`training.py:210`](../src/ecog_tpe/training.py#L210) |
| Arquitectura e hiperparámetros | J_s en los 5 folds | [`search.py:268`](../src/ecog_tpe/search.py#L268), [`verification.py:50`](../src/ecog_tpe/verification.py#L50) |
| Elegida | `verification_cv_best` | [`final_test.py:208`](../src/ecog_tpe/final_test.py#L208) |
| Máximo de 70 épocas en test | Configuración | [`tpe.yaml:125`](../configs/tpe.yaml#L125) |
| — | Test: no decide nada | [`final_test.py:296-299`](../src/ecog_tpe/final_test.py#L296-L299) |

---

## 4. Cómo leer las columnas de `results/summary/`

| Archivo · columna | Qué es | Tipo de cifra |
|---|---|---|
| `tpe_folds.csv` · `best_pearson_macro` | q de un fold de un trial | Validación (máximo por época) |
| `tpe_trials.csv` · `objective_cv_best` | J_s del trial | Selección |
| `tpe_verification.csv` · `best_pearson_macro` | q de un finalista con semilla nueva | Validación (estabilidad) |
| `tpe_selected_models.csv` · `search_cv_best` | J_s del finalista con la semilla de búsqueda | Selección (la más optimista) |
| `tpe_selected_models.csv` · `verification_cv_best` | Media de las dos semillas de verificación | Selección (menos optimista) |
| `tpe_test.csv` · `val_pearson_macro` | Máximo en B5 del modelo de test | Validación en el conjunto de parada |
| `tpe_test.csv` · `test_pearson_macro` | Pearson en test de una semilla | Generalización |
| `tpe_test_summary.csv` · fila `global` / `elegida` | Media de la elegida en los tres sujetos | **Resultado principal** |

Lo esperable es este orden: `search_cv_best` ≥ `verification_cv_best` ≳ `test_pearson`. No es una ley, pero es lo habitual.

---

## 5. Por qué el test suele quedar por debajo

La diferencia entre `verification_cv_best` y el test mezcla varias fuentes:

1. **Máximo sobre épocas:** el optimismo del early stopping.
2. **Máximo sobre trials y entre las dos arquitecturas:** el optimismo de la selección.
3. **Otros datos:** la verificación promedia cinco bloques del desarrollo; el test es un único segmento de 197.5 s, con sus propios movimientos y con la deriva que tenga la señal.
4. **Otro objetivo:** `test_dg` viene suavizado a 1000 Hz, mientras que `train_dg` está en escalones y se reconstruye con spline; la fase de esa interpolación es desconocida (hasta ≈ 20 ms, §10.12 de la metodología).
5. **Otra receta:** los modelos de test entrenan hasta 70 épocas, no 50, y solo con B1–B4, como el fold B5; no son un promedio de folds.

Una brecha **no es por sí sola evidencia de sobreajuste**: es sobre todo el precio de haber usado la validación para decidir. Si la otra finalista supera a la elegida en test, se reporta tal cual. No se corrige.

**Comprobación útil.** El `val_pearson_macro` de los modelos de test (B5, semillas 3042–5042) debería caer en el rango de los q del fold B5 en la verificación. Si es así, el entrenamiento de test reprodujo bien el de desarrollo. Las diferencias de semilla y de máximo de épocas impiden una igualdad exacta.

---

## 6. Revisión del código de evaluación final

Ejecuté `pytest tests/test_final_test.py` en el entorno `tpe_ecog`: **5 pruebas pasan** (42 s en CPU). No encontré errores que cambien resultados.

### 6.1. Lo que el código garantiza

- **Plan congelado antes de abrir el test.** `run_final_test` llama a `freeze_plan` ([línea 366](../src/ecog_tpe/final_test.py#L366)) antes de construir `SubjectTestData` ([línea 371](../src/ecog_tpe/final_test.py#L371)), que es el único lugar que llama a `load_test`. `plan.json` se escribe una vez; al relanzar solo se lee, así que no se vuelve a elegir.
- **Ningún parámetro se ajusta con el test.** El clip sale de los bloques de train ([`preprocessing.py:237`](../src/ecog_tpe/preprocessing.py#L237)), los escaladores de las columnas de train y `eval_segment` solo los aplica ([línea 115](../src/ecog_tpe/final_test.py#L115)). La prueba [`test_test_data_never_changes_what_is_fitted`](../tests/test_final_test.py#L75) altera el test por completo (×100, +10⁴, un pico de 10⁸) y comprueba que todo lo ajustado y las características de desarrollo quedan idénticos bit a bit.
- **B5 no entrena.** La misma prueba verifica que el bloque 4 (B5) no aparece en las ventanas de entrenamiento.
- **El early stopping nunca mira el test.** `train_fold` solo recibe el fold; el test se predice después, con el checkpoint ya restaurado, y una sola vez. No hay curva de test por época en `history.csv`.
- **Elección coherente con el reporte.** `freeze_plan` y `selected_models` usan el mismo criterio y el mismo desempate (el primero en el orden A5b, A5).
- **Comparación apareada.** A5b y A5 usan las mismas semillas dentro de cada sujeto.
- **Canales dañados en test.** El cociente pico/típico solo emite un aviso ([línea 169](../src/ecog_tpe/final_test.py#L169)); la lista de exclusión no cambia.

### 6.2. Observaciones

1. **El máximo de 70 épocas solo se aplica en test** ([`tpe.yaml:125`](../configs/tpe.yaml#L125), [`final_test.py:375`](../src/ecog_tpe/final_test.py#L375)). Los hiperparámetros se eligieron y verificaron con 50, así que los modelos de test siguen una receta distinta de la validada. No es una fuga si se decidió con información de desarrollo y antes de abrir el test, pero tiene dos consecuencias:
   - La metodología (§6.5) y el registro de cambios del plan (§0) dan el valor, pero **no el motivo**. Conviene añadir una línea con la razón (por ejemplo, cuántas corridas de verificación pararon por `max_epochs`) y la constancia de que se fijó antes del test.
   - `verification_cv_best` describe la receta de 50 épocas. Para comparar validación y test con la misma receta, usa el `val_pearson_macro` de los propios modelos de test.

2. **`val_pearson_macro` en `tpe_test.csv` es la cifra que detuvo el entrenamiento** ([`reporting.py:258`](../src/ecog_tpe/reporting.py#L258)). Es el máximo en B5, el mismo bloque que eligió la época, y B5 fue además uno de los cinco folds con que se seleccionaron los hiperparámetros. Sirve para comprobar el entrenamiento; no debe presentarse como «validación» al lado del test. En la tesis, mejor llamarla «Pearson en B5 (conjunto de parada)».

3. **`test_scores.png` pone la verificación (×) junto al test** ([`reporting.py:540`](../src/ecog_tpe/reporting.py#L540)). Está bien como referencia visual, pero el pie de figura debería decir que la × es una métrica de selección sobre cinco folds y que una brecha es esperable (§5).

4. **El resultado principal es la elegida, no la mejor de las dos en test.** El código lo hace bien (`plan["chosen"]`, fila `elegida` de `final_test_summary`). El riesgo está en la redacción: si el texto destaca «A5 obtuvo el mejor resultado en test» y usa esa cifra como principal, se está seleccionando con el test. A5b frente a A5 en test es una comparación descriptiva.

5. **El monitor muestra el test en vivo** (`_print_final_test`). Después del plan no queda ninguna decisión, así que no es un problema. Pero si tras ver el test aparece un error de software y hay que repetir algo, regístralo como desviación en «Notas de ejecución» del README y conserva ambos resultados.

6. **Hubo contacto con el test, pero sin usar el rendimiento.** Dos decisiones se tomaron mirando archivos de test: la convención del objetivo `test_dg[::10]` ([línea 93](../src/ecog_tpe/final_test.py#L93)), tras ver que `test_dg` viene suavizado, y la revisión de canales dañados en `test_data`. Ninguna usa predicciones de modelos, así que no favorecen a ninguna configuración. Ya están declaradas en §10 (puntos 7 y 12); conviene mantener esa redacción en la tesis.

7. **La desviación entre semillas no es un intervalo de confianza del test.** Las tres semillas miden la variabilidad del entrenamiento sobre los mismos 197.5 s; no capturan la incertidumbre de que el test sea un solo segmento, muy autocorrelacionado. Si hace falta un intervalo, un bootstrap por bloques (por ejemplo de 10 s) sobre `predictions.npz` lo da sin reentrenar. Es opcional.

---

## 7. Propuesta de redacción para la tesis

> Los hiperparámetros y la arquitectura se seleccionaron exclusivamente mediante validación cruzada por bloques sobre los 400 s de desarrollo. Las cifras de esa etapa (el objetivo J_s y su verificación con semillas nuevas) son métricas de selección: incluyen el máximo sobre épocas y sobre 32 configuraciones, por lo que sobreestiman el rendimiento esperado en datos nuevos. Antes de abrir el conjunto de test se fijaron, para cada sujeto, la configuración elegida y el protocolo de entrenamiento (B1–B4 para entrenar, B5 para el early stopping, un máximo de 70 épocas y tres semillas). El rendimiento de generalización reportado es el Pearson sobre los 197.5 s evaluables del test oficial, obtenido con una única predicción por modelo; ningún resultado de test se usó para modificar decisiones.

# Metodología

**Proyecto:** decodificación continua de la flexión de cinco dedos a partir de ECoG con dos familias de encoder–decoder 1D (A5b y A5) y optimización conjunta de hiperparámetros con TPE.
**Datos:** BCI Competition IV, dataset 4 (S1, S2, S3).
**Alcance:** lo implementado en este repositorio: lectura de datos, preprocesamiento, modelo, entrenamiento con early stopping, búsqueda con Optuna TPE, verificación de finalistas y reportes. No incluye evaluación en test ni reentrenamiento final sobre todo el desarrollo.
**Relación con otros documentos:** el [plan de implementación](plan_implementacion_TPE_ECoG.md) es la especificación; [`configs/tpe.yaml`](../configs/tpe.yaml) es la única fuente de parámetros; el [README](../README.md) describe el uso.

Todas las decisiones se toman solo con datos de desarrollo. El lector abre únicamente `train_data` y `train_dg`; los datos y etiquetas de test no se cargan en ninguna etapa.

---

## 1. Datos

### 1.1. Archivos

Por sujeto se lee `subX_comp.mat` con `variable_names=["train_data", "train_dg"]`. El lector comprueba la forma (400 000 muestras × canales originales; guante 400 000 × 5) y que todos los valores sean finitos.

| Sujeto | Canales originales | Excluidos (base cero) | Canales usados | Muestras (1 kHz) | Duración |
|---|---:|---|---:|---:|---:|
| S1 | 62 | 54 | 61 | 400 000 | 400 s |
| S2 | 48 | 20, 37 | 46 | 400 000 | 400 s |
| S3 | 64 | 49 | 63 | 400 000 | 400 s |

- Los canales excluidos se **eliminan** (no se anulan) antes de cualquier otra operación, así que no participan en la referencia común ni en la entrada del modelo.
- La lista es fija (`data.subjects.*.excluded` del YAML) y no se decide dentro de los folds.
- Las unidades eléctricas y de flexión se conservan como vienen en el archivo.

### 1.2. Almacenamiento del guante

El guante mide a 25 Hz, pero `train_dg` está guardado a 1000 filas por segundo con **retención de orden cero de 40 filas**. Las pruebas comprueban con los datos reales que todos los cambios de valor caen en filas múltiplo de 40 en los tres sujetos. Cada bloque de 80 s tiene 2000 muestras efectivas; el instante de cada una es la primera fila de su grupo de 40.

### 1.3. Bloques cronológicos

El desarrollo de cada sujeto se divide en cinco bloques de 80 s (80 000 muestras) **antes de cualquier transformación**: B1 = 0–80 s, …, B5 = 320–400 s. El bloque es la unidad independiente de todo el filtrado: ningún filtro cruza una frontera y no se concatenan segmentos.

---

## 2. Preprocesamiento

La cadena se identifica como `P0` en el código y es la misma para todos los trials. TPE no puede modificarla.

### 2.1. Cadena

| # | Operación | Parámetros | Resultado |
|---|---|---|---|
| 1 | Lectura | `train_data`, `train_dg` en float64 | 400 000 muestras por sujeto |
| 2 | Exclusión de canales | S1-54, S2-20, S2-37, S3-49 | 61 / 46 / 63 canales |
| 3 | Bloques | 5 × 80 s | 5 bloques de 80 000 muestras |
| 4 | Limitación robusta de amplitud | Por canal: mediana ± 25 · (1.4826 · MAD), mínimo de MAD 10⁻⁶; ajustada con los cuatro bloques de train del fold y aplicada a los cinco | Fracción recortada registrada en train y validación (`clip_fraction_*`) |
| 5 | CMR | Resta de la mediana espacial instantánea de los canales conservados | — |
| 6 | Pasabanda | FIR Hamming de fase cero (`mne.filter.create_filter`), 40–200 Hz, transiciones de 10 y 20 Hz (−6 dB en 35 y 210 Hz) | 331 coeficientes |
| 7 | Notch | Band-stop FIR en 60, 120 y 180 Hz con las convenciones de `mne.filter.notch_filter` (ancho f/200, transición de 2 Hz por lado) | 1651 coeficientes |
| 8 | Morlet | 40 frecuencias logarítmicas entre 40 y 200 Hz, 7 ciclos, `zero_mean=True`; potencia \|W\|² por canal, sin logaritmo | σt de 28 a 6 ms |
| 9 | Antialias y decimación | Kaiser: paso 35 Hz, rechazo 50 Hz, 60 dB (corte 42.5 Hz); después se toma una de cada 10 muestras | 243 coeficientes; 8000 columnas a 100 Hz por bloque |
| 10 | Bordes | Se descartan 125 columnas en cada extremo del bloque | Región válida [125, 7875): **7750 columnas** |
| 11 | Guante | Muestras efectivas (fase 0) → spline cúbico *not-a-knot* por bloque, evaluado en la rejilla de 100 Hz | 5 × 8000 por bloque |
| 12 | Retardo | X(t) ↔ Y(t + 2 columnas) = **20 ms** | Objetivo validado en las columnas [127, 7877) |
| 13 | Escaladores | `RobustScaler(quantile_range=(10, 90), unit_variance=True)` por columna canal × frecuencia; `MinMaxScaler` por dedo; ambos ajustados con las columnas válidas de train del fold | Validación solo transformada, sin recorte |
| 14 | Ventanas | 256 columnas (2.56 s); train con paso 1 dentro de la región válida de cada bloque; validación con paso 64 más una última ventana | §2.2 |

**Detalles de implementación.**
- Todo el DSP se calcula en float64, por bloque, en la GPU cuando está disponible. Las convoluciones se hacen por FFT en modo *same* con relleno de ceros; los bordes contaminados se descartan en el paso 10.
- El borde de 125 columnas es el semisoporte de la cadena: 825 (notch) + 165 (pasabanda) + 121 (antialias) + 139 (wavelet de 40 Hz) = 1250 muestras a 1 kHz. La configuración se rechaza si el valor del YAML no coincide con el calculado.
- El spline del guante no cruza bloques. Las 3 últimas columnas de cada bloque quedan después del último nudo (79.96 s), pero caen en la zona de borde, que no se usa.
- Las características se guardan escaladas en float32. Cada fold recalcula la cadena completa, porque la limitación de amplitud y los escaladores dependen de su entrenamiento.
- Las características de los cinco folds de un sujeto se calculan una vez por proceso (≈ 7 s por fold en GPU) y viven en RAM y GPU (≈ 0.4 GB por fold en S3). Todos los trials las reutilizan; no se escriben a disco.

### 2.2. Folds (iguales en los tres sujetos)

Cada fold valida un bloque y entrena con los otros cuatro.

| Fold | Bloques de train purgados | Columnas de ajuste de escaladores | Ventanas de train | Batches por época | Ventanas de validación | Columnas validadas |
|---|---|---:|---:|---:|---:|---:|
| B1 | B2 (al inicio) | 30 744 | 29 724 | 232 | 119 | 7750 |
| B2, B3, B4 | El anterior (al final) y el siguiente (al inicio) | 30 488 | 29 468 | 230 | 119 | 7750 |
| B5 | B4 (al final) | 30 744 | 29 724 | 232 | 119 | 7750 |

Es validación cruzada por bloques con purga, no *walk-forward*: el entrenamiento incluye bloques anteriores y posteriores al validado. Las ventanas de train nunca cruzan fronteras entre bloques ni entran en el bloque validado.

### 2.3. Regla de purga

La purga es de **256 columnas (2.56 s)**, solo en entrenamiento y sumada a los bordes de 125 columnas:
- el bloque de train inmediatamente anterior al validado usa las columnas [125, 7619);
- el inmediatamente posterior usa [381, 7875);
- los demás bloques de train y el bloque validado no se purgan.

---

## 3. Modelo

### 3.1. Familias A5b y A5

Ambas son instancias de `EncoderDecoder1D` y solo difieren en el upsampling del decoder.

| Componente | A5b | A5 |
|---|---|---|
| Entrada | (B, C, 40, 256) reorganizada como (B, C·40, 256) | Igual |
| Stem | Bloque convolucional C·40 → f, kernel 3 | Igual |
| Bloque convolucional | Conv1d sin sesgo con padding *same* → LayerNorm sobre canales en cada instante → GELU → Dropout 0.1 | Igual |
| Encoder (L niveles) | Bloque dilatado → MaxPool1d(2); el *skip* se toma antes del pooling | Igual |
| Decoder (L niveles) | **Interpolación lineal** hasta la longitud del *skip* → concatenación → bloque con el kernel espejo y dilatación 1 | **ConvTranspose1d depthwise** (kernel 4, paso 2, padding 1, `groups=C`, sin sesgo) → concatenación → bloque |
| Cabeza | Conv1d(f → 5, kernel 1), sin activación final | Igual |
| Salida | (B, 5, 256) | Igual |

- La interpolación de A5b es una implementación vectorizada con la misma fórmula de índices y pesos que `F.interpolate(mode="linear", align_corners=False)`. Las pruebas comprueban la igualdad de valores y gradientes en todas las longitudes del decoder.
- Si una concatenación no coincide en longitud, el modelo lanza un error: no hay recortes ni proyecciones ocultas.

### 3.2. Parametrización por (k, L, f)

Para cada trial, `space.architecture_spec` construye la red a partir de la base de la familia:

```python
channels          = [f, f, 2*f, 4*f, 8*f, 8*f][:L + 1]   # stem + L niveles
encoder_channels  = channels[1:]
encoder_kernels   = [7, 7] + [k] * (L - 2)
encoder_dilations = [1, 2, 3, 1, 2][:L]
decoder_channels  = encoder_channels[::-1]
decoder_kernels   = encoder_kernels[::-1]
```

Ejemplo con k = 5, L = 5 y f = 64: stem de 64 canales, encoder [64, 128, 256, 512, 512] con kernels [7, 7, 5, 5, 5] y dilataciones [1, 2, 3, 1, 2], cuello de botella de 512 canales y 8 columnas, y decoder [512, 512, 256, 128, 64].

**Inicialización.** La de PyTorch por defecto, después de fijar la semilla del fold (§4.4). Dos trials con la misma arquitectura parten de los mismos pesos en un fold dado. Cada fold empieza con un modelo y un optimizador nuevos; no se reutilizan pesos entre folds ni trials.

### 3.3. Capacidad de las 16 combinaciones

Antes del primer estudio se midió una actualización completa (forward, backward y paso de Adam) con batch 128 en las 16 combinaciones discretas, con el sujeto de más canales (S3, 2520 entradas) y un tensor del tamaño de X de un fold reservado en la GPU. El tiempo es la mediana de 3 pasos tras uno de calentamiento; la época estimada multiplica por 230.8 batches.

| k-L-f | Parámetros A5b | Parámetros A5 | s/época A5b | s/época A5 | Memoria pico A5b / A5 (GB) |
|---|---:|---:|---:|---:|---|
| k5-L4-f64 | 5 427 909 | 5 433 541 | 6.03 | 2.87 | 1.13 / 1.29 |
| k5-L4-f96 | 11 846 693 | 11 855 141 | 8.75 | 3.94 | 1.40 / 1.44 |
| k5-L5-f64 | 9 362 117 | 9 369 797 | 5.70 | 3.34 | 1.22 / 1.24 |
| k5-L5-f96 | 20 697 125 | 20 708 645 | 8.06 | 4.64 | 1.58 / 1.61 |
| k7-L4-f64 | 7 197 381 | 7 203 013 | 5.24 | 3.05 | 1.22 / 1.24 |
| k7-L4-f96 | 15 828 005 | 15 836 453 | 7.41 | 4.22 | 1.48 / 1.49 |
| k7-L5-f64 | 12 704 453 | 12 712 133 | 5.89 | 3.56 | 1.27 / 1.29 |
| k7-L5-f96 | 28 217 381 | 28 228 901 | 8.46 | 5.05 | 1.69 / 1.72 |

- **Las 16 caben** con batch 128 y float32, así que el espacio no se recortó. Batch y precisión son comunes a todos los trials.
- A5 tarda entre el 45 % y el 60 % de lo que tarda A5b con la misma (k, L, f).
- En S1 y S2 el stem tiene menos entradas (2440 y 1840) y, por tanto, algo menos de parámetros.
- El tiempo real por época de A5b k5-L5-f64, con validación incluida, fue de 5.6–6.3 s en la campaña.

---

## 4. Entrenamiento

### 4.1. Pérdida, optimizador y batches

| Componente | Valor |
|---|---|
| Pérdida | 0.5 · MSE + 0.5 · (1 − coseno temporal). El coseno se calcula a lo largo del tiempo para cada dedo y ventana (eps 10⁻⁸) y se promedia sobre dedos y batch |
| Optimizador | **Adam con L2 acoplada** (`decoupled_weight_decay=False`, no AdamW); LR y L2 los propone TPE |
| `betas` / `eps` | (0.9, 0.999) / 10⁻⁸ |
| Scheduler | Ninguno: LR constante |
| Batch | 128 ventanas, `drop_last=True`, barajadas en cada época |
| Precisión | Tensores float32 con TF32 permitido en convoluciones y matmul |

El orden de los batches lo genera un `torch.Generator` con la semilla del fold, así que es el mismo en todos los trials de ese fold. Las ventanas se cortan como vistas de los bloques del fold residentes en la GPU, sin copiar el tensor completo.

### 4.2. Validación y métrica

Al final de **cada época**:
1. Se infiere en modo `eval` sobre las 119 ventanas del bloque validado (paso 64 más una última ventana).
2. Se reconstruye la trayectoria con overlap-add de pesos uniformes, exigiendo que todas las columnas queden cubiertas.
3. Se calcula Pearson por dedo sobre las 7750 columnas validadas y se promedia entre los cinco dedos: `val_pearson_macro`.

- El Pearson por época se calcula en la escala MinMax. Como el MinMax es una transformación afín creciente por dedo, el valor es idéntico al de unidades originales.
- La pérdida de validación por ventanas también se registra, pero no interviene en ninguna decisión.
- **Evaluación inválida.** Si una predicción no es finita o es constante (desviación estándar ≤ 10⁻¹⁰ · max(1, máx\|x\|)), el Pearson de ese dedo es NaN. En ese caso, o si la pérdida de entrenamiento no es finita, se registra la fila de la época y el fold falla. Ningún dedo se descarta en silencio.

**Métricas finales del fold.** Se calculan con el mejor checkpoint restaurado, en unidades originales del guante (inversa del MinMax): Pearson por dedo y macro, MSE, RMSE y el coeficiente de concordancia de Lin (CCC), por dedo y macro.

### 4.3. Early stopping con el mejor checkpoint

| Parámetro | Valor |
|---|---|
| Monitor | `val_pearson_macro`, modo `max` |
| `min_delta` | 0: cualquier nuevo máximo estricto cuenta como mejora; un empate no |
| `patience` | 10 validaciones (una por época) |
| `min_epochs` | 15 |
| `max_epochs` | 50 |
| Checkpoint | Pesos de la época con mayor Pearson, copiados en memoria |
| Pruning | Ninguno (`NopPruner`) |

**Semántica.**
1. Las épocas 1–15 guardan checkpoints, pero no pueden detener el entrenamiento.
2. Al terminar la época 15, el contador de paciencia se pone a cero.
3. Desde la época 16, cada validación sin un nuevo máximo suma uno al contador; un nuevo máximo lo reinicia.
4. Se detiene cuando el contador llega a 10 (`patience_exhausted`) o en la época 50 (`max_epochs`). La primera parada posible es la época **25**.
5. Al terminar se restauran los pesos de la mejor época y se vuelve a validar. Si el Pearson recalculado difiere en más de 10⁻⁴ del mejor observado, se emite un aviso.

**Score del fold.** Es el mejor `val_pearson_macro` observado hasta la parada. Por fold se registran `best_epoch`, `epochs_trained` y `stop_reason`; `epochs_trained` no es la época del checkpoint elegido.

### 4.4. Semillas

| Uso | Fórmula | S1 (B1…B5) | S2 | S3 |
|---|---|---|---|---|
| Búsqueda | 42 + 5 · sujeto + fold | 42–46 | 47–51 | 52–56 |
| Verificación, semilla 1 | 1042 + 5 · sujeto + fold | 1042–1046 | 1047–1051 | 1052–1056 |
| Verificación, semilla 2 | 2042 + 5 · sujeto + fold | 2042–2046 | 2047–2051 | 2052–2056 |
| Sampler TPE | Fija por sujeto | 2026 | 2027 | 2028 |

Antes de construir cada modelo se reinician los generadores de Python, NumPy y PyTorch (CPU y CUDA) con la semilla del fold. La semilla de búsqueda es la misma en todos los trials, así que el orden de batches y el estado inicial de los generadores son comunes (y la inicialización, cuando la arquitectura coincide): las comparaciones entre trials quedan emparejadas por fold.

### 4.5. Determinismo

- `cudnn.deterministic = True`, `cudnn.benchmark = False` y `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- `torch.use_deterministic_algorithms(True, warn_only=True)`: las operaciones sin implementación determinista solo emiten un aviso.
- Por eso la reproducibilidad bit a bit en GPU no está garantizada; la campaña usa una sola clase de GPU.

### 4.6. Artefactos por fold

Cada fold escribe `status.json` (latido y estado), `history.csv` (una fila por época), `metrics.json` (métricas del mejor checkpoint) y `predictions.npz` (`y_true`, `y_pred` en unidades originales, tiempo de cada columna y mejor época).

**No se guardan pesos** en ninguna corrida, ni de búsqueda ni de verificación. El mejor checkpoint existe solo en memoria mientras dura el entrenamiento del fold.

---

## 5. Búsqueda con TPE

### 5.1. Espacio

| Parámetro | Optuna | Dominio |
|---|---|---|
| `architecture_id` | `suggest_categorical` | {A5b, A5} |
| `learning_rate` | `suggest_float(log=True)` | [5 × 10⁻⁵, 5 × 10⁻⁴] |
| `weight_decay` (L2) | `suggest_float(log=True)` | [10⁻⁷, 10⁻³] |
| `kernel_size` (k) | `suggest_categorical` | {5, 7} |
| `encoder_levels` (L) | `suggest_categorical` | {4, 5} |
| `base_filters` (f) | `suggest_categorical` | {64, 96} |

Son 16 combinaciones discretas más dos dimensiones continuas, optimizadas **conjuntamente**: LR y L2 se proponen junto con la arquitectura, no se fijan en una arquitectura para heredarlos en otra. Dropout, normalización, preprocesamiento, retardo, batch, scheduler y pérdida quedan fuera del espacio.

### 5.2. Objetivo

Para el sujeto s y el trial θ, con r_{s,b,e} el `val_pearson_macro` del fold b en la época e y T_{s,b}(θ) la época de parada:

$$
q_{s,b}(\theta)=\max_{1\le e\le T_{s,b}(\theta)} r_{s,b,e},
\qquad
J_s(\theta)=\frac{1}{5}\sum_{b=1}^{5} q_{s,b}(\theta).
$$

Optuna maximiza J_s, la **media no ponderada de los mejores Pearson de los cinco folds**. Cada trial entrena cinco modelos desde cero, uno por fold, en orden B1 → B5.

### 5.3. Estudio y sampler

| Campo | Valor |
|---|---|
| Estudios | Uno independiente por sujeto (`joint_tpe_S1`, `joint_tpe_S2`, `joint_tpe_S3`) |
| Sampler | `TPESampler(multivariate=True, n_startup_trials=10, constant_liar=False)`, semilla 2026/2027/2028 |
| Pruner | `NopPruner` |
| Ejecución | Secuencial (un trial a la vez), dirección `maximize` |
| Presupuesto | **32 trials COMPLETE por sujeto**; los FAIL no cuentan |
| Versión | `optuna==5.0.0` |

### 5.4. Warmups equilibrados

Al crear cada estudio se encolan **una sola vez** cinco pares de hiperparámetros, cada uno con A5b y con A5. Son 10 de los 32 trials, en el orden p1-A5b, p1-A5, p2-A5b, …

| Par | k | L | f | LR | L2 |
|---|---:|---:|---:|---:|---:|
| 1 (referencia) | 5 | 5 | 64 | 8.42 × 10⁻⁵ | 10⁻⁶ |
| 2 | 7 | 5 | 64 | 10⁻⁴ | 10⁻⁵ |
| 3 | 5 | 4 | 96 | 2 × 10⁻⁴ | 10⁻⁶ |
| 4 | 7 | 4 | 96 | 7 × 10⁻⁵ | 10⁻⁴ |
| 5 | 5 | 5 | 96 | 3 × 10⁻⁴ | 10⁻⁷ |

- Los warmups ocupan exactamente el arranque de TPE (`n_startup_trials=10`). Desde el trial 10, los **22 trials restantes** los propone TPE con esas observaciones; no hay trials aleatorios de arranque.
- El par 1 es la configuración de referencia sin optimizar, entrenada con la misma política que el resto: sirve de comparador interno (§6.4).
- TPE puede repetir una estructura discreta con otro LR o L2. Una repetición exacta de los seis valores no se impide; solo se anota (`duplicate_of`).

### 5.5. Presupuesto

| Etapa | Por sujeto | Entrenamientos (3 sujetos) |
|---|---|---:|
| Warmups (5 pares × A5b/A5) | 10 trials × 5 folds | 150 |
| TPE guiado | 22 trials × 5 folds | 330 |
| Verificación (2 finalistas × 2 semillas) | 2 × 2 × 5 | 60 |
| **Total** | | **540** |

### 5.6. Fallos, persistencia y reanudación

**Fallos.**
- Un trial con NaN, OOM u otro error queda `FAIL` con su causa (`error` en el estudio y `error.txt` en su carpeta). No recibe un score inventado y no cuenta para los 32.
- Un warmup fallido se vuelve a encolar con los mismos parámetros, hasta 3 intentos.
- Tres trials FAIL seguidos detienen el proceso, para no gastar presupuesto ante un fallo sistemático.

**Persistencia.**
- La base SQLite del estudio se trabaja en disco local y, tras cada trial, se publica en `results/studies/<S>_study.zip` con un reemplazo atómico. El zip contiene la base, el sampler serializado (`sampler.pkl`) y `meta.json` (versión de Optuna, número de trials).
- SQLite nunca se escribe en vivo sobre Drive, y base y sampler no pueden quedar desincronizados.
- Si hay carpetas de trials pero no existe el zip del estudio, el proceso se niega a crear un estudio nuevo encima.

**Reanudación.**
- Al relanzar se restaura el zip. El trial que estaba en curso no llegó a publicarse y se repite con los **mismos parámetros**: si era un warmup, sigue en la cola; si lo propuso TPE, el RNG del sampler restaurado continúa exactamente donde estaba.
- Los restos del intento interrumpido se archivan en `trials/<S>/_interrupted/`.
- Si la versión de Optuna cambiara entre sesiones, se crea un sampler nuevo con la misma semilla y se avisa de que las propuestas siguientes ya no continúan el RNG anterior. Cada trial registra su sesión y si el sampler se restauró (`session`, `sampler_restored`).

**Concurrencia.** Un solo runner por máquina (lock con latido en `results/runners/`). Varios runtimes pueden trabajar sobre el mismo `results/` con sujetos distintos, nunca dos sobre el mismo sujeto.

---

## 6. Verificación de finalistas y selección

### 6.1. Finalistas

Cuando un sujeto completa sus 32 trials, se toman sus **primeros 32 trials COMPLETE** y se elige el de mayor J_s **dentro de A5b** y el de mayor J_s **dentro de A5**. En un empate gana el trial con número menor. La elección se guarda en `verification/<S>/finalists.json` y no se recalcula al reanudar.

### 6.2. Reentrenamiento con semillas nuevas

Cada finalista se reentrena en los **mismos cinco folds** con las dos semillas de verificación (§4.4): 2 finalistas × 2 semillas × 5 folds = 20 entrenamientos por sujeto. Preprocesamiento, pérdida, optimizador, early stopping y reconstrucción son idénticos a los de la búsqueda. Las corridas completadas se saltan al reanudar; un intento interrumpido se archiva en `_attempts/` y se repite.

Las tres semillas (búsqueda + dos de verificación) miden la **estabilidad del entrenamiento sobre los mismos folds**, no la generalización a datos nuevos.

### 6.3. Configuración final por sujeto

- Para cada finalista se promedia el mejor Pearson de los cinco folds en cada semilla y luego se promedian las dos semillas (`verification_cv_best`). Se elige la arquitectura con mayor valor; también se reporta la desviación estándar entre semillas.
- No se ejecutan trials nuevos después de conocer la verificación.
- Se registra además la mediana de las diez `best_epoch` de verificación (redondeo de medios hacia arriba). Queda disponible para un posible reentrenamiento con todo el desarrollo, que no está implementado.

**Comparación A5 frente a A5b.** Se calcula la diferencia A5 − A5b emparejada por sujeto, semilla y fold (10 pares por sujeto). Por sujeto se reportan la media, la desviación estándar y el número de pares positivos; el valor global es la media de los tres sujetos con igual peso.

### 6.4. Comparación interna

`tpe_comparison.csv` resume el Pearson macro y por dedo, por sujeto, arquitectura y fold, para cada fuente de esta fase:
1. **Warmup de referencia** (par 1): sin optimizar, con la misma política de entrenamiento.
2. **Mejor trial TPE** de cada arquitectura.
3. **Verificación**: media de las dos semillas.

---

## 7. Comprobaciones

`pytest` ejecuta 55 pruebas; en Colab, con los datos reales, pasaron todas (86.75 s).

- **Datos.** Formas tras la exclusión (61 / 46 / 63 canales), bloques de 80 000 muestras, retención del guante alineada a 40 filas y guante interpolado de 5 × 8000 por bloque.
- **Constantes derivadas.** Decimación 10, 8000 columnas por bloque, región válida [125, 7875) y retardo de 2 columnas.
- **Limpieza.** El recorte usa los límites de train; un canal excluido con valores absurdos no altera la CMR de los demás.
- **Ausencia de fugas.** Se altera fuertemente el bloque validado (×100, desplazamiento y un impulso de 10⁸). Los límites de recorte, los escaladores de X e Y, los índices de ventanas y las características de los bloques de train quedan idénticos bit a bit; las del bloque validado sí cambian. También se comprueban las regiones purgadas, el retardo y los inicios de validación.
- **DSP.** Número de coeficientes de los tres FIR (331, 1651 y 243) y borde de 125 columnas. La potencia Morlet coincide con `mne.time_frequency.tfr_array_morlet` (rtol 10⁻⁶) y la convolución FFT con `scipy.signal.fftconvolve`.
- **Interpolación.** Valores y gradientes iguales a `F.interpolate` en float32 y float64.
- **Métricas.** Pearson de una serie consigo misma = 1; una predicción constante se marca inválida con su motivo; el overlap-add reconstruye exactamente.
- **Espacio y modelos.** La combinación k5-L5-f64 reproduce exactamente las especificaciones base de A5b y A5 del YAML y da salida (B, 5, 256) con los canales de los tres sujetos. Las 16 combinaciones (2 familias × 8 de k, L y f) hacen forward y backward con S3, con salida (B, 5, 256), gradientes finitos y el orden upsampling → concatenación → convolución en cada nivel del decoder. Los warmups tienen el orden y los valores de §5.4 y caen dentro del espacio.
- **Early stopping.** Mejora monótona → 50 épocas; pico antes de la época 15 → parada en la 25; una mejora tras el mínimo reinicia la paciencia; un empate no es mejora; paciencia agotada justo en la 50; un score no finito lanza un error.
- **`train_fold`.** Restaura el mejor checkpoint, sigue la política con scores guionizados y falla después de registrar la época cuando el Pearson es inválido.
- **Estudio** (con un runner simulado):
  - los 10 warmups se encolan una sola vez;
  - reanudar reproduce el mismo estudio que una ejecución sin interrupciones;
  - un warmup fallido se reintenta y los FAIL no cuentan;
  - tres FAIL seguidos detienen el proceso;
  - no se crea un estudio sobre carpetas huérfanas;
  - un trial interrumpido se repite con los mismos parámetros;
  - la verificación y el reporte funcionan de punta a punta;
  - un segundo runner en la misma máquina se rechaza.

Antes de la campaña, un *smoke* recorre la cadena completa con datos reales (S1, fold B1, los dos warmups del par 1, 2 épocas × 2 batches) en una carpeta separada.

---

## 8. Ejecución

### 8.1. Entorno

- **Colab desde VS Code** ([`scripts/colab_tpe.ipynb`](../scripts/colab_tpe.ipynb)): el código se clona de GitHub en una referencia fija, los `.mat` y `results/` están en Google Drive y la base SQLite de trabajo en el disco local del servidor.
- El runner corre en segundo plano y un monitor reejecutable muestra progreso, ETA, trial y fold en curso con su curva, y el latido del runner.
- Cada trial registra en su `config.json` las versiones de Python, PyTorch, CUDA, cuDNN, MNE, NumPy, SciPy, scikit-learn, pandas y Optuna, el modelo de GPU, el estado de TF32 y del determinismo y el commit de git. En el estudio quedan la sesión, la GPU y las versiones de PyTorch, CUDA y Optuna.

### 8.2. Estado de la campaña (corte: 2026-10-29, 18:51 UTC)

Valores del objetivo de búsqueda J_s; son métricas de selección, no resultados finales.

| Sujeto | Trials COMPLETE | FAIL | Mejor A5 | Mejor A5b |
|---|---:|---:|---|---|
| S1 | 32 / 32 | 0 | 0.7117 (trial 10) | 0.6618 (trial 24) |
| S2 | 32 / 32 | 0 | 0.6878 (trial 30) | 0.6480 (trial 0, warmup de referencia) |
| S3 | 0 / 32 | 0 | en curso | en curso |

- S1 y S2 tardaron unas 13.3 h en total desde el arranque (≈ 12.5 min por trial).
- La verificación de finalistas está pendiente. Las tablas definitivas salen de `results/summary/` (§9).

---

## 9. Resultados y archivos

```text
results/
├── studies/<S>_study.zip           estudio Optuna publicado (study.sqlite + sampler.pkl + meta.json)
├── trials/<S>/trial_NNN/           config.json, trial.json (estado y valor), error.txt si falló
│   └── <fold>/                     status.json, history.csv, metrics.json, predictions.npz
├── verification/<S>/               finalists.json; <A5b|A5>/seed_<base>/<fold>/ (mismos archivos)
├── capacity/                       capacity_<gpu>.csv
├── runners/  logs/                 latido del runner y logs del proceso en segundo plano
└── summary/                        tablas y figuras de report_tpe.py
```

Cada trial guarda en sus `user_attrs` el resumen de sus folds: semilla, mejor época, épocas entrenadas, motivo de parada, Pearson por dedo, MSE, CCC, curva de validación, sesión, GPU y versiones. Por eso los reportes se regeneran desde los zip aunque falte algún archivo de las carpetas.

| Archivo de `summary/` | Contenido |
|---|---|
| `tpe_trials.csv` | Una fila por trial ejecutado: hiperparámetros, objetivo, parámetros del modelo, medias de mejor época y épocas entrenadas, tiempo, estado, error, sesión y GPU |
| `tpe_folds.csv` | Una fila por fold: semilla, mejor época, épocas entrenadas, motivo de parada, Pearson macro y por dedo, MSE, CCC y tiempo |
| `tpe_verification.csv` | Una fila por finalista × semilla × fold, con las mismas métricas |
| `tpe_verification_paired*.csv` | Diferencias A5 − A5b emparejadas y su resumen (§6.3) |
| `tpe_selected_models.csv` | Configuración elegida por sujeto, objetivo de búsqueda, media y DE de verificación y mediana de la mejor época |
| `tpe_comparison*.csv` | Comparación interna (§6.4) por sujeto, dedo y fold |
| `figures/` | Historia de la optimización (`optimization_history.png`), hiperparámetros frente al objetivo (`params_vs_objective.png`) y búsqueda frente a verificación (`verification.png`) |

`report_tpe.py` puede ejecutarse a mitad de campaña: reporta lo que haya.

---

## 10. Limitaciones metodológicas

1. **Optimismo de selección.** El objetivo toma el máximo de validación de cada fold y luego se elige el mejor de 32 trials en los mismos folds. J_s es una métrica de selección, no una estimación de rendimiento en datos nuevos (Cawley y Talbot, 2010).
2. **Verificación en los mismos folds.** Las semillas nuevas miden estabilidad del entrenamiento, no generalización. No hay evaluación independiente en test.
3. **Comparador interno sesgado.** El warmup de referencia es uno de los 32 trials, así que el mejor trial de su arquitectura es, por construcción, mayor o igual que él. La diferencia es descriptiva.
4. **Sin control de búsqueda aleatoria.** Sin un Random Search con el mismo presupuesto no se puede atribuir ventaja a TPE frente a otros buscadores.
5. **Semillas fijas por fold.** Emparejan las comparaciones entre trials, pero el objetivo puede ajustarse a esas semillas concretas; la verificación con semillas nuevas lo acota.
6. **Early stopping.** Con `min_delta = 0`, cualquier mejora mínima, incluso por ruido, reinicia la paciencia. Las configuraciones lentas pueden no converger en 50 épocas.
7. **Exclusión de canales.** La lista es fija y no se decide dentro de cada fold: no es un criterio anidado.
8. **Retardo fijo de 20 ms.** No se ajusta ni se prueban otros valores.
9. **Causalidad.** Los filtros de fase cero, las wavelets y el modelo usan contexto futuro, y el entrenamiento incluye bloques posteriores al validado. La validación por bloques con purga no equivale a predicción causal en tiempo real.
10. **No determinismo de la GPU.** La reproducibilidad es estadística, no bit a bit (§4.5).
11. **Sin pesos guardados.** Reproducir una inferencia exige reentrenar; el reentrenamiento final con todo el desarrollo no está implementado.

---

## Referencias

- Tangermann, M. et al. (2012). Review of the BCI Competition IV. *Frontiers in Neuroscience*, 6:55. https://doi.org/10.3389/fnins.2012.00055
- Liu, Y., Coon, W. G., de Pesters, A., Brunner, P. y Schalk, G. (2015). The effects of spatial filtering and artifacts on electrocorticographic signals. *Journal of Neural Engineering*, 12(5):056008.
- Bergstra, J., Bardenet, R., Bengio, Y. y Kégl, B. (2011). Algorithms for hyper-parameter optimization. *Advances in Neural Information Processing Systems*, 24.
- Falkner, S., Klein, A. y Hutter, F. (2018). BOHB: Robust and efficient hyperparameter optimization at scale. *ICML 2018*.
- Akiba, T., Sano, S., Yanase, T., Ohta, T. y Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of KDD 2019*.
- Kingma, D. P. y Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015*.
- Lin, L. I.-K. (1989). A concordance correlation coefficient to evaluate reproducibility. *Biometrics*, 45(1):255–268.
- Cawley, G. C. y Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research*, 11:2079–2107.

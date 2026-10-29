# Plan de implementación — Optimización conjunta con TPE para ECoG

**Versión:** 2.1 — v2.0 con las correcciones de implementación de la sección 0  
**Estado:** protocolo de implementación, previo a la ejecución de TPE  
**Proyecto:** decodificación continua de flexión de cinco dedos a partir de ECoG  
**Sujetos:** S1, S2 y S3 — BCI Competition IV, Dataset 4  
**Repositorio:** `TPE_ECoG`, que copia el pipeline P0 del repositorio de ablaciones (commit `043c44f`)  
**Alcance:** optimización conjunta de **A5b y A5**, sin nuevas arquitecturas ni cambios de preprocesamiento  
**Presupuesto:** **32 trials por sujeto**, cinco folds por trial; dos semillas adicionales para un finalista de cada arquitectura  
**Optimizador de red:** **Adam + L2 acoplada**  
**Optimizador de hiperparámetros:** **Optuna TPE multivariante**  
**Métrica principal:** Pearson macro de los cinco dedos en trayectorias temporales reconstruidas

> **Decisión central de esta versión:** cada entrenamiento conserva el checkpoint con **mejor Pearson de validación** y utiliza Early Stopping con `min_delta=0.0`, `patience=10`, `min_epochs=15` y `max_epochs=50`. El objetivo de TPE deja de ser la media de las épocas 46–50 y pasa a ser el **promedio de los mejores Pearson de los cinco folds**.

> **Fuera de alcance:** no implementar FingerFlex (R-FF), DTCNet (R-DTC), nuevas variantes A5, nuevos preprocesamientos, búsqueda aleatoria, ensambles ni evaluación del test oficial. No introducir procedimientos administrativos innecesarios como hashes o puertas formales de aprobación.

---

## 0. Registro de cambios de la implementación (v2.1)

Todos se decidieron antes de lanzar el primer estudio.

| # | Cambio respecto a v2.0 | Motivo |
|---|---|---|
| 1 | Retardo ECoG(t) → guante(t + **20 ms**) = **2 columnas** a 100 Hz (la v2.0 decía 200 ms). | Es el retardo del pipeline P0 de las ablaciones (`delay_ms: 20`), con el que se obtuvieron todas las cifras de referencia de este plan. Los 200 ms venían del proyecto anterior (TT\Code). |
| 2 | Repositorio nuevo `TPE_ECoG`, paquete `ecog_tpe`. `data.py`, `cleaning.py`, `preprocessing.py`, `evaluation.py` y `models.py` son copia literal de `ecog_ablations` (commit `043c44f`); `training.py` parte de la misma copia. | Repositorio independiente para la fase TPE sin tocar P0 ni A5b/A5. |
| 3 | El SQLite de cada estudio se trabaja en disco local y se publica en Drive tras cada trial como `results/studies/<S>_study.zip` (base de datos + sampler serializado + versión de Optuna), igual que el `StudyArchive` del proyecto anterior. | SQLite no se escribe en vivo sobre Drive; el sampler serializado continúa exactamente el RNG al reanudar. |
| 4 | Resultados en `results/` sin el nivel `tpe/` de la sección 9. | El repositorio entero es de TPE. |
| 5 | Un warmup que falla se reencola una vez; el proceso se detiene tras 3 trials FAIL consecutivos. | Concreta "reintentar fallas transitorias" (sección 6) sin buscar indefinidamente ante un fallo sistemático. |
| 6 | **No se guardan pesos** en ninguna corrida (ni búsqueda ni verificación); cada fold conserva métricas y `predictions.npz` del mejor checkpoint. El refit opcional (7.1) no se implementa por ahora. | Decisión del autor (espacio en Drive), igual que en las ablaciones. |
| 7 | `precision: float32` se implementa como en las ablaciones: tensores float32 con **TF32 permitido** en convoluciones y matmul (GPUs Ampere+), igual en todos los trials. | Mantener la aritmética con la que se obtuvieron las referencias y que el warmup del par 1 reproduzca las curvas de A5b/A5. |

---

## 1. Motivación y punto de partida

La fase de ablaciones quedó cerrada con **11 configuraciones entrenadas, 3 sujetos y 5 folds**, equivalentes a **165 entrenamientos**. Las dos arquitecturas finalistas son:

| Arquitectura | Componentes distintivos | Pearson CV en época 50, media de S1–S3 |
|---|---|---:|
| **A5b** | LayerNorm, dropout 0.1, interpolación y concatenación **antes** de Conv1d | **0.677915** |
| **A5** | Igual que A5b, pero reemplaza interpolación por `ConvTranspose1d` depthwise | **0.683715** |

Esos valores corresponden a la **última época** del estudio de ablaciones, no a los mejores checkpoints.

### 1.1. Evidencia histórica para cambiar el entrenamiento

Se analizaron `history_A5.csv` y `history_A5b.csv`: **30 curvas reales de 50 épocas** (2 arquitecturas × 3 sujetos × 5 folds). Una simulación retrospectiva de la política acordada (`min_delta=0`, paciencia 10 y contador activo a partir de la época 16) produce:

| Métrica de la simulación retrospectiva | Resultado |
|---|---:|
| Épocas entrenadas en promedio | **32.3** |
| Pearson medio recuperado con Early Stopping | **0.7222** |
| Mejor Pearson medio si se observan las 50 épocas completas | **0.7254** |
| Diferencia media respecto al máximo de 50 épocas | **−0.0032** |

Desglose por arquitectura, promediando los 15 sujeto-fold:

| Arquitectura | Pearson época 50 | Mejor Pearson recuperado con Early Stopping | Media de épocas entrenadas |
|---|---:|---:|---:|
| A5b | 0.677915 | **0.711371** | 34.0 |
| A5 | 0.683715 | **0.733021** | 30.6 |

**Interpretación:** seleccionar checkpoints y detener entrenamientos sin progreso recupera una parte importante del rendimiento perdido al utilizar obligatoriamente la época 50, con menos épocas en promedio. La simulación no garantiza que las nuevas configuraciones TPE converjan igual: explorarán otros LR, L2, kernels y anchos.

**Cautela científica:** las cifras de mejor checkpoint son métricas de **selección sobre validación**, sujetas a optimismo. No deben presentarse como rendimiento de test independiente. La selección retrospectiva de una época tampoco implica que se conservaran físicamente sus pesos en las ejecuciones antiguas.

### 1.2. Pregunta experimental

¿En qué medida la optimización conjunta de arquitectura, tasa de aprendizaje y regularización L2 mejora el Pearson macro de **A5b** y **A5**, con un esquema uniforme de validación cruzada temporal y selección de checkpoints?

La fase analiza el rendimiento **tras optimización con TPE**. Sin un buscador aleatorio de control con igual presupuesto, no puede concluirse que TPE sea superior a Random Search.

---

## 2. Elementos fijos: reutilizar el pipeline P0

Reutilizar el código y las particiones que generaron los resultados anteriores. TPE **no puede modificar** el pipeline de señales ni los criterios de evaluación según el trial.

| Elemento | Especificación fija |
|---|---|
| Datos | Desarrollo de S1, S2 y S3; cinco bloques cronológicos B1–B5 |
| Canales eliminados antes del procesamiento | S1: 54; S2: 20 y 37; S3: 49; **índices base cero** |
| Artefactos restantes | Control robusto de amplitud por canal con umbral de 25 MAD escaladas; ajustes únicamente con entrenamiento |
| Referencia espacial | **P0:** Common Median Reference (CMR) sobre señal cruda depurada |
| Características | Morlet, banda 40–200 Hz, 40 frecuencias; señal de características a 100 Hz |
| Escalador de entradas | `RobustScaler(quantile_range=(10,90), unit_variance=True)` por columna canal-frecuencia |
| Escalador de salidas | `MinMaxScaler` independiente por dedo |
| Ajuste de escaladores | Exclusivamente muestras válidas de entrenamiento, sin purga ni validación |
| Alineamiento | ECoG(t) → flexión del guante(t + **20 ms**) = **2 columnas** a 100 Hz (`delay_ms: 20`, igual que en ablaciones) |
| Ventanas | 256 muestras; stride 1 para entrenar, stride 64 para validar |
| Validación cruzada | B1–B5; respetar purga y máscaras temporales existentes |
| Reconstrucción | Overlap-add de ventanas de validación, igual que en ablaciones |
| Métrica | Pearson macro de los cinco dedos, calculado sobre trayectorias reconstruidas |
| Pérdida | `0.5*MSE + 0.5*(1 - cosine_similarity_temporal)` |

**Reglas de datos:** ningún escalador ni estadística de control de amplitudes se ajusta con validación o test. No cruzar límites temporales entre entrenamiento y validación al formar ventanas. Los folds deben ser idénticos para todos los trials.

Se permite guardar en caché las características de P0 por sujeto y fold, siempre que correspondan a las estadísticas de entrenamiento de **ese mismo fold**. Cada trial inicia pesos y estado del optimizador desde cero.

---

## 3. Arquitecturas candidatas y espacio de búsqueda

Ambas familias reutilizan la implementación verificada de `EncoderDecoder1D`:

| Componente | A5b | A5 |
|---|---|---|
| Stem y encoder | Conv1d + LayerNorm + GELU + dropout 0.1 | Igual |
| Dilataciones | `[1,2,3,1,2]`, truncadas al número de niveles | Igual |
| Conexiones de salto | Antes del pooling | Igual |
| Orden del decoder | **Interpolar → concatenar skip → Conv1d** | **ConvTranspose depthwise → concatenar skip → Conv1d** |
| Upsampling | Interpolación lineal | `ConvTranspose1d(C,C,k=4,s=2,p=1,groups=C,bias=False)` |
| Normalización de decoder | LayerNorm | LayerNorm |
| Cabeza | Conv1d → cinco valores continuos, sin activación final | Igual |

Para `L=5`, `f=64`, `k=5`, la construcción debe reproducir exactamente A5b o A5 de las ablaciones. En otros casos, construir la estructura mediante:

```python
encoder_channels = [f, f, 2*f, 4*f, 8*f, 8*f][:L + 1]
encoder_kernels = [7, 7] + [k] * (L - 2)
encoder_dilations = [1, 2, 3, 1, 2][:L]
decoder_kernels = list(reversed(encoder_kernels))
```

La salida de todos los modelos debe conservar forma `(B, 5, 256)`. No introducir capas adicionales ni alteraciones silenciosas de anchos para resolver incompatibilidades.

### 3.1. Parámetros optimizados conjuntamente

| Parámetro | Optuna | Dominio |
|---|---|---|
| `architecture_id` | `suggest_categorical` | `['A5b','A5']` |
| `learning_rate` | `suggest_float(..., log=True)` | `5e-5` a `5e-4` |
| `weight_decay` | `suggest_float(..., log=True)` | `1e-7` a `1e-3` |
| `kernel_size` | `suggest_categorical` | `[5,7]` |
| `encoder_levels` | `suggest_categorical` | `[4,5]` |
| `base_filters` | `suggest_categorical` | `[64,96]` |

Son **16 combinaciones discretas** (2 × 2 × 2 × 2) más dos dimensiones continuas. Se evita volver al esquema antiguo de optimizar LR/L2 en una arquitectura y congelarlos al modificarla.

No añadir a TPE dropout, normalización, preprocesamiento, retardo, batch size, scheduler ni función de pérdida.

### 3.2. Comprobación previa de capacidad

Ejecutar forward/backward sintético en las 16 combinaciones discretas y confirmar viabilidad de memoria en S1–S3, especialmente `f=96, L=5, k=7`. Si alguna combinación no cabe con batch 128, **ajustar el espacio antes del primer estudio**. No permitir que solo algunos trials modifiquen batch o precisión para evitar OOM.

---

## 4. Política de entrenamiento y Early Stopping

Esta sección sustituye íntegramente el entrenamiento a 50 épocas fijas y la media de las últimas cinco épocas del protocolo anterior.

| Componente | Valor definitivo de esta fase |
|---|---|
| Optimizador | **Adam + L2 acoplada** (`decoupled_weight_decay=False`) |
| LR / L2 | Propuestos por TPE en cada trial |
| `betas` / `eps` | `(0.9,0.999)` / `1e-8` |
| Batch size | 128 |
| Scheduler | Ninguno |
| Monitor | `val_pearson_macro` sobre trayectorias reconstruidas |
| Frecuencia de validación | **Una vez al final de cada época** |
| Modo | `max` |
| **`min_delta`** | **`0.0`** |
| **`patience`** | **`10` comprobaciones de validación** |
| **`min_epochs`** | **`15`** |
| **`max_epochs`** | **`50`** |
| Checkpoint | Pesos de la época con **mayor Pearson** observado |
| Pruning de Optuna | Desactivado (`NopPruner`) |
| Semillas | Fijas por sujeto/fold en búsqueda; adicionales en verificación |

El valor `min_delta=0` es deliberado: **cualquier nuevo máximo estricto de Pearson cuenta como mejora**. No es un criterio de significancia estadística. Los históricos no mostraron beneficio práctico relevante al pasar de `0` a `0.0005`, y `0` evita introducir un umbral sin justificación independiente.

### 4.1. Semántica precisa del Early Stopping

1. Calcular Pearson de validación al final de **cada época**, desde la época 1.
2. Mantener `best_score=-inf`, `best_epoch` y los mejores pesos.
3. Si `val_pearson_macro > best_score`, guardar ese checkpoint, actualizar mejor época y reiniciar contador de falta de mejora.
4. **Las épocas 1–15 sirven para aprender y guardar checkpoints, pero no pueden provocar parada anticipada.** Al terminar la época 15, iniciar el contador de paciencia en cero.
5. Desde la época 16, si no se registra un nuevo máximo estricto, incrementar el contador en una unidad por validación; una mejora lo reinicia a cero.
6. Detener después de **10 validaciones consecutivas sin nuevo máximo** a partir de la época 16; en todo caso detener en la época 50.
7. Al finalizar, **restaurar el mejor checkpoint**, que puede ser anterior a la época final entrenada.
8. Calcular las métricas finales del fold con ese checkpoint, **no** con los pesos de la última época alcanzada.

En ausencia de mejoras después de la época 15, la primera parada posible ocurre al terminar la época **25**. Esto implementa el mínimo de 15 épocas junto con la paciencia acordada; `min_epochs=15` por sí solo no sustituye la política exacta de activación del contador.

Pseudocódigo del mecanismo:

```python
best_score = float('-inf')
best_epoch = None
wait = 0

for epoch in range(1, 51):
    train_one_epoch(...)
    metrics = validate_reconstructed_trajectories(...)
    score = float(metrics['pearson_macro'])

    if not np.isfinite(score):
        raise ValueError('Pearson de validacion no finito')

    if score > best_score:  # min_delta = 0.0
        best_score = score
        best_epoch = epoch
        save_or_copy_best_weights(...)
        wait = 0
    elif epoch >= 16:
        wait += 1

    if epoch == 15:
        wait = 0  # la paciencia empieza DESPUES del minimo

    record_history(epoch=epoch, metrics=metrics, ...)

    if epoch >= 16 and wait >= 10:
        stop_reason = 'patience_exhausted'
        break
else:
    stop_reason = 'max_epochs'

restore_best_weights(...)
fold_metrics = evaluate_best_checkpoint(...)
```

La política es válida tanto con un loop PyTorch propio como con callbacks, pero, si se utiliza PyTorch Lightning, la combinación de `min_epochs`, `EarlyStopping` y frecuencia de validación deberá reproducir **exactamente** la semántica especificada arriba. En Lightning, `patience` cuenta comprobaciones de validación, no épocas por definición.

### 4.2. Adam + L2

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=trial_params['learning_rate'],
    weight_decay=trial_params['weight_decay'],
    betas=(0.9, 0.999),
    eps=1e-8,
    decoupled_weight_decay=False,
)
```

No sumar L2 nuevamente a la función de pérdida ni utilizar AdamW.

### 4.3. Casos especiales

- Predicción constante en algún dedo, Pearson no finito, NaN/Inf u otro fallo numérico: registrar el fallo; no eliminar ese dedo ni convertirlo silenciosamente en cero.
- Si el entrenamiento termina en época 50, igualmente usar el mejor checkpoint, aunque sea de una época anterior.
- Registrar `best_epoch`, `epochs_trained` y `stop_reason` por fold. No interpretar `epochs_trained` como la época del checkpoint seleccionado.
- Si alguna configuración aprende lentamente debido al LR, dispondrá de hasta 50 épocas y diez validaciones sin mejora; esta regla no garantiza que todos los LR puedan converger por completo, pero proporciona una comparación uniforme.

---

## 5. Función objetivo de TPE

Cada trial `theta` entrena desde cero cinco modelos, uno por fold, con el mismo esquema de validación temporal de las ablaciones.

En el sujeto `s`, fold `b`, época `e`, la métrica es:

$$
r_{s,b,e}=\frac{1}{5}\sum_{d=1}^{5}
\operatorname{Pearson}(y_{s,b,d},\hat y_{s,b,e,d}).
$$

El **score del fold** es el mejor Pearson observado hasta la parada:

$$
q_{s,b}(\theta)=\max_{1\leq e\leq T_{s,b}(\theta)} r_{s,b,e},
$$

siendo `T` la época de parada, con `15 <= T <= 50` y la política específica del apartado 4.

El **objetivo que Optuna maximiza** es la media no ponderada de los cinco folds:

$$
\boxed{J_s(\theta)=\frac{1}{5}\sum_{b=1}^{5}q_{s,b}(\theta)}.
$$

**No utilizar** media de las épocas 46–50, máximo global entre folds, MSE o la última época para seleccionar hiperparámetros.

Guardar también Pearson por dedo, MSE, CCC, curva completa observada, mejor época y duración de cada fold. La evaluación final de esas métricas corresponde a los pesos restaurados del mejor checkpoint. La época 50 solo puede registrarse como tal **si realmente fue entrenada**.

**Limitación:** seleccionar el máximo de validación por fold y luego el mejor de 32 trials aumenta el optimismo de la puntuación de selección. La media de los folds es el **objetivo de optimización**, no una prueba independiente de generalización. Evitar cualquier ajuste guiado por el test oficial.

---

## 6. Optuna: tres estudios separados y diez propuestas iniciales equilibradas

Ejecutar un estudio independiente por sujeto; fijar `optuna==5.0.0`, `TPESampler(multivariate=True, n_startup_trials=10, constant_liar=False)`, ejecución secuencial y almacenamiento SQLite.

### 6.1. Propuestas iniciales

Encolar cinco configuraciones de hiperparámetros idénticas para ambas arquitecturas; son **10 trials dentro de los 32**, no adicionales.

| Par | `k` | `L` | `f` | `learning_rate` | `weight_decay` | Ejecutar con |
|---|---:|---:|---:|---:|---:|---|
| 1 — referencia | 5 | 5 | 64 | `8.42e-5` | `1e-6` | A5b y A5 |
| 2 | 7 | 5 | 64 | `1e-4` | `1e-5` | A5b y A5 |
| 3 | 5 | 4 | 96 | `2e-4` | `1e-6` | A5b y A5 |
| 4 | 7 | 4 | 96 | `7e-5` | `1e-4` | A5b y A5 |
| 5 | 5 | 5 | 96 | `3e-4` | `1e-7` | A5b y A5 |

Los siguientes **22 trials** son propuestos por TPE. Puede repetir una estructura discreta con LR/L2 diferentes: eso no es duplicado. Evitar únicamente una repetición **exacta** de todos los hiperparámetros y semilla/folds, sin excluir repeticiones arquitectónicas útiles.

### 6.2. Semillas

Para emparejar los resultados dentro del estudio, cada fold utiliza la misma semilla base en todos los trials de búsqueda:

```python
subject_idx = {'S1': 0, 'S2': 1, 'S3': 2}[subject]
fold_idx = {'B1': 0, 'B2': 1, 'B3': 2, 'B4': 3, 'B5': 4}[fold]
seed_train = 42 + 5 * subject_idx + fold_idx
```

Semillas para el sampler TPE: `S1=2026`, `S2=2027`, `S3=2028`. Reiniciar generadores de Python, NumPy, PyTorch y dataloader para cada fold/trial. Registrar la versión de PyTorch, CUDA, Optuna y el modelo de GPU en el README o un resumen simple.

### 6.3. Configuración `configs/tpe.yaml`

```yaml
experiment: joint_tpe_a5b_a5_best_checkpoint
preprocessing: P0
subjects: [S1, S2, S3]
folds: [B1, B2, B3, B4, B5]
architectures: [A5b, A5]

training:
  optimizer: adam
  decoupled_weight_decay: false
  betas: [0.9, 0.999]
  eps: 1.0e-8
  batch_size: 128
  loss: mse_cosine_0.5_0.5
  scheduler: null
  precision: float32
  early_stopping:
    enabled: true
    monitor: val_pearson_macro
    mode: max
    min_delta: 0.0
    patience: 10
    min_epochs: 15
    max_epochs: 50
    patience_count_start_epoch: 16
    validation_every_n_epochs: 1
    restore_best_weights: true
  pruning: none

search:
  sampler: tpe
  optuna_version: '5.0.0'
  multivariate: true
  constant_liar: false
  n_startup_trials: 10
  n_trials_per_subject: 32
  n_jobs: 1
  objective: mean_best_checkpoint_pearson_cv
  seed_by_subject: {S1: 2026, S2: 2027, S3: 2028}
  ranges:
    learning_rate: [0.00005, 0.0005]
    weight_decay: [0.0000001, 0.001]
    kernel_size: [5, 7]
    encoder_levels: [4, 5]
    base_filters: [64, 96]

verification:
  select: best_trial_per_architecture_and_subject
  additional_seed_bases: [1042, 2042]
  folds: [B1, B2, B3, B4, B5]
  training_policy: same_as_search
```

### 6.4. Esqueleto de ejecución

```python
import optuna
import numpy as np

FOLDS = ['B1', 'B2', 'B3', 'B4', 'B5']


def suggest(trial):
    return {
        'architecture_id': trial.suggest_categorical('architecture_id', ['A5b', 'A5']),
        'learning_rate': trial.suggest_float('learning_rate', 5e-5, 5e-4, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True),
        'kernel_size': trial.suggest_categorical('kernel_size', [5, 7]),
        'encoder_levels': trial.suggest_categorical('encoder_levels', [4, 5]),
        'base_filters': trial.suggest_categorical('base_filters', [64, 96]),
    }


def objective(trial, subject):
    params = suggest(trial)
    fold_scores = []
    fold_best_epochs = []
    fold_train_epochs = []

    for fold_idx, fold in enumerate(FOLDS):
        result = train_one_fold(
            subject=subject,
            fold=fold,
            params=params,
            seed=42 + 5 * SUBJECT_INDEX[subject] + fold_idx,
            preprocessing='P0',
            early_stopping={
                'monitor': 'val_pearson_macro',
                'mode': 'max',
                'min_delta': 0.0,
                'patience': 10,
                'min_epochs': 15,
                'max_epochs': 50,
            },
        )
        # train_one_fold restaura el mejor checkpoint y devuelve
        # el Pearson macro de las trayectorias reconstruidas.
        score = float(result.best_val_pearson_macro)
        if not np.isfinite(score):
            raise ValueError(f'Score invalido: {subject}, {fold}, {trial.number}')
        fold_scores.append(score)
        fold_best_epochs.append(int(result.best_epoch))
        fold_train_epochs.append(int(result.epochs_trained))
        # Exportar historia, score por dedo, MSE, CCC y parada del fold.

    trial.set_user_attr('fold_best_pearson', fold_scores)
    trial.set_user_attr('fold_best_epochs', fold_best_epochs)
    trial.set_user_attr('fold_epochs_trained', fold_train_epochs)
    return float(np.mean(fold_scores))


sampler = optuna.samplers.TPESampler(
    seed=2026 + SUBJECT_INDEX[subject],
    n_startup_trials=10,
    multivariate=True,
    constant_liar=False,
)

study = optuna.create_study(
    study_name=f'joint_tpe_{subject}',
    direction='maximize',
    sampler=sampler,
    storage=f'sqlite:///results/tpe/studies/{subject}.sqlite',
    load_if_exists=True,
    pruner=optuna.pruners.NopPruner(),
)

# En un estudio nuevo, encolar exactamente los diez trials iniciales.
# Al reanudar, NO volver a encolarlos.
# Ejecutar secuencialmente hasta acumular 32 trials COMPLETE,
# no simplemente hasta contar 32 intentos si hubo fallos.
```

`train_one_fold` es una **interfaz propuesta**, no una función verificada del repositorio. Adaptarla a las rutinas existentes sin duplicar el preprocesamiento ni cambiar el criterio de Pearson.

**Persistencia:** conservar estudios en SQLite para retomar ejecuciones; no volver a encolar warmups al reanudar. Un reinicio del proceso puede modificar el estado aleatorio interno del sampler aunque los trials anteriores permanezcan registrados; documentar reinicios y los hiperparámetros efectivamente propuestos.

**Fallos:** OOM, NaN, error de código o pérdida de datos implican trial `FAIL` con causa, no score inventado. Reintentar fallas transitorias y obtener **32 trials completos por sujeto**; registrar los intentos adicionales por separado.

---

## 7. Verificación de finalistas

Una vez completados los 32 trials de cada sujeto:

1. Seleccionar el trial con mayor `J_s` **dentro de A5b**.
2. Seleccionar el trial con mayor `J_s` **dentro de A5**.
3. Reentrenar ambos, sobre **los mismos cinco folds**, con dos semillas adicionales por fold:

```python
seed_verify_1 = 1042 + 5 * subject_idx + fold_idx
seed_verify_2 = 2042 + 5 * subject_idx + fold_idx
```

4. Aplicar **exactamente la misma política de Early Stopping** (`min_delta=0`, paciencia 10, mínimo 15, máximo 50), preprocesamiento P0, pérdida, optimizador y reconstrucción.
5. Comparar Pearson del mejor checkpoint por sujeto/fold/semilla y su dispersión, con especial atención a si A5 conserva ventaja frente a A5b.
6. Elegir una arquitectura/configuración final por sujeto a partir de la **media de las dos semillas de verificación**. No ejecutar nuevos trials después de conocer sus resultados.

Estas dos semillas son nuevas respecto a la de búsqueda. Las tres semillas representan una comprobación de **estabilidad de entrenamiento sobre los mismos folds**, no una evaluación independiente del conjunto de datos.

### 7.1. Reentrenamiento final opcional

Si posteriormente se requiere un modelo entrenado sobre todo el conjunto de desarrollo, ya no existirá un fold de validación para aplicar Early Stopping. La duración puede fijarse **antes de ver test** mediante la mediana, redondeada al entero más cercano, de las diez `best_epoch` de verificación de la configuración elegida (2 semillas × 5 folds). Reentrenar desde cero durante ese número fijo de épocas con todos los datos de desarrollo.

Este refit es **opcional** y queda fuera del presupuesto cerrado de búsqueda y verificación. Si se ejecuta uno por sujeto, sumar tres entrenamientos adicionales y reportarlos por separado.

---

## 8. Presupuesto computacional

| Etapa | Trials o configuraciones por sujeto | Folds | Semillas de entrenamiento | Entrenamientos totales |
|---|---:|---:|---:|---:|
| Warmup equilibrado | 10 | 5 | 1 | **150** |
| TPE guiado | 22 | 5 | 1 | **330** |
| Verificación de finalistas | 2 | 5 | 2 adicionales | **60** |
| **Total TPE + verificación** | | | | **540** |

Los 540 entrenamientos corresponden a 3 sujetos. El acumulado del proyecto será **165 ablaciones + 540 TPE = 705 entrenamientos completos**, sin contar reintentos técnicos ni refits opcionales.

Cada entrenamiento tendrá como máximo 50 épocas, pero Early Stopping reducirá la duración de muchas ejecuciones. La media de **32.3 épocas** observada retrospectivamente sirve solo como orientación de costo, **no como promesa** para los nuevos LR y arquitecturas.

---

## 9. Resultados y archivos mínimos

Estructura sugerida, adaptando los nombres a lo ya implementado:

```text
configs/
  tpe.yaml
src/ecog_ablations/
  tpe_search.py
  tpe_reporting.py
scripts/
  run_tpe.py
  report_tpe.py
results/tpe/
  studies/
    S1.sqlite
    S2.sqlite
    S3.sqlite
  trials/
    S1/trial_000/
      config.json
      B1/history.csv
      B1/metrics.json
      B2/...
    S2/...
    S3/...
  verification/
    S1/A5b/...
    S1/A5/...
    S2/...
    S3/...
  summary/
    tpe_trials.csv
    tpe_folds.csv
    tpe_verification.csv
    tpe_selected_models.csv
    figures/
```

**`tpe_trials.csv`**, una fila por trial: `subject`, `trial`, `state`, `architecture_id`, `learning_rate`, `weight_decay`, `kernel_size`, `encoder_levels`, `base_filters`, `objective_cv_best`, `n_parameters`, `mean_best_epoch`, `mean_epochs_trained`, `total_train_seconds`, `n_valid_folds`, `error`.

**`tpe_folds.csv`**, cinco filas por trial completo: `subject`, `trial`, `fold`, `seed`, `best_epoch`, `epochs_trained`, `stop_reason`, `best_pearson_macro`, `pearson_thumb`, `pearson_index`, `pearson_middle`, `pearson_ring`, `pearson_little`, `mse_macro_best`, `ccc_macro_best`, `n_parameters`, `train_seconds`.

**`tpe_verification.csv`**: `subject`, `architecture_id`, `source_trial`, `seed`, `fold`, `best_epoch`, `epochs_trained`, `best_pearson_macro`, `mse_macro_best`, `ccc_macro_best`.

**`tpe_selected_models.csv`**: `subject`, `chosen_architecture`, `chosen_trial`, `learning_rate`, `weight_decay`, `kernel_size`, `encoder_levels`, `base_filters`, `search_cv_best`, `verification_cv_best`, `verification_seed_sd`, `median_best_epoch`.

Guardar `history.csv` por época para estudiar sobreajuste y variabilidad de convergencia. Conservar los mejores pesos **durante** cada entrenamiento; persistir en disco al menos los checkpoints de los finalistas cuando se necesite reproducir inferencia, sin obligarse a almacenar todos los checkpoints de la búsqueda.

---

## 10. Comparaciones para el artículo

**La regla de selección de checkpoint cambió.** Por ello, no comparar directamente `study.best_value` (máximo por fold) con Pearson **época 50** de las ablaciones y atribuir toda la diferencia a TPE.

Usar las curvas ya disponibles de A5b y A5 para crear una **referencia retrospectiva con el mismo Early Stopping**:

| Referencia | Pearson CV época 50 | Pearson CV con política de parada simulada |
|---|---:|---:|
| A5b sin TPE, `k=5,L=5,f=64` | 0.677915 | **0.711371** |
| A5 sin TPE, `k=5,L=5,f=64` | 0.683715 | **0.733021** |

Estas cifras retrospectivas provienen de los registros históricos de validación, **no** son resultados nuevos de test ni implican recuperar pesos no guardados. Como el trial de referencia con `LR=8.42e-5,L2=1e-6` está incluido en los diez warmups de TPE, también habrá una comparación prospectiva bajo la **misma política real de entrenamiento**.

**Reportar de forma separada:**

1. Ablaciones originales con época 50 (resultado histórico).
2. A5b/A5 sin optimización, puntuados con la política de mejor checkpoint (histórico reconstruido y/o warmup reproducido).
3. Mejor trial por arquitectura según TPE, con su objetivo sobre cinco folds.
4. Resultados de verificación con dos semillas adicionales, manteniendo folds y política de parada.
5. Diferencias por sujeto, dedo y fold; dispersión entre semillas; parámetros, mejor época y costo de cómputo.

Las ventanas solapadas no cuentan como muestras biológicas independientes. Los cinco folds se usan para seleccionar hiperparámetros y los finalistas se vuelven a probar en esos mismos folds; por tanto, **no** se dispone todavía de una estimación independiente insesgada de generalización. La exclusión histórica del canal S3-49 también debe declararse como limitación exploratoria.

No afirmar superioridad directa frente a FingerFlex/DTCNet sin evaluarlos bajo un protocolo comparable, ni atribuir a TPE ventaja causal frente a Random Search cuando ese método no fue ejecutado.

---

## 11. Secuencia concreta de implementación

1. **Conservar P0 y las clases A5b/A5** como fueron evaluadas en la fase de ablaciones; no tocar filtros, escaladores, folds ni reconstrucción.
2. Extender `EncoderDecoder1D` para permitir `k in {5,7}`, `L in {4,5}`, `f in {64,96}` y confirmar que la configuración original se construye sin diferencias estructurales.
3. Probar forma de salida, backward, cantidad de parámetros y memoria para las 16 combinaciones discretas, antes de iniciar Optuna.
4. Implementar y probar por separado **guardar mejor Pearson, activar paciencia después de la época 15, detener después de diez comprobaciones y restaurar mejores pesos**. Validar con secuencias sintéticas de Pearson, incluidas mejoras posteriores a la época 15 y detención por máximo 50.
5. Reproducir retrospectivamente la política en `history_A5.csv` y `history_A5b.csv`; confirmar las medias indicadas en el apartado 1.1 con el mismo simulador que se usará conceptualmente para la parada real.
6. Implementar `train_one_fold` y el objetivo como **media de cinco mejores checkpoints**, con un modelo/optimizador nuevos en cada fold, semillas y métricas registradas.
7. Crear los tres estudios SQLite; encolar los diez warmups por sujeto **una sola vez** y ejecutar secuencialmente hasta alcanzar 32 trials completos por sujeto.
8. Seleccionar el mejor A5b y el mejor A5 por sujeto; reentrenarlos con dos semillas adicionales en los cinco folds bajo la misma política.
9. Exportar reportes, curvas y comparación retrospectiva justa con ablaciones; elegir configuración final solo con resultados de desarrollo.
10. Dejar documentados versiones principales, hardware, fallos y posibles reinicios del sampler, sin hashes ni controles administrativos adicionales.

Comandos orientativos (los scripts pueden adaptar su CLI al repositorio):

```bash
python scripts/run_tpe.py --subject S1 --trials 32
python scripts/run_tpe.py --subject S2 --trials 32
python scripts/run_tpe.py --subject S3 --trials 32
python scripts/report_tpe.py
```

---

## 12. Referencias técnicas

- Optuna 5.0.0 — notas de versión y TPE multivariante: https://github.com/optuna/optuna/releases/tag/v5.0.0
- API de Optuna `TPESampler`: https://optuna.readthedocs.io/en/latest/reference/samplers/generated/optuna.samplers.TPESampler.html
- PyTorch — Adam y `decoupled_weight_decay`: https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html
- Lightning — definición de `min_delta`, `patience` y frecuencia de comprobación: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html
- BCI Competition IV — conjunto de datos: https://www.bbci.de/competition/iv/

**Resumen operativo:** TPE selecciona por sujeto la mejor configuración de A5b/A5 según la **media del mejor Pearson de los cinco folds**, con **Adam + L2**, `min_delta=0`, paciencia 10 después de la época 15 y máximo 50. La mejora atribuible a la optimización se analizará contra una referencia con **la misma política de checkpoints**, sin consultar el test oficial.

# TPE conjunto A5b/A5 — ECoG BCI IV-4

¿Cuánto mejora el Pearson macro de A5b y A5 al optimizar conjuntamente arquitectura (`kernel_size`, `encoder_levels`, `base_filters`), tasa de aprendizaje y L2 con **Optuna TPE**? Cada entrenamiento usa **early stopping con el mejor checkpoint** de validación.

La especificación completa es [docs/plan_implementacion_TPE_ECoG.md](docs/plan_implementacion_TPE_ECoG.md) (v2.1; su sección 0 recoge los cambios de implementación). [configs/tpe.yaml](configs/tpe.yaml) es la única fuente de parámetros.

**Origen del código**
- El pipeline **P0** y los modelos son una copia literal de `ecog_ablations` (repositorio *Experimento Arquitecturas*, commit `043c44f`): limpieza, CMR, Morlet, escaladores, folds B1–B5, `EncoderDecoder1D`, overlap-add y Pearson.
- El flujo de búsqueda sigue el del proyecto anterior (*TT\Code*): un estudio Optuna por sujeto, SQLite con copia en Drive y notebook de Colab. Se ha quitado su maquinaria de hashes, manifiestos y puertas de aprobación.

**Retardo:** ECoG(t) → guante(t + **20 ms**) = **2 columnas** a 100 Hz, el mismo de las ablaciones.

## Campaña

| Etapa | Por sujeto | Entrenamientos (3 sujetos) |
|---|---|---:|
| Warmups equilibrados (5 pares × A5b/A5) | 10 trials × 5 folds | 150 |
| TPE guiado | 22 trials × 5 folds | 330 |
| Verificación (mejor A5b y mejor A5 × semillas 1042/2042) | 2 × 2 × 5 | 60 |
| **Total** | | **540** |

**Entrenamiento:**
- Adam + L2 acoplada (no AdamW), batch 128, pérdida `0.5·MSE + 0.5·(1 − coseno)`.
- Validación al final de cada época.
- Se conserva el checkpoint con **mayor Pearson macro** de las trayectorias reconstruidas (`min_delta = 0`).
- La paciencia de 10 empieza a contar en la época 16, con un máximo de 50 épocas; la primera parada posible es la 25.

**Objetivo de TPE:** media de los mejores Pearson de los cinco folds.

**Sampler:** `TPESampler(multivariate=True, n_startup_trials=10, constant_liar=False)`, semillas 2026/2027/2028, `NopPruner`, ejecución secuencial.

**Pesos:** no se guardan (como en las ablaciones). Cada fold conserva métricas y `predictions.npz` de su mejor checkpoint.

## Uso local

```bash
pip install -r requirements.txt
# los .mat (sub1_comp.mat, sub2_comp.mat, sub3_comp.mat) en data/raw/ o en la carpeta de ECOG_DATA_DIR
python -m pytest -q                                   # comprobaciones (~1 min en CPU)
python scripts/run_tpe.py --capacity                  # 16 combinaciones en la GPU → results/capacity/
python scripts/run_tpe.py --smoke                     # S1/B1, 2 trials × 2 épocas × 2 batches → results/_smoke/
python scripts/run_tpe.py --subjects S1 S2 S3         # búsqueda: 32 trials COMPLETE por sujeto (reanudable)
python scripts/run_tpe.py --subjects S1 S2 S3 --verify   # verificación de finalistas (reanudable)
python scripts/report_tpe.py                          # tablas y figuras → results/summary/
```

Otras opciones:
- `--trials N` cambia el número de trials COMPLETE.
- `--data-dir`, `--results-dir` y `--device` cambian rutas y dispositivo.
- `--work-dir` es el disco local para el SQLite de trabajo; por defecto, la carpeta temporal del sistema.

`report_tpe.py` funciona en cualquier momento. La simulación retrospectiva sobre las ablaciones (`reference/ablaciones/history_A5*.csv`) no necesita resultados de TPE.

En Windows, usa un entorno virtual y una carpeta de resultados **fuera de OneDrive**: OneDrive puede bloquear el reemplazo de archivos mientras sincroniza.

## Uso en Google Colab desde VS Code

El notebook [scripts/colab_tpe.ipynb](scripts/colab_tpe.ipynb) sigue el mismo esquema que el de las ablaciones: kernel remoto de la extensión *Google Colab*, código clonado de GitHub y datos y resultados en Drive.

1. Sube el repositorio a GitHub (los datos y `results/` están en `.gitignore`) y ajusta `GIT_URL` y `GIT_REF` en la celda 2. `DATA_DIR` puede apuntar a los `.mat` que ya usaron las ablaciones.
2. Ejecuta las celdas en orden:
   1. pytest;
   2. capacidad de la GPU;
   3. smoke;
   4. búsqueda en segundo plano;
   5. monitor;
   6. verificación;
   7. reporte.
3. **Paralelizar:** un runtime de Colab por sujeto (`SUBJECTS = ["S2"]`, etc.) sobre el mismo `RESULTS_DIR`. Nunca dos procesos sobre el mismo sujeto. Usa la misma clase de GPU en toda la campaña.

Reanudación:
- El estudio de cada sujeto se publica en `results/studies/<S>_study.zip` después de cada trial: base SQLite, sampler serializado y metadatos.
- Si el servidor se recicla o detienes el runner, al relanzar se restaura el zip.
- El trial que estaba en curso se repite con los **mismos parámetros**: si era un warmup, sigue en cola; si lo propuso TPE, el RNG del sampler continúa exactamente.
- Sus restos se archivan en `trials/<S>/_interrupted/`.

Fallos:
- Un trial con NaN, OOM u otro error queda FAIL con su causa (`error.txt`) y no cuenta para los 32.
- Un warmup fallido se reencola hasta 3 intentos.
- Tres FAIL seguidos detienen el proceso.

**Cómputo orientativo:** ≈ 9–10 h de GPU por sujeto más ≈ 1 h de verificación, unas 30 h en total, con ~32 épocas por fold de media según la retrospectiva. La celda de capacidad da la estimación con tu GPU.

## Resultados

```text
results/
├── studies/<S>_study.zip           estudio Optuna publicado (study.sqlite + sampler.pkl + meta.json)
├── trials/<S>/trial_NNN/           config.json, trial.json (estado y valor), error.txt si falló
│   ├── <fold>/                     status.json, history.csv (por época), metrics.json, predictions.npz
│   └── ../_interrupted/            intentos no terminados, archivados al reanudar
├── verification/<S>/               finalists.json; <A5b|A5>/seed_<base>/<fold>/ (mismos archivos)
├── capacity/                       capacity_<gpu>.csv
├── runners/  logs/                 latido del runner y logs del proceso en segundo plano
└── summary/                        tpe_trials.csv, tpe_folds.csv, tpe_verification.csv (+ _paired),
                                    tpe_selected_models.csv, ablation_retrospective*.csv,
                                    tpe_comparison*.csv, figures/
```

Cada trial guarda en el estudio (user_attrs) el resumen de sus folds:
- semilla, mejor época, épocas entrenadas y motivo de parada;
- Pearson por dedo, MSE, CCC y curva de validación;
- sesión, GPU y versiones.

Por eso los reportes se regeneran desde los zip aunque falte algún archivo en Drive.

## Estructura del código

| Módulo | Contenido |
|---|---|
| `src/ecog_tpe/data.py`, `cleaning.py`, `preprocessing.py`, `evaluation.py`, `models.py` | **copia literal** de `ecog_ablations` (P0, A5b/A5, overlap-add, métricas) |
| `src/ecog_tpe/config.py` | carga y validación de `configs/tpe.yaml`, semillas |
| `src/ecog_tpe/training.py` | Adam + L2, pérdida y determinismo (copiados); `EarlyStopping`, `simulate`, `train_fold` |
| `src/ecog_tpe/space.py` | espacio de búsqueda, warmups y `architecture_spec(k, L, f)` |
| `src/ecog_tpe/runs.py` | caché P0 por sujeto, `run_fold` y artefactos, lock del runner |
| `src/ecog_tpe/search.py` | estudio Optuna (zip, warmups, objetivo, bucle), capacidad, smoke |
| `src/ecog_tpe/verification.py` | finalistas y su reentrenamiento con dos semillas |
| `src/ecog_tpe/reporting.py` | tablas, retrospectiva, comparación con ablaciones, figuras y monitor |

## Límites de interpretación

- El objetivo usa el **máximo** de validación de cada fold, y luego se elige el mejor de 32 trials. Son métricas de selección, optimistas; no son rendimiento de test.
- Los finalistas se verifican en los **mismos** folds. Las tres semillas miden estabilidad de entrenamiento, no generalización independiente.
- Sin un Random Search con el mismo presupuesto, no se puede atribuir ventaja a TPE frente a otros buscadores.
- La comparación justa con las ablaciones usa la misma política de checkpoints:
  - la ablación con la parada simulada (`ablation_retrospective.csv`);
  - el warmup de referencia (par 1). Usa la misma arquitectura, LR, L2, semilla e inicialización que las ablaciones, así que sus curvas deberían reproducir las de `history_A5*.csv` con el mismo tipo de GPU (figura `reference_curves.png`).

## Notas de ejecución

Anota aquí cualquier corrección de software, reinicio relevante o trial repetido por ese motivo (fecha, cambio, trials afectados). Los reinicios del sampler también quedan en `tpe_trials.csv` (`session`, `sampler_restored`).

- _(sin incidencias todavía)_
#   T P E _ E C o G  
 
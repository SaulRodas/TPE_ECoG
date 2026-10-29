"""Entrenamiento común: Adam con L2 acoplada, pérdida 0.5·MSE + 0.5·(1 − coseno), semillas, determinismo y
early stopping con el mejor checkpoint de validación (plan §4).

Todo salvo ``EarlyStopping``, ``simulate`` y ``train_fold`` es copia de ecog_ablations (commit 043c44f).
"""

from __future__ import annotations

import inspect
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .evaluation import overlap_add, pearson_per_channel
from .preprocessing import FoldFeatures


def configure_torch(runtime: dict[str, Any]) -> None:
    """TF32 según la config; determinismo «cuando sea viable» (avisos en lugar de errores)."""
    allow = bool(runtime["allow_tf32"])
    torch.backends.cuda.matmul.allow_tf32 = allow
    torch.backends.cudnn.allow_tf32 = allow
    if runtime["deterministic"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_optimizer(model: torch.nn.Module, tcfg: dict[str, Any]) -> torch.optim.Optimizer:
    kwargs = dict(lr=tcfg["lr"], weight_decay=tcfg["weight_decay"], betas=tuple(tcfg["betas"]), eps=tcfg["eps"])
    if "decoupled_weight_decay" in inspect.signature(torch.optim.Adam.__init__).parameters:
        kwargs["decoupled_weight_decay"] = False  # L2 acoplada, no AdamW
    return torch.optim.Adam(model.parameters(), **kwargs)


def mse_cosine_loss(pred: torch.Tensor, target: torch.Tensor, lcfg: dict[str, Any]):
    """Coseno a lo largo de las posiciones temporales por dedo y ventana, promediado sobre dedos y batch."""
    mse = F.mse_loss(pred, target)
    cos = F.cosine_similarity(pred, target, dim=-1, eps=lcfg["cosine_eps"]).mean()
    loss = lcfg["mse_weight"] * mse + lcfg["cosine_weight"] * (1.0 - cos)
    return loss, mse, cos


# ----------------------------------------------------------------------------- early stopping


@dataclass
class EarlyStopping:
    """Mejor Pearson de validación (plan §4.1).

    Mejora = nuevo máximo estricto (``min_delta`` = 0). Las épocas 1..min_epochs guardan checkpoints pero no pueden
    detener el entrenamiento: el contador de paciencia se pone a cero al terminar la época ``min_epochs`` y solo
    cuenta desde la siguiente. Con los valores del plan la primera parada posible es la época 25.
    """

    patience: int = 10
    min_epochs: int = 15
    max_epochs: int = 50
    min_delta: float = 0.0
    best_score: float = float("-inf")
    best_epoch: int | None = None
    wait: int = 0

    @classmethod
    def from_config(cls, es: dict[str, Any], **overrides: Any) -> "EarlyStopping":
        """Instancia nueva desde ``training.early_stopping`` (``overrides`` solo para el smoke)."""
        params = {k: es[k] for k in ("patience", "min_epochs", "max_epochs", "min_delta")}
        return cls(**{**params, **overrides})

    def update(self, epoch: int, score: float) -> bool:
        """Registra la validación de ``epoch``; devuelve True si es un nuevo mejor checkpoint."""
        if not np.isfinite(score):
            raise ValueError(f"Pearson de validación no finito en la época {epoch}")
        improved = score > self.best_score + self.min_delta
        if improved:
            self.best_score, self.best_epoch, self.wait = float(score), epoch, 0
        elif epoch > self.min_epochs:
            self.wait += 1
        if epoch == self.min_epochs:
            self.wait = 0  # la paciencia empieza después del mínimo
        return improved

    def stop_reason(self, epoch: int) -> str | None:
        """Motivo de parada tras la validación de ``epoch`` (None = seguir entrenando)."""
        if epoch > self.min_epochs and self.wait >= self.patience:
            return "patience_exhausted"
        if epoch >= self.max_epochs:
            return "max_epochs"
        return None


def simulate(scores: Sequence[float], stopper: EarlyStopping) -> dict[str, Any]:
    """Aplica la política a una curva de validación ya observada (simulación retrospectiva, plan §1.1)."""
    epoch, reason = 0, None
    for epoch, score in enumerate(scores[: stopper.max_epochs], start=1):
        stopper.update(epoch, float(score))
        reason = stopper.stop_reason(epoch)
        if reason is not None:
            break
    return {"best_score": stopper.best_score, "best_epoch": stopper.best_epoch, "epochs_trained": epoch,
            "stop_reason": reason or "curve_ended"}


# ----------------------------------------------------------------------------- entrenamiento


@dataclass
class FoldResult:
    history: list[dict[str, Any]]
    y_pred_scaled: np.ndarray        # (dedos, columnas validadas) del mejor checkpoint restaurado, escala MinMax
    best_epoch: int
    best_score: float                # máximo de val_pearson_macro observado: score del fold para TPE (plan §5)
    restored_score: float            # Pearson macro recalculado tras restaurar los pesos de best_epoch
    epochs_trained: int
    stop_reason: str
    train_seconds: float


class _Windows:
    """Ventanas extraídas por indexado de vistas (sin copiar el tensor completo por ventana)."""

    def __init__(self, feats: FoldFeatures, device: torch.device):
        self.X = torch.from_numpy(feats.X).to(device).unfold(2, feats.window, 1)  # (bloques, F, T−W+1, W)
        self.Y = torch.from_numpy(feats.Y).to(device).unfold(2, feats.window, 1)
        self.delay = feats.delay

    def get(self, blocks: torch.Tensor, starts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[blocks, :, starts], self.Y[blocks, :, starts + self.delay]


@torch.no_grad()
def _validate(model, windows: _Windows, feats: FoldFeatures, device, lcfg, batch_size: int):
    model.eval()
    starts = torch.as_tensor(feats.val_starts, device=device)
    blocks = torch.full_like(starts, feats.val_block)
    preds, sums, n = [], torch.zeros(3, device=device, dtype=torch.float64), 0
    for i in range(0, len(starts), batch_size):
        xb, yb = windows.get(blocks[i:i + batch_size], starts[i:i + batch_size])
        pred = model(xb)
        loss, mse, cos = mse_cosine_loss(pred, yb, lcfg)
        sums += torch.stack([loss, mse, cos]).double() * len(xb)
        n += len(xb)
        preds.append(pred.float().cpu().numpy())
    model.train()
    y_rec = overlap_add(np.concatenate(preds), feats.val_starts, feats.valid_lo, feats.valid_hi)
    return (sums / n).tolist(), y_rec


def train_fold(
    model: torch.nn.Module,
    feats: FoldFeatures,
    tcfg: dict[str, Any],
    stopper: EarlyStopping,
    device: torch.device,
    seed: int,
    fingers: list[str],
    on_epoch: Callable[[dict[str, Any]], None] | None = None,
    max_batches: int | None = None,
) -> FoldResult:
    """Entrena con early stopping sobre el Pearson de validación y restaura el mejor checkpoint (plan §4).

    ``tcfg`` es la sección ``training`` con ``lr`` y ``weight_decay`` del trial. Se valida al final de cada época;
    una pérdida o un Pearson no finitos lanzan una excepción después de registrar la fila de la época.
    """
    lcfg, batch = tcfg["loss"], tcfg["batch_size"]
    windows = _Windows(feats, device)
    train_index = torch.from_numpy(feats.train_index).to(device)
    y_val = feats.Y[feats.val_block][:, feats.val_target_columns()]
    n_train = len(train_index)
    n_batches = n_train // batch if tcfg["drop_last"] else -(-n_train // batch)
    if max_batches is not None:
        n_batches = min(n_batches, max_batches)

    optimizer = make_optimizer(model, tcfg)
    order = torch.Generator().manual_seed(seed)  # mismo orden de batches en todos los trials del fold
    history, best_state, reason = [], None, None
    t_start = time.perf_counter()
    model.train()
    for epoch in range(1, stopper.max_epochs + 1):
        t0 = time.perf_counter()
        perm = torch.randperm(n_train, generator=order).to(device)
        sums = torch.zeros(3, device=device, dtype=torch.float64)
        for i in range(n_batches):
            sel = train_index[perm[i * batch:(i + 1) * batch]]
            xb, yb = windows.get(sel[:, 0], sel[:, 1])
            loss, mse, cos = mse_cosine_loss(model(xb), yb, lcfg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sums += torch.stack([loss.detach(), mse.detach(), cos.detach()]).double()
        train_loss, train_mse, train_cos = (sums / max(n_batches, 1)).tolist()
        t1 = time.perf_counter()
        (val_loss, val_mse, val_cos), y_rec = _validate(model, windows, feats, device, lcfg, batch)
        r, reasons = pearson_per_channel(y_val, y_rec)
        score = float(np.mean(r))
        if np.isfinite(train_loss) and np.isfinite(score) and stopper.update(epoch, score):
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        t2 = time.perf_counter()
        row = {
            "epoch": epoch, "train_loss": train_loss, "train_mse": train_mse, "train_cos": train_cos,
            "val_loss": val_loss, "val_mse": val_mse, "val_cos": val_cos,
            "val_pearson_macro": score,
            **{f"val_pearson_{f}": float(v) for f, v in zip(fingers, r)},
            "best_val_pearson_macro": stopper.best_score, "best_epoch": stopper.best_epoch, "wait": stopper.wait,
            "n_batches": n_batches, "train_seconds": t1 - t0, "val_seconds": t2 - t1, "epoch_seconds": t2 - t0,
        }
        history.append(row)
        if on_epoch is not None:
            on_epoch(row)
        if not np.isfinite(train_loss):
            raise FloatingPointError(f"pérdida de entrenamiento no finita en la época {epoch}")
        if not np.isfinite(score):
            detail = "; ".join(f"{f}: {why}" for f, why in zip(fingers, reasons) if why) or "Pearson no finito"
            raise ValueError(f"Pearson de validación inválido en la época {epoch} ({detail})")
        reason = stopper.stop_reason(epoch)
        if reason is not None:
            break
    train_seconds = time.perf_counter() - t_start

    model.load_state_dict(best_state)  # las métricas finales salen del mejor checkpoint, no de la última época
    _, y_best = _validate(model, windows, feats, device, lcfg, batch)
    restored = float(np.mean(pearson_per_channel(y_val, y_best)[0]))
    return FoldResult(history, y_best, stopper.best_epoch, stopper.best_score, restored, len(history), reason,
                      train_seconds)

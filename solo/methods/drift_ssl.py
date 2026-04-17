import math
from typing import Any, Dict, List, Sequence, Tuple

import omegaconf
import torch
import torch.nn as nn
from solo.losses.drift_ssl import drift_ssl_loss_func
from solo.methods.base import BaseMomentumMethod
from solo.utils.momentum import initialize_momentum_params


class TauScheduler:
    """Cosine decay schedule for the bandwidth parameter Tau."""

    def __init__(self, epochs: int, tau_max: float = 0.5, tau_min: float = 0.1):
        self.epochs = epochs
        self.tau_max = tau_max
        self.tau_min = tau_min

    def get_tau(self, current_epoch: int) -> float:
        progress = current_epoch / max(1, self.epochs)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.tau_min + (self.tau_max - self.tau_min) * cosine_decay


class DriftSSL(BaseMomentumMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements DriftSSL."""
        super().__init__(cfg)

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        pred_hidden_dim: int = cfg.method_kwargs.pred_hidden_dim

        # DriftSSL specific params
        self.tau_max: float = cfg.method_kwargs.get("tau_max", 0.5)
        self.tau_min: float = cfg.method_kwargs.get("tau_min", 0.1)
        self.tau_scheduler = TauScheduler(cfg.max_epochs, self.tau_max, self.tau_min)

        # projector (Online)
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_output_dim, bias=False),
            nn.BatchNorm1d(proj_output_dim)
        )

        # momentum projector (Target)
        self.momentum_projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_output_dim, bias=False),
            nn.BatchNorm1d(proj_output_dim)
        )
        initialize_momentum_params(self.projector, self.momentum_projector)

        # predictor (Online only)
        self.predictor = nn.Sequential(
            nn.Linear(proj_output_dim, pred_hidden_dim, bias=False),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden_dim, proj_output_dim)
        )

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        cfg = super(DriftSSL, DriftSSL).add_and_assert_specific_cfg(cfg)
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.pred_hidden_dim")
        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        extra_learnable_params = [
            {"name": "projector", "params": self.projector.parameters()},
            {"name": "predictor", "params": self.predictor.parameters()},
        ]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        return super().momentum_pairs + extra_momentum_pairs

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        out = super().forward(X)
        z = self.projector(out["feats"])
        p = self.predictor(z)
        out.update({"z": z, "p": p})
        return out

    @torch.no_grad()
    def momentum_forward(self, X: torch.Tensor) -> Dict:
        out = super().momentum_forward(X)
        z = self.momentum_projector(out["feats"])
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for DriftSSL."""
        # batch[0] contains the instance indices (labels)
        indexes = batch[0]

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]

        # In solo-learn, out["p"] and out["momentum_z"] are lists of tensors (one per crop).
        # We concatenate them to get shape [2B, D] as expected by our loss function.
        p_all = torch.cat(out["p"][:self.num_large_crops], dim=0)
        z_target_all = torch.cat(out["momentum_z"][:self.num_large_crops], dim=0)

        # Repeat the indices for the number of crops to match shapes [2B]
        labels = indexes.repeat(self.num_large_crops)

        # Get current Tau from scheduler
        current_tau = self.tau_scheduler.get_tau(self.current_epoch)

        # Calculate DriftSSL Loss
        drift_loss, drift_mag = drift_ssl_loss_func(
            p=p_all,
            z_target=z_target_all,
            labels=labels,
            tau=current_tau
        )

        metrics = {
            "train_drift_loss": drift_loss,
            "train_drift_magnitude": drift_mag,
            "train_tau": current_tau
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return drift_loss + class_loss

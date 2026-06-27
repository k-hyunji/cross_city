# models/trainer.py

import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.model import DualModalNet
from models.losses import TotalLoss


class Trainer:
    def __init__(self, cfg, model, train_loader, val_loader, device="cuda"):
        self.cfg          = cfg
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device

        t = cfg["training"]
        self.epochs   = t["epochs"]
        self.use_adv  = cfg.get("model", {}).get("use_adv", False)

        self.criterion = TotalLoss(cfg).to(device)

        params = (list(model.parameters()) +
                  [p for p in self.criterion.parameters() if p.requires_grad])
        self.optimizer = torch.optim.Adam(
            params, lr=float(t["lr"]), weight_decay=float(t["weight_decay"])
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs
        )

        os.makedirs(cfg["logging"]["log_dir"],        exist_ok=True)
        os.makedirs(cfg["logging"]["checkpoint_dir"], exist_ok=True)

        self.best_val_loss = float("inf")
        self.history       = []

    def _lam_adv(self, epoch: int) -> float:
        """DANN lambda ramp: 0 → 1 over training."""
        p = epoch / max(self.epochs - 1, 1)
        return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        keys   = ["total", "contrast", "dis", "align", "proto", "cross_city", "adv", "vicreg"]
        totals = {k: 0.0 for k in keys}

        lam = self._lam_adv(epoch) if self.use_adv else 0.0

        pbar = tqdm(self.train_loader,
                    desc=f"Ep {epoch+1}/{self.epochs}", leave=False)
        for batch in pbar:
            self.optimizer.zero_grad()
            outputs = self.model(batch, lam_adv=lam)
            losses  = self.criterion(outputs)

            losses["total"].backward()
            nn.utils.clip_grad_norm_(
                list(self.model.parameters()) +
                list(self.criterion.parameters()), 1.0)
            self.optimizer.step()

            for k in totals:
                totals[k] += losses[k].item()

            pbar.set_postfix(
                loss=f"{losses['total'].item():.4f}",
                cont=f"{losses['contrast'].item():.4f}",
                cc=f"{losses['cross_city'].item():.4f}",
                vic=f"{losses['vicreg'].item():.4f}",
                proto=f"{losses['proto'].item():.4f}",
            )

        n = len(self.train_loader)
        return {k: v / n for k, v in totals.items()}

    @torch.no_grad()
    def val_epoch(self, loader: DataLoader) -> dict:
        self.model.eval()
        keys   = ["total", "contrast", "dis", "align", "proto", "cross_city", "adv", "vicreg"]
        totals = {k: 0.0 for k in keys}

        for batch in loader:
            outputs = self.model(batch, lam_adv=0.0)
            losses  = self.criterion(outputs)
            for k in totals:
                totals[k] += losses[k].item()

        n = len(loader)
        return {k: v / n for k, v in totals.items()}

    def save_checkpoint(self, epoch: int, val_losses: dict, tag: str = "") -> str:
        path = os.path.join(
            self.cfg["logging"]["checkpoint_dir"], f"epoch_{epoch:03d}{tag}.pt"
        )
        torch.save({
            "epoch":           epoch,
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_losses":      val_losses,
            "cfg":             self.cfg,
        }, path)
        return path

    def run(self) -> list:
        t = self.cfg["training"]
        if t.get("use_clip_shared", False):
            loss_desc = "CLIPSharedCon (32-d) + CrossCitySharedLoss + L_dis + L_proto"
            if self.use_adv:
                loss_desc += " + DANN"
        elif t.get("use_clip", False):
            loss_desc = "L_clip (CLIP B×B) + L_dis + L_proto"
        elif t.get("use_unified_con", False):
            loss_desc = "L_unified (UnifiedCityCon 2B×2B) + L_dis + L_proto"
        else:
            loss_desc = "L_contrast + L_dis + L_align"
        print(f"[Trainer] {self.epochs} epochs")
        print(f"[Trainer] Loss: {loss_desc}\n")

        for epoch in range(self.epochs):
            tl = self.train_epoch(epoch)
            vl = self.val_epoch(self.val_loader)
            self.scheduler.step()

            self.history.append({
                "epoch": epoch + 1,
                **{f"train_{k}": v for k, v in tl.items()},
                **{f"val_{k}":   v for k, v in vl.items()},
            })

            print(
                f"Ep {epoch+1:3d} | "
                f"Train L={tl['total']:.4f} "
                f"(cont={tl['contrast']:.4f} "
                f"cc={tl['cross_city']:.4f} "
                f"vic={tl['vicreg']:.4f} "
                f"dis={tl['dis']:.4f} "
                f"proto={tl['proto']:.4f}"
                + (f" adv={tl['adv']:.4f}" if self.use_adv else "")
                + f") | Val L={vl['total']:.4f} "
                f"(cont={vl['contrast']:.4f} "
                f"cc={vl['cross_city']:.4f} "
                f"vic={vl['vicreg']:.4f} "
                f"proto={vl['proto']:.4f})"
            )

            if vl["total"] < self.best_val_loss:
                self.best_val_loss = vl["total"]
                path = self.save_checkpoint(epoch, vl, tag="_best")
                print(f"  >> Best: {path}")

            if (epoch + 1) % self.cfg["logging"]["save_every"] == 0:
                self.save_checkpoint(epoch, vl)

        print(f"\n[Trainer] Done. Best val loss: {self.best_val_loss:.4f}")
        return self.history

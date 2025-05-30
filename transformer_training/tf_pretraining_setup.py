# ===========================
# 3. Training Management (SequenceModel)
# ===========================

import os
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import wandb
import time
from typing import List, Tuple

class SequenceModel:
    def __init__(
        self,
        model: nn.Module,
        d_feat: int,
        d_model: int,
        n_epochs: int,
        lr: float,
        device: torch.device,
        input_feature_ranges: List[Tuple[int,int]],
        lambda_contrastive: float = 0.1,
        temperature: float = 0.1,
        save_path: str = "model/",
        save_prefix: str = "",
        ):
        """
        model:          your MASTER (or similar) that maps [N,T,d_feat]→[N,T,d_model]
        d_feat:         input feature dim
        d_model:        transformer output dim
        lambda_contrastive: weight on the InfoNCE loss
        temperature:    temperature for InfoNCE
        """
        self.device = device
        self.model = model.to(device)
        # store your slice specs
        self.input_ranges = input_feature_ranges
        # a small projection to turn input d_feat→d_model as our reconstruction target
        self.input_proj = nn.Linear(d_feat, d_model).to(device)

        # losses & hyperparams
        self.recon_loss          = nn.MSELoss()
        self.lambda_contrastive  = lambda_contrastive
        self.temperature         = temperature

        # optimizer over both model & projection head
        self.optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.input_proj.parameters()),
            lr=lr
        )

        self.n_epochs   = n_epochs
        self.save_path  = save_path
        self.save_prefix= save_prefix
        os.makedirs(save_path, exist_ok=True)
        
        # bookkeeping of the best checkpoint
        self.best_val        = float('inf')
        self.best_epoch      = None
        self.best_model_path = None

    def train_epoch(self, train_loader):
        self.model.train()
        losses = []
        for batch in train_loader:
            # batch: [N, T, raw_feat_total]
            batch = batch.to(self.device).float()
            #print(f"Batch_shape:{batch.shape}")
            N, T, _ = batch.shape

            self.optimizer.zero_grad()

            # 1) get embeddings from transformer
            output = self.model(batch)  # [N, T, d_model]

            # 2) slice out exactly the input features
            src_chunks = [batch[:, :, lo:hi] for (lo, hi) in self.input_ranges]
            # now src has shape [N, T, d_feat]
            src = torch.cat(src_chunks, dim=-1)

            # 3) project src → d_model as reconstruction target
            target    = self.input_proj(src)   # [N, T, d_model]
            L_recon   = self.recon_loss(output, target)

            # 4) temporal contrastive on last two time‐steps
            if N > 1 and T >= 2:
                z_anchor = F.normalize(output[:, -2, :], dim=1)  # [N, d_model]
                z_pos    = F.normalize(output[:, -1, :], dim=1)  # [N, d_model]
                logits   = torch.matmul(z_anchor, z_pos.t()) / self.temperature
                labels   = torch.arange(N, device=self.device)
                L_ctr    = F.cross_entropy(logits, labels)
            else:
                L_ctr = torch.tensor(0.0, device=self.device)

            # 5) combine losses, backprop, step
            loss = L_recon + self.lambda_contrastive * L_ctr
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        return float(np.mean(losses))


    def evaluate(self, val_loader):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device).float()
                N, T, _ = batch.shape

                output = self.model(batch)             # [N, T, d_model]
                # slice+concat to get the same d_feat input
                src_chunks = [batch[:, :, lo:hi] for (lo,hi) in self.input_ranges]
                src        = torch.cat(src_chunks, dim=-1)
                target     = self.input_proj(src)  # [N, T, d_model]
                L_recon = self.recon_loss(output, target)

                if N > 1 and T >= 2:
                    z_anchor = F.normalize(output[:, -2, :], dim=1)
                    z_pos    = F.normalize(output[:, -1, :], dim=1)
                    logits   = torch.matmul(z_anchor, z_pos.t()) / self.temperature
                    labels   = torch.arange(N, device=self.device)
                    L_ctr    = F.cross_entropy(logits, labels)
                else:
                    L_ctr = torch.tensor(0.0, device=self.device)

                losses.append((L_recon + self.lambda_contrastive * L_ctr).item())

        return float(np.mean(losses))

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader:   torch.utils.data.DataLoader,
    ):
        train_losses = []
        val_losses   = []
        self.best_val = float('inf')
        
        run_id = wandb.run.id
        
        for epoch in range(1, self.n_epochs + 1):
            # 1) train + eval
            t0 = time.time()
            tr_loss = self.train_epoch(train_loader)
            vl_loss = self.evaluate(val_loader)
            epoch_time = time.time() - t0
            
            train_losses.append(tr_loss)
            val_losses.append(vl_loss)

            print(f"Epoch {epoch}/{self.n_epochs} — "
                  f"Train: {tr_loss:.6f}, Val: {vl_loss:.6f}"
                  f"Time: {epoch_time:.1f}s")

            # — log to WandB
            wandb.log({
              "train_loss": tr_loss,
              "val_loss":   vl_loss,
              "epoch_time": epoch_time
            }, step=epoch)
            
            # checkpoint best
            if vl_loss < self.best_val:
                self.best_val        = vl_loss
                self.best_epoch      = epoch
                best_path = os.path.join(
                    self.save_path,f"{self.save_prefix}_{run_id}_best.pth")
                self.save_model(best_path)
                self.best_model_path = best_path

        # 3) at the end, always save a “final” copy
        # final_path = os.path.join(
        #     self.save_path, f"{self.save_prefix}_final.pth"
        # )
        # self.save_model(final_path)
        
        return train_losses, val_losses

    def save_model(self, path: str):
        """Dump both the transformer and the reconstruction head."""
        torch.save({
            "model":      self.model.state_dict(),
            "input_proj": self.input_proj.state_dict()
        }, path)

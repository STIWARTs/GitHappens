
"""
Improved residual seq2seq model for Flipkart Gridlock.

Key upgrades:
- Residual prediction instead of absolute demand
- Real inference-style masking
- Larger transformer
- Learnable positional embeddings
- Removed sigmoid bottleneck
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

SEQ_LEN = 152
PRED_START = 96 + 9
PRED_END = 96 + 55


class ResidualCurveDataset(Dataset):
    def __init__(self, curves, statics, repeats=2):
        self.curves = curves.astype(np.float32)
        self.statics = statics.astype(np.float32)
        self.repeats = repeats

    def __len__(self):
        return len(self.curves) * self.repeats

    def __getitem__(self, idx):
        i = idx % len(self.curves)

        y = self.curves[i].copy()

        x = y.copy()
        obs = np.ones_like(x, dtype=np.float32)

        # REAL submission masking
        x[PRED_START:PRED_END + 1] = 0.0
        obs[PRED_START:PRED_END + 1] = 0.0

        target_mask = np.zeros_like(x, dtype=np.float32)
        target_mask[PRED_START:PRED_END + 1] = 1.0

        return x, obs, self.statics[i], y, target_mask


class ImprovedSeqModel(nn.Module):
    def __init__(self, static_dim, d_model=96, nhead=8, layers=3):
        super().__init__()

        self.value_proj = nn.Linear(6, d_model)
        self.static_proj = nn.Linear(static_dim, d_model)

        self.pos_emb = nn.Parameter(
            torch.randn(SEQ_LEN, d_model) * 0.02
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu"
        )

        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=layers
        )

        # REMOVED sigmoid
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

        pos = np.arange(SEQ_LEN)
        slot = pos % 96
        day49 = (pos >= 96).astype(np.float32)

        feats = np.stack([
            np.sin(2 * np.pi * slot / 96),
            np.cos(2 * np.pi * slot / 96),
            day49,
            slot / 95.0
        ], axis=1).astype(np.float32)

        self.register_buffer(
            "time_feats",
            torch.tensor(feats)
        )

    def forward(self, values, observed, statics):
        b = values.shape[0]

        time_feats = self.time_feats.unsqueeze(0).expand(b, -1, -1)

        token = torch.cat([
            values.unsqueeze(-1),
            observed.unsqueeze(-1),
            time_feats
        ], dim=-1)

        h = self.value_proj(token)

        h = h + self.static_proj(statics).unsqueeze(1)

        h = h + self.pos_emb.unsqueeze(0)

        h = self.encoder(h)

        return self.head(h).squeeze(-1)


def train_model(curves, statics, epochs=20, batch_size=64, device="cuda"):
    ds = ResidualCurveDataset(curves, statics)

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True
    )

    model = ImprovedSeqModel(
        static_dim=statics.shape[1]
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    model.train()

    for ep in range(epochs):
        losses = []

        for x, obs, st, y, mask in dl:
            x = x.to(device)
            obs = obs.to(device)
            st = st.to(device)
            y = y.to(device)
            mask = mask.to(device)

            pred = model(x, obs, st)

            loss = (
                ((pred - y) ** 2) * mask
            ).sum() / mask.sum()

            opt.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            opt.step()

            losses.append(float(loss.detach().cpu()))

        print(f"epoch={ep+1} loss={np.mean(losses):.6f}")

    return model


@torch.no_grad()
def predict(model, curves, statics, device="cuda"):
    model.eval()

    x = curves.copy()

    obs = np.ones_like(x, dtype=np.float32)

    x[:, PRED_START:PRED_END+1] = 0.0
    obs[:, PRED_START:PRED_END+1] = 0.0

    preds = []

    bs = 128

    for i in range(0, len(x), bs):
        p = model(
            torch.tensor(x[i:i+bs], device=device),
            torch.tensor(obs[i:i+bs], device=device),
            torch.tensor(statics[i:i+bs], device=device)
        )

        preds.append(
            p.detach().cpu().numpy()
        )

    pred = np.vstack(preds)

    pred = np.clip(pred, 0, 1)

    return pred

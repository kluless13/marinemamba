"""
BarcodeMamba fine-tuning for marine fish barcodes.
Handles Models C, D, E from the experiment matrix.

Usage (run in Colab with GPU):
    # Model C: Direct transfer (insect -> fish)
    python scripts/04_barcodemamba_models.py --mode transfer --data-dir data/processed

    # Model D: Pretrain from scratch on fish, then fine-tune
    python scripts/04_barcodemamba_models.py --mode scratch --data-dir data/processed

    # Model E: Domain-adapt insect model on fish, then fine-tune
    python scripts/04_barcodemamba_models.py --mode adapt --data-dir data/processed
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from tqdm import tqdm

# ── Tokenizer (character-level, matching BarcodeMamba) ──────────────────────

CHAR_VOCAB = {"[MASK]": 0, "[SEP]": 1, "[UNK]": 2, "A": 3, "C": 4, "G": 5, "T": 6, "N": 7}
VOCAB_SIZE = 8
MAX_SEQ_LEN = 660


def tokenize(seq, max_len=MAX_SEQ_LEN):
    tokens = [CHAR_VOCAB.get(ch, CHAR_VOCAB["N"]) for ch in seq.upper()]
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    pad_len = max_len - len(tokens)
    return [CHAR_VOCAB["N"]] * pad_len + tokens


# ── Dataset ─────────────────────────────────────────────────────────────────

class BarcodeDataset(Dataset):
    def __init__(self, csv_path, label_to_idx=None, label_col="species_name"):
        df = pd.read_csv(csv_path)
        self.seqs = df["nucleotides"].tolist()
        self.labels_raw = df[label_col].tolist()

        if label_to_idx is None:
            unique = sorted(set(self.labels_raw))
            self.label_to_idx = {lab: i for i, lab in enumerate(unique)}
        else:
            self.label_to_idx = label_to_idx

        self.labels = [self.label_to_idx.get(lab, 0) for lab in self.labels_raw]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        x = torch.tensor(tokenize(self.seqs[idx]), dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    @property
    def n_classes(self):
        return len(self.label_to_idx)


class PretrainDataset(Dataset):
    """NTP pretraining dataset — returns (input[:-1], target[1:])."""
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.seqs = df["nucleotides"].tolist()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        tokens = tokenize(self.seqs[idx])
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


# ── Model ───────────────────────────────────────────────────────────────────

def build_model(n_classes=None, pretrain=False):
    """Build BarcodeMamba model. Requires mamba_ssm to be installed."""
    # Clone BarcodeMamba repo if not present
    if not os.path.exists("BarcodeMamba"):
        os.system("git clone https://github.com/bioscan-ml/BarcodeMamba.git")
    sys.path.insert(0, "BarcodeMamba")

    from utils.barcode_mamba import BarcodeMamba

    config = {
        "d_model": 384,
        "n_layer": 2,
        "d_inner": 384 * 4,
        "vocab_size": VOCAB_SIZE,
        "resid_dropout": 0.0,
        "embed_dropout": 0.1,
        "residual_in_fp32": True,
        "pad_vocab_size_multiple": 8,
        "mamba_ver": "mamba2",
        "layer": {
            "d_model": 384,
            "d_state": 64,
            "d_conv": 4,
            "expand": 2,
            "headdim": 48,
        },
    }

    if pretrain:
        config["n_classes"] = VOCAB_SIZE
        config["use_head"] = "pretrain"
    else:
        config["n_classes"] = n_classes
        config["use_head"] = "finetune"

    return BarcodeMamba(**config)


def load_pretrained_weights(model, ckpt_path):
    """Load weights from a Lightning checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    cleaned = {k.replace("model.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"  Loaded weights from {ckpt_path}")
    print(f"  Missing (new head): {len(missing)} keys")
    print(f"  Unexpected (old head): {len(unexpected)} keys")
    return model


def download_insect_checkpoint(dest_dir="checkpoints/insect"):
    """Download BarcodeMamba insect pretrained weights from HuggingFace."""
    from huggingface_hub import hf_hub_download
    os.makedirs(dest_dir, exist_ok=True)
    ckpt_path = hf_hub_download(
        repo_id="bioscan-ml/BarcodeMamba",
        filename="BarcodeMamba-dim384-layer2-char/checkpoints/last.ckpt",
        local_dir=dest_dir,
    )
    print(f"  Downloaded insect checkpoint: {ckpt_path}")
    return ckpt_path


# ── Lightning Modules ───────────────────────────────────────────────────────

class PretrainModule(pl.LightningModule):
    def __init__(self, model, lr=8e-4):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.1)
        return opt


class FinetuneModule(pl.LightningModule):
    def __init__(self, model, lr=8e-4):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage):
        x, y = batch
        logits = self(x).squeeze()
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/acc", acc, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.1)


# ── Evaluation ──────────────────────────────────────────────────────────────

def extract_embeddings(model, dataset, batch_size=128, device="cuda"):
    """Extract mean-pooled backbone embeddings."""
    model.eval()
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeds, labels = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Extracting embeddings"):
            h = model.get_hidden_states(x.to(device))  # (B, seq, d_model)
            h = h.mean(dim=1)  # (B, d_model)
            embeds.append(h.cpu().numpy())
            labels.append(y.numpy())
    return np.vstack(embeds), np.concatenate(labels)


def evaluate_knn(model, train_ds, unseen_ds, device="cuda"):
    """kNN probe on unseen genera."""
    X_train, y_train = extract_embeddings(model, train_ds, device=device)
    X_unseen, y_unseen = extract_embeddings(model, unseen_ds, device=device)

    knn = KNeighborsClassifier(n_neighbors=1, metric="cosine")
    knn.fit(X_train, y_train)
    preds = knn.predict(X_unseen)

    acc = accuracy_score(y_unseen, preds)
    bacc = balanced_accuracy_score(y_unseen, preds)
    return {"genus_accuracy_unseen": float(acc), "genus_balanced_accuracy_unseen": float(bacc)}


def evaluate_linear_probe(model, train_ds, test_ds, d_model=384, epochs=200, device="cuda"):
    """Linear probing: frozen backbone + trained linear head."""
    X_train, y_train = extract_embeddings(model, train_ds, device=device)
    X_test, y_test = extract_embeddings(model, test_ds, device=device)

    # Normalize
    mean, std = X_train.mean(), X_train.std()
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)

    n_classes = len(np.unique(y_train))
    clf = nn.Linear(d_model, n_classes).to(device)
    opt = torch.optim.SGD(clf.parameters(), lr=1.0, momentum=0.95, weight_decay=1e-10)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    ds = torch.utils.data.TensorDataset(X_t, y_t)
    dl = DataLoader(ds, batch_size=1024, shuffle=True)

    clf.train()
    for epoch in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(clf(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    clf.eval()
    X_te = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = clf(X_te).argmax(dim=-1).cpu().numpy()
    acc = accuracy_score(y_test, preds)
    return {"linear_probe_accuracy": float(acc)}


# ── Main Pipeline ───────────────────────────────────────────────────────────

def run_pretrain(data_dir, ckpt_dir, lr=8e-4, max_epochs=50, from_checkpoint=None):
    """Pretrain with NTP objective."""
    print("\n=== PRETRAINING ===")
    ds = PretrainDataset(os.path.join(data_dir, "pre_training.csv"))
    val_size = min(2000, len(ds) // 10)
    train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds) - val_size, val_size])

    model = build_model(pretrain=True)
    if from_checkpoint:
        model = load_pretrained_weights(model, from_checkpoint)

    lit = PretrainModule(model, lr=lr)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu", devices=1,
        gradient_clip_val=1.0,
        default_root_dir=ckpt_dir,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor="val/loss", mode="min", save_last=True),
            pl.callbacks.EarlyStopping(monitor="val/loss", patience=3, mode="min"),
        ],
    )
    trainer.fit(
        lit,
        DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2),
        DataLoader(val_ds, batch_size=32, num_workers=2),
    )
    return os.path.join(ckpt_dir, "lightning_logs", "version_0", "checkpoints", "last.ckpt")


def run_finetune(data_dir, ckpt_dir, pretrained_ckpt, lr=8e-4, max_epochs=20):
    """Fine-tune with classification head."""
    print("\n=== FINE-TUNING ===")
    train_ds = BarcodeDataset(os.path.join(data_dir, "supervised_train.csv"))
    val_ds = BarcodeDataset(os.path.join(data_dir, "supervised_val.csv"), label_to_idx=train_ds.label_to_idx)
    test_ds = BarcodeDataset(os.path.join(data_dir, "supervised_test.csv"), label_to_idx=train_ds.label_to_idx)

    n_classes = train_ds.n_classes
    print(f"  N classes: {n_classes}")

    model = build_model(n_classes=n_classes, pretrain=False)
    if pretrained_ckpt:
        model = load_pretrained_weights(model, pretrained_ckpt)

    lit = FinetuneModule(model, lr=lr)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu", devices=1,
        gradient_clip_val=1.0,
        default_root_dir=ckpt_dir,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor="val/loss", mode="min", save_last=True),
            pl.callbacks.EarlyStopping(monitor="val/loss", patience=3, mode="min"),
        ],
    )
    trainer.fit(
        lit,
        DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2),
        DataLoader(val_ds, batch_size=32, num_workers=2),
    )
    results = trainer.test(lit, DataLoader(test_ds, batch_size=32, num_workers=2))
    return model, train_ds, test_ds, results


def run_full_evaluation(model, data_dir, train_ds, test_ds, device="cuda"):
    """Run all evaluation protocols."""
    print("\n=== EVALUATION ===")
    results = {}

    # Linear probe
    print("  Linear probing...")
    lp = evaluate_linear_probe(model, train_ds, test_ds, device=device)
    results.update(lp)
    print(f"  Linear probe accuracy: {lp['linear_probe_accuracy']:.4f}")

    # kNN on unseen genera
    unseen_ds = BarcodeDataset(
        os.path.join(data_dir, "unseen.csv"),
        label_to_idx=None,  # Fresh labels for genus
        label_col="genus_name",
    )
    train_genus_ds = BarcodeDataset(
        os.path.join(data_dir, "supervised_train.csv"),
        label_to_idx=None,
        label_col="genus_name",
    )

    print("  kNN probing on unseen genera...")
    knn = evaluate_knn(model, train_genus_ds, unseen_ds, device=device)
    results.update(knn)
    print(f"  Genus accuracy (unseen): {knn['genus_accuracy_unseen']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["transfer", "scratch", "adapt"], required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-pretrain-epochs", type=int, default=50)
    parser.add_argument("--max-finetune-epochs", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    mode_names = {"transfer": "model_c", "scratch": "model_d", "adapt": "model_e"}
    model_name = mode_names[args.mode]
    ckpt_dir = f"checkpoints/{model_name}"

    if args.mode == "transfer":
        # Model C: Load insect checkpoint, fine-tune directly
        print("=" * 60)
        print("MODEL C: Direct Transfer (insect -> fish)")
        print("=" * 60)
        insect_ckpt = download_insect_checkpoint()
        model, train_ds, test_ds, ft_results = run_finetune(
            args.data_dir, ckpt_dir, insect_ckpt, max_epochs=args.max_finetune_epochs
        )

    elif args.mode == "scratch":
        # Model D: Pretrain from scratch, then fine-tune
        print("=" * 60)
        print("MODEL D: Pretrain from Scratch")
        print("=" * 60)
        pretrain_ckpt = run_pretrain(
            args.data_dir, ckpt_dir, lr=8e-4, max_epochs=args.max_pretrain_epochs
        )
        model, train_ds, test_ds, ft_results = run_finetune(
            args.data_dir, ckpt_dir, pretrain_ckpt, max_epochs=args.max_finetune_epochs
        )

    elif args.mode == "adapt":
        # Model E: Load insect checkpoint, continue pretrain on fish, then fine-tune
        print("=" * 60)
        print("MODEL E: Domain Adaptation (insect -> fish pretrain -> fine-tune)")
        print("=" * 60)
        insect_ckpt = download_insect_checkpoint()
        adapted_ckpt = run_pretrain(
            args.data_dir, ckpt_dir, lr=2e-4,  # Lower LR for adaptation
            max_epochs=20, from_checkpoint=insect_ckpt,
        )
        model, train_ds, test_ds, ft_results = run_finetune(
            args.data_dir, ckpt_dir, adapted_ckpt, max_epochs=args.max_finetune_epochs
        )

    # Full evaluation
    eval_results = run_full_evaluation(model, args.data_dir, train_ds, test_ds, device=device)

    # Save
    all_results = {
        "mode": args.mode,
        "model_name": model_name,
        "finetune_test": ft_results[0] if ft_results else {},
        **eval_results,
    }
    output_path = os.path.join(args.output_dir, f"{model_name}_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {model_name}")
    print(f"{'=' * 60}")
    print(json.dumps(all_results, indent=2))
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()

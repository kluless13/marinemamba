"""
Experiment: Phylogenetic Embeddings + BarcodeMamba

Hypothesis: If we train the model to place sequences in a space where distances
match phylogenetic/taxonomic distances (not just classify species), the embeddings
will capture hierarchical structure that generalizes to unseen genera.

Method (adapted from Stalder et al. 2025, improved):
  1. Compute pairwise taxonomic distances from our taxonomy (rank-based)
  2. Learn embedding vectors per species where cosine dist ≈ taxonomic dist
  3. Train BarcodeMamba to map COI sequences into this embedding space
  4. At inference: sequence → embedding → nearest species → that species' family
  5. Optional: add contrastive loss for same-family clustering

Usage:
    python3 scripts/08_phylo_embeddings.py --data-dir data/processed --output-dir results
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "BarcodeMamba")

# ── Tokenizer ────────────────────────────────────────────────────────────────

CHAR_VOCAB = {"[MASK]": 0, "[SEP]": 1, "[UNK]": 2, "A": 3, "C": 4, "G": 5, "T": 6, "N": 7}
MAX_SEQ_LEN = 660


def tokenize(seq):
    tokens = [CHAR_VOCAB.get(ch, CHAR_VOCAB["N"]) for ch in seq.upper()]
    if len(tokens) > MAX_SEQ_LEN:
        tokens = tokens[:MAX_SEQ_LEN]
    return [CHAR_VOCAB["N"]] * (MAX_SEQ_LEN - len(tokens)) + tokens


# ── Taxonomic Distance ───────────────────────────────────────────────────────

RANK_LEVELS = {
    "species": 0,
    "genus": 1,
    "family": 2,
    "order": 3,
    "class": 4,
    "phylum": 5,
}


def compute_taxonomic_distance(tax_a, tax_b):
    """Compute rank-based taxonomic distance between two species.

    Returns distance 0-5 based on lowest common rank:
      same species = 0, same genus = 1, same family = 2,
      same order = 3, same class = 4, different phylum = 5
    """
    if tax_a.get("species") and tax_a["species"] == tax_b.get("species"):
        return 0.0
    if tax_a.get("genus") and tax_a["genus"] == tax_b.get("genus"):
        return 1.0
    if tax_a.get("family") and tax_a["family"] == tax_b.get("family"):
        return 2.0
    if tax_a.get("order") and tax_a["order"] == tax_b.get("order"):
        return 3.0
    if tax_a.get("class") and tax_a["class"] == tax_b.get("class"):
        return 4.0
    return 5.0


def build_species_taxonomy(df):
    """Build species → taxonomy lookup from dataframe."""
    tax = {}
    for _, row in df.iterrows():
        sp = row.get("species_name", "")
        if sp and sp not in tax:
            tax[sp] = {
                "species": sp,
                "genus": row.get("genus_name", ""),
                "family": row.get("family_name", ""),
                "order": row.get("order_name", ""),
                "class": row.get("class_name", ""),
                "phylum": row.get("phylum_name", ""),
            }
    return tax


# ── Step 1: Learn Phylogenetic Embeddings ────────────────────────────────────

def learn_phylo_embeddings(species_list, species_tax, embed_dim=256, n_epochs=200,
                            lr=0.01, n_negatives=32, device="cuda"):
    """Learn embedding vectors where cosine distance ≈ taxonomic distance."""
    n_species = len(species_list)
    sp_to_idx = {sp: i for i, sp in enumerate(species_list)}

    print(f"  Learning phylogenetic embeddings for {n_species} species (dim={embed_dim})...")

    # Learnable embeddings
    embeddings = nn.Embedding(n_species, embed_dim).to(device)
    nn.init.normal_(embeddings.weight, std=0.1)
    optimizer = torch.optim.AdamW(embeddings.parameters(), lr=lr, weight_decay=1e-4)

    # Precompute taxonomy for fast distance lookup
    tax_list = [species_tax.get(sp, {}) for sp in species_list]

    for epoch in range(n_epochs):
        total_loss = 0
        # Sample random anchor species
        anchors = torch.randperm(n_species)[:min(512, n_species)]

        for anchor_idx in anchors:
            anchor_idx = anchor_idx.item()
            anchor_tax = tax_list[anchor_idx]

            # Sample negatives: mix of close and distant species
            neg_indices = torch.randint(0, n_species, (n_negatives,))
            all_indices = torch.cat([torch.tensor([anchor_idx]), neg_indices]).to(device)

            embs = embeddings(all_indices)
            anchor_emb = embs[0:1]  # (1, dim)
            other_embs = embs[1:]   # (n_neg, dim)

            # Cosine distances
            cos_dists = 1 - F.cosine_similarity(anchor_emb.expand_as(other_embs), other_embs)

            # Target taxonomic distances (normalized to 0-1)
            target_dists = torch.tensor([
                compute_taxonomic_distance(anchor_tax, tax_list[idx.item()]) / 5.0
                for idx in neg_indices
            ], dtype=torch.float32, device=device)

            # MSE loss between cosine distance and taxonomic distance
            loss = F.mse_loss(cos_dists, target_dists)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            avg_loss = total_loss / len(anchors)
            print(f"    Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.6f}")

    # Normalize embeddings
    with torch.no_grad():
        emb_matrix = embeddings.weight.data
        emb_matrix = F.normalize(emb_matrix, dim=1)

    return emb_matrix.cpu().numpy(), sp_to_idx


# ── Step 2: Train BarcodeMamba with Phylo Regression ─────────────────────────

class PhyloDataset(Dataset):
    """Dataset that returns (tokenized_seq, target_embedding)."""
    def __init__(self, df, phylo_embeddings, sp_to_idx):
        self.seqs = df["nucleotides"].tolist()
        self.species = df["species_name"].tolist()
        self.phylo_embeddings = phylo_embeddings
        self.sp_to_idx = sp_to_idx

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        x = torch.tensor(tokenize(self.seqs[idx]), dtype=torch.long)
        sp = self.species[idx]
        sp_idx = self.sp_to_idx.get(sp, 0)
        target = torch.tensor(self.phylo_embeddings[sp_idx], dtype=torch.float32)
        return x, target


class PhyloMamba(nn.Module):
    """BarcodeMamba backbone + phylogenetic embedding projection head."""
    def __init__(self, backbone, d_model=384, embed_dim=256):
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, embed_dim),
        )

    def forward(self, x):
        h = self.backbone.get_hidden_states(x)  # (B, seq, d_model)
        h = h.mean(dim=1)  # (B, d_model)
        return self.proj(h)  # (B, embed_dim)

    def get_embeddings(self, x):
        """Get normalized embeddings for inference."""
        emb = self.forward(x)
        return F.normalize(emb, dim=1)


class PhyloContrastiveLoss(nn.Module):
    """Combined phylo regression + contrastive loss."""
    def __init__(self, alpha=1.0, beta=0.5, temperature=0.07):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def forward(self, pred_emb, target_emb, family_labels):
        # Regression loss: match phylogenetic embeddings
        pred_norm = F.normalize(pred_emb, dim=1)
        target_norm = F.normalize(target_emb, dim=1)
        regression_loss = 1 - F.cosine_similarity(pred_norm, target_norm).mean()

        # Contrastive loss: same family = close, different family = far
        if self.beta > 0 and len(pred_emb) > 1:
            sim_matrix = torch.mm(pred_norm, pred_norm.t()) / self.temperature
            # Create family mask: 1 if same family, 0 if different
            family_mask = (family_labels.unsqueeze(0) == family_labels.unsqueeze(1)).float()
            family_mask.fill_diagonal_(0)  # Exclude self

            # InfoNCE-style contrastive loss
            pos_sum = (sim_matrix * family_mask).sum(dim=1)
            neg_sum = (sim_matrix * (1 - family_mask - torch.eye(len(pred_emb), device=pred_emb.device))).sum(dim=1)

            n_pos = family_mask.sum(dim=1).clamp(min=1)
            contrastive_loss = -torch.log(
                (pos_sum / n_pos).exp() / ((pos_sum / n_pos).exp() + neg_sum.exp() + 1e-8)
            ).mean()
        else:
            contrastive_loss = torch.tensor(0.0, device=pred_emb.device)

        return self.alpha * regression_loss + self.beta * contrastive_loss


def build_backbone(pretrain_ckpt=None):
    """Build BarcodeMamba backbone."""
    if not os.path.exists("BarcodeMamba"):
        os.system("git clone https://github.com/bioscan-ml/BarcodeMamba.git")

    from utils.barcode_mamba import BarcodeMamba

    config = {
        "d_model": 384, "n_layer": 2, "d_inner": 384 * 4,
        "vocab_size": 8, "resid_dropout": 0.0, "embed_dropout": 0.1,
        "residual_in_fp32": True, "pad_vocab_size_multiple": 8,
        "mamba_ver": "mamba2", "n_classes": 8, "use_head": "pretrain",
        "layer": {"d_model": 384, "d_state": 64, "d_conv": 4, "expand": 2, "headdim": 48},
    }

    model = BarcodeMamba(**config)

    if pretrain_ckpt:
        ckpt = torch.load(pretrain_ckpt, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        cleaned = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned, strict=False)
        print(f"  Loaded backbone from {pretrain_ckpt}")

    return model


def train_phylo_model(model, train_ds, val_ds, family_to_idx, embed_dim=256,
                       epochs=30, lr=5e-4, batch_size=64, device="cuda",
                       use_contrastive=True):
    """Train PhyloMamba model."""
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    loss_fn = PhyloContrastiveLoss(
        alpha=1.0,
        beta=0.5 if use_contrastive else 0.0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.to(device)
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0, 0
        for x, target_emb in tqdm(train_dl, desc=f"  Epoch {epoch+1}/{epochs}", leave=False):
            x = x.to(device)
            target_emb = target_emb.to(device)

            pred_emb = model(x)

            # Get family labels for contrastive loss
            # (approximate: use target embedding clustering)
            family_labels = torch.zeros(len(x), dtype=torch.long, device=device)
            if use_contrastive:
                # Cluster by target embedding similarity as proxy for family
                with torch.no_grad():
                    target_sim = torch.mm(
                        F.normalize(target_emb, dim=1),
                        F.normalize(target_emb, dim=1).t()
                    )
                    family_labels = (target_sim > 0.8).long().sum(dim=1)

            loss = loss_fn(pred_emb, target_emb, family_labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0
        val_n = 0
        with torch.no_grad():
            for x, target_emb in val_dl:
                x, target_emb = x.to(device), target_emb.to(device)
                pred_emb = model(x)
                cos_sim = F.cosine_similarity(
                    F.normalize(pred_emb, dim=1),
                    F.normalize(target_emb, dim=1),
                ).mean()
                val_loss += (1 - cos_sim.item()) * len(x)
                val_n += len(x)

        avg_train = total_loss / n_batches
        avg_val = val_loss / val_n
        print(f"  Epoch {epoch+1}/{epochs}: train_loss={avg_train:.4f} val_cos_dist={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), "results/phylo_model_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load("results/phylo_model_best.pt", weights_only=True))
    return model


# ── Step 3: Evaluate ─────────────────────────────────────────────────────────

def extract_model_embeddings(model, sequences, batch_size=128, device="cuda"):
    """Extract normalized embeddings from PhyloMamba."""
    model.eval()
    model.to(device)

    class SeqDataset(Dataset):
        def __init__(self, seqs):
            self.seqs = seqs
        def __len__(self):
            return len(self.seqs)
        def __getitem__(self, idx):
            return torch.tensor(tokenize(self.seqs[idx]), dtype=torch.long)

    dl = DataLoader(SeqDataset(sequences), batch_size=batch_size, shuffle=False, num_workers=4)
    embeds = []
    with torch.no_grad():
        for x in tqdm(dl, desc="  Extracting embeddings"):
            emb = model.get_embeddings(x.to(device))
            embeds.append(emb.cpu().numpy())

    return np.vstack(embeds)


def hierarchical_eval(pred_genera, true_genera, tax_map):
    """Evaluate at genus, family, order levels."""
    results = {}
    for level in ["genus", "family", "order"]:
        if level == "genus":
            true_l = true_genera
            pred_l = pred_genera
        else:
            true_l = [tax_map.get(g, {}).get(level, "") for g in true_genera]
            pred_l = [tax_map.get(g, {}).get(level, "") for g in pred_genera]

        valid = [(t, p) for t, p in zip(true_l, pred_l) if t and p]
        if valid:
            t, p = zip(*valid)
            acc = accuracy_score(t, p)
            n_correct = sum(a == b for a, b in valid)
            results[level] = {"accuracy": float(acc), "n_correct": n_correct, "n_total": len(valid)}
            print(f"  {level.title()}: {acc:.4f} ({n_correct}/{len(valid)})")
        else:
            results[level] = {"accuracy": 0.0, "n_correct": 0, "n_total": 0}
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--phylo-epochs", type=int, default=200)
    parser.add_argument("--train-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-contrastive", action="store_true")
    parser.add_argument("--pretrain-ckpt", type=str, default=None,
                        help="Path to pretrained BarcodeMamba checkpoint")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    data_dir = Path(args.data_dir)
    train_df = pd.read_csv(data_dir / "supervised_train.csv")
    val_df = pd.read_csv(data_dir / "supervised_val.csv")
    test_df = pd.read_csv(data_dir / "supervised_test.csv")
    unseen_df = pd.read_csv(data_dir / "unseen.csv")

    print("=" * 60)
    print("PHYLOGENETIC EMBEDDINGS + BARCODEMAMBA")
    print("=" * 60)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)} | Unseen: {len(unseen_df)}")
    print(f"Embed dim: {args.embed_dim} | Contrastive: {not args.no_contrastive}")

    # Build species taxonomy
    all_df = pd.concat([train_df, val_df, test_df, unseen_df])
    species_tax = build_species_taxonomy(all_df)
    species_list = sorted(species_tax.keys())
    print(f"Species in taxonomy: {len(species_list)}")

    # Build genus → family/order map for eval
    genus_tax_map = {}
    for _, row in all_df.iterrows():
        g = row.get("genus_name", "")
        if g and g not in genus_tax_map:
            genus_tax_map[g] = {
                "family": row.get("family_name", ""),
                "order": row.get("order_name", ""),
            }

    # Step 1: Learn phylogenetic embeddings
    print("\n[1/4] Learning phylogenetic embeddings...")
    phylo_embeddings, sp_to_idx = learn_phylo_embeddings(
        species_list, species_tax,
        embed_dim=args.embed_dim,
        n_epochs=args.phylo_epochs,
        device=device,
    )
    np.save(os.path.join(args.output_dir, "phylo_embeddings.npy"), phylo_embeddings)
    print(f"  Phylo embeddings shape: {phylo_embeddings.shape}")

    # Step 2: Build and train PhyloMamba
    print("\n[2/4] Building PhyloMamba model...")

    # Try to find a pretrained checkpoint
    ckpt = args.pretrain_ckpt
    if not ckpt:
        for candidate in [
            "checkpoints/model_d/lightning_logs/version_0/checkpoints/last.ckpt",
            "checkpoints/model_e/lightning_logs/version_0/checkpoints/last.ckpt",
        ]:
            if os.path.exists(candidate):
                ckpt = candidate
                break

    backbone = build_backbone(pretrain_ckpt=ckpt)
    model = PhyloMamba(backbone, d_model=384, embed_dim=args.embed_dim)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    # Build family label mapping for contrastive loss
    families = sorted(set(train_df["family_name"].dropna().unique()))
    family_to_idx = {f: i for i, f in enumerate(families)}

    train_ds = PhyloDataset(train_df, phylo_embeddings, sp_to_idx)
    val_ds = PhyloDataset(val_df, phylo_embeddings, sp_to_idx)

    print("\n[3/4] Training PhyloMamba...")
    model = train_phylo_model(
        model, train_ds, val_ds, family_to_idx,
        embed_dim=args.embed_dim,
        epochs=args.train_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        use_contrastive=not args.no_contrastive,
    )

    # Step 3: Extract embeddings and evaluate
    print("\n[4/4] Evaluating...")

    X_train = extract_model_embeddings(model, train_df["nucleotides"].tolist(), device=device)
    X_test = extract_model_embeddings(model, test_df["nucleotides"].tolist(), device=device)
    X_unseen = extract_model_embeddings(model, unseen_df["nucleotides"].tolist(), device=device)

    # Species classification (1-NN cosine)
    print("\n=== SPECIES (1-NN cosine) ===")
    knn_sp = KNeighborsClassifier(n_neighbors=1, metric="cosine", n_jobs=-1)
    knn_sp.fit(X_train, train_df["species_name"])
    species_acc = float(accuracy_score(test_df["species_name"], knn_sp.predict(X_test)))
    print(f"  Species accuracy: {species_acc:.4f}")

    # Unseen genus evaluation
    print("\n=== HIERARCHICAL ZERO-SHOT ===")
    knn_g = KNeighborsClassifier(n_neighbors=1, metric="cosine", n_jobs=-1)
    knn_g.fit(X_train, train_df["genus_name"])
    genus_preds = knn_g.predict(X_unseen).tolist()
    true_genera = unseen_df["genus_name"].tolist()

    hierarchical = hierarchical_eval(genus_preds, true_genera, genus_tax_map)

    # Save
    results = {
        "experiment": "phylo_embeddings",
        "embed_dim": args.embed_dim,
        "contrastive": not args.no_contrastive,
        "pretrain_ckpt": ckpt or "none",
        "species_accuracy": species_acc,
        "hierarchical_unseen": hierarchical,
    }

    output_path = os.path.join(args.output_dir, "phylo_embeddings_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("PHYLOGENETIC EMBEDDINGS RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()

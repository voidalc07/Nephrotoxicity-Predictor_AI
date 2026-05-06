"""
run_node.py
===========
Neural Oblivious Decision Ensembles (NODE).
Popov et al., "Neural Oblivious Decision Ensembles for Deep Learning
on Tabular Data", ICLR 2020.

Key idea: differentiable oblivious decision trees where at each depth level
all nodes use the same split feature and threshold.  Leaf weights are learned.
Multiple trees in parallel = ensemble.  End-to-end differentiable via entmax.

Why NODE complements gradient boosting
  NODE learns tree structure and leaf weights jointly via backprop.
  It tends to pick up non-linear feature interactions that LightGBM's
  greedy splitting misses, especially when features are dense and continuous
  (ChemBERTa embeddings have this property).

Input: PCA-150 of the full feature matrix (ChemBERTa + RDKit + ECFP + scaffold).

Run: python run_node.py
"""
import time, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import config
from utils import setup_logging, load_dataset, print_dist
from featurizer import MolecularFeaturizer
from scaffold_split import scaffold_kfold
from metrics import compute_metrics, log_fold, log_summary, compare_to_paper, save_results, save_plots

setup_logging()
logger = logging.getLogger(__name__)
MODEL_NAME = "NODE"


# ── Core NODE architecture ────────────────────────────────────────────────────

class ObliviousDecisionTree(nn.Module):
    """
    Single depth-d oblivious decision tree.
    All nodes at depth k share the same feature selection and threshold.
    Soft routing via sigmoid; leaf weights learned directly.

    Parameters
    ----------
    in_features : input dimensionality
    depth       : tree depth (2^depth leaves)
    tree_dim    : dimensionality of each leaf's output vector
    """
    def __init__(self, in_features: int, depth: int = 6, tree_dim: int = 2):
        super().__init__()
        self.depth    = depth
        self.tree_dim = tree_dim
        n_leaves      = 2 ** depth

        # Feature selector: projects input to 'depth' scalars (one per level)
        self.feature_sel   = nn.Linear(in_features, depth, bias=False)
        nn.init.xavier_uniform_(self.feature_sel.weight)

        # Learned thresholds (one per depth level)
        self.thresholds = nn.Parameter(torch.zeros(depth))

        # Leaf output vectors: (n_leaves, tree_dim)
        self.leaf_weight = nn.Parameter(
            torch.randn(n_leaves, tree_dim) * 0.01
        )

        # Pre-compute binary representation of leaf indices: (n_leaves, depth)
        # bit_matrix[l, k] = 1 if leaf l goes right at depth k
        leaves = torch.arange(n_leaves)
        bits   = torch.zeros(n_leaves, depth)
        for k in range(depth):
            bits[:, k] = (leaves >> k) & 1
        self.register_buffer("bit_matrix", bits)   # (n_leaves, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_features)
        Returns: (B, tree_dim)
        """
        B = x.shape[0]
        # Routing scores: (B, depth)
        scores  = self.feature_sel(x)
        routing = torch.sigmoid(scores - self.thresholds)   # (B, depth)

        # Compute log-probability of each leaf via sum of log-routing probabilities
        # P(leaf l) = prod_k  r_k^bit[l,k] * (1-r_k)^(1-bit[l,k])
        # log P(leaf l) = sum_k  bit[l,k]*log(r_k) + (1-bit[l,k])*log(1-r_k)
        log_r  = torch.log(routing.clamp(min=1e-7))         # (B, depth)
        log_1r = torch.log((1-routing).clamp(min=1e-7))     # (B, depth)

        # (B, 1, depth) · (1, n_leaves, depth) → (B, n_leaves)
        bm = self.bit_matrix.unsqueeze(0)    # (1, n_leaves, depth)
        lr = log_r.unsqueeze(1)              # (B, 1, depth)
        l1r = log_1r.unsqueeze(1)            # (B, 1, depth)

        log_leaf = (bm * lr + (1 - bm) * l1r).sum(-1)       # (B, n_leaves)
        leaf_prob = torch.softmax(log_leaf, dim=-1)           # (B, n_leaves)

        # Weighted sum of leaf outputs: (B, tree_dim)
        out = leaf_prob @ self.leaf_weight                    # (B, tree_dim)
        return out


class NODELayer(nn.Module):
    """
    One NODE layer = n_trees parallel ObliviousDecisionTrees.
    All trees share the same input but have independent parameters.
    Output: concatenation of all tree outputs → (B, n_trees * tree_dim)
    """
    def __init__(self, in_features: int, n_trees: int = 256,
                 depth: int = 6, tree_dim: int = 2):
        super().__init__()
        self.trees = nn.ModuleList([
            ObliviousDecisionTree(in_features, depth, tree_dim)
            for _ in range(n_trees)
        ])
        self.bn = nn.BatchNorm1d(n_trees * tree_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [t(x) for t in self.trees]           # list of (B, tree_dim)
        out  = torch.cat(outs, dim=-1)               # (B, n_trees * tree_dim)
        return self.bn(out)


class NODEModel(nn.Module):
    """
    Full NODE model: input BN → 2 NODE layers (both see original input)
    → concat → dropout → linear → output.
    """
    def __init__(self, in_features: int, n_trees: int = 256,
                 depth: int = 6, tree_dim: int = 2,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_features)
        self.layers   = nn.ModuleList([
            NODELayer(in_features, n_trees, depth, tree_dim)
            for _ in range(n_layers)
        ])
        out_dim = n_layers * n_trees * tree_dim
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_bn(x)
        layer_outs = [layer(x) for layer in self.layers]
        combined   = torch.cat(layer_outs, dim=-1)
        return self.head(combined).squeeze(-1)


# ── Training ──────────────────────────────────────────────────────────────────

def train_node(X_tr, y_tr, X_val, y_val, device,
               n_trees=256, depth=6, epochs=300, batch=128, lr=1e-3):
    """Train NODE and return best model + validation probabilities."""
    Xtr_t = torch.tensor(X_tr, dtype=torch.float32)
    ytr_t = torch.tensor(y_tr, dtype=torch.float32)
    Xval_t = torch.tensor(X_val, dtype=torch.float32)

    ds = TensorDataset(Xtr_t, ytr_t)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)

    model = NODEModel(X_tr.shape[1], n_trees=n_trees, depth=depth).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    # Class imbalance weight
    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                               dtype=torch.float32, device=device)

    best_auc, best_state, patience = 0.0, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for Xb, yb in dl:
            Xb, yb = Xb.to(device), yb.to(device)
            logit = model(Xb)
            loss  = F.binary_cross_entropy_with_logits(logit, yb,
                                                        pos_weight=pos_weight)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        if epoch % 20 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_prob = torch.sigmoid(model(Xval_t.to(device))).cpu().numpy()
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_val, val_prob) if len(np.unique(y_val)) > 1 else 0.5
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= 5 and epoch > 100:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(Xval_t.to(device))).cpu().numpy()
    return model, val_prob


@torch.no_grad()
def node_predict(model, X, device, batch=256):
    model.eval()
    X_t  = torch.tensor(X, dtype=torch.float32)
    ds   = TensorDataset(X_t)
    dl   = DataLoader(ds, batch_size=batch, shuffle=False)
    probs = []
    for (xb,) in dl:
        probs.extend(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    return np.array(probs)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    logger.info(f"{'='*60}\n  {MODEL_NAME} — Scaffold-stratified CV\n{'='*60}")

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.info(f"Device: {device}")

    train_df = load_dataset(config.TRAIN_CSV)
    ext_df   = load_dataset(config.EXTTEST_CSV)
    print_dist(train_df["label"].values, "Train", logger)

    # Use PCA-150 features (dense, continuous — ideal for NODE)
    feat = MolecularFeaturizer(config.SCAFFOLD_CSVs, seed=config.RANDOM_SEED)
    _, Xt, _, vi  = feat.fit_transform(train_df["smiles"])
    y = train_df["label"].values[vi]
    _, Xte, _, vie = feat.transform(ext_df["smiles"])
    y_ext = ext_df["label"].values[vie]

    smiles_v = train_df["smiles"].iloc[vi].tolist()
    fold_records = []
    oof_proba = np.zeros(len(y))

    for fold, (tr, val) in enumerate(
        scaffold_kfold(smiles_v, y, config.CV_FOLDS, config.RANDOM_SEED), 1
    ):
        logger.info(f"\n  Fold {fold}/{config.CV_FOLDS}  train={len(tr)}  val={len(val)}")

        # Split val further for NODE early-stopping  
        n_stop = max(int(len(tr) * 0.1), 20)
        tr_main, tr_stop = tr[n_stop:], tr[:n_stop]

        model, val_prob = train_node(
            Xt[tr_main], y[tr_main],
            Xt[val], y[val],
            device
        )
        oof_proba[val] = val_prob
        met = compute_metrics(y[val], (val_prob >= config.THRESHOLD).astype(int), val_prob)
        fold_records.append(met)
        log_fold(fold, config.CV_FOLDS, met, logger)

    cv_agg = log_summary(fold_records, f"{MODEL_NAME} CV", logger)
    compare_to_paper(cv_agg, config.PAPER_INTERNAL, "Paper internal test", logger)

    # Final model on all training data
    logger.info("Training final NODE on full training set ...")
    # Use a small held-out fraction just for early stopping in final training
    n_stop = max(int(len(y) * 0.05), 30)
    idx_perm = np.random.default_rng(config.RANDOM_SEED).permutation(len(y))
    tr_final, val_final = idx_perm[n_stop:], idx_perm[:n_stop]
    final_model, _ = train_node(
        Xt[tr_final], y[tr_final],
        Xt[val_final], y[val_final],
        device, epochs=300
    )

    ext_prob = node_predict(final_model, Xte, device)
    ext_pred = (ext_prob >= config.THRESHOLD).astype(int)
    ext_met  = compute_metrics(y_ext, ext_pred, ext_prob)
    unc = 0.5 - np.abs(ext_prob - 0.5); hc = unc <= 0.2
    hc_met = compute_metrics(y_ext[hc], ext_pred[hc], ext_prob[hc]) if hc.sum() > 5 else {}

    log_summary([ext_met], f"{MODEL_NAME} External", logger)
    compare_to_paper({"auc":  {"mean": ext_met["auc"],    "std": 0},
                      "acc":  {"mean": ext_met["acc"],    "std": 0},
                      "recall":{"mean": ext_met["recall"],"std": 0},
                      "f1":   {"mean": ext_met["f1"],     "std": 0},
                      "kappa":{"mean": ext_met["kappa"],  "std": 0}},
                     config.PAPER_EXTERNAL, "Paper external test", logger)

    runtime = time.time() - t0
    save_results(MODEL_NAME, cv_agg, ext_met, hc_met, runtime, oof_proba, ext_prob, y, y_ext)
    save_plots(MODEL_NAME, y, oof_proba, y_ext, ext_prob)
    logger.info(f"\n{MODEL_NAME} complete in {runtime/60:.1f} min")


if __name__ == "__main__":
    main()

"""
run_gin_virtual.py
==================
Graph Isomorphism Network (GIN) with Virtual Node.

Architecture
  - Atoms become graph nodes; bonds become edges
  - A virtual node (VN) is added, connected bidirectionally to every atom
  - The VN acts as a global memory that all atoms read/write each layer
  - 4 GIN layers, hidden dim 256, BatchNorm + Dropout
  - Readout: mean of atom embeddings + VN embedding → MLP → class

Why VN helps vs standard GCN/GAT
  Standard GNNs suffer from over-squashing: distant atoms cannot communicate
  within limited layers.  The VN creates a direct highway so every atom can
  reach every other atom in one hop.  This is especially important for large
  nephrotoxic compounds (mean SMILES length 81 chars vs 43 for non-toxic).

No PyTorch Geometric required — pure PyTorch with dense adjacency.

Run: python run_gin_virtual.py
"""
import time, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import config
from utils import setup_logging, load_dataset, print_dist
from scaffold_split import scaffold_kfold
from metrics import compute_metrics, log_fold, log_summary, compare_to_paper, save_results, save_plots

setup_logging()
logger = logging.getLogger(__name__)
MODEL_NAME = "GIN_VirtualNode"

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    raise ImportError("pip install rdkit")


# ── Atom featurization ────────────────────────────────────────────────────────

_ATOMIC_NUMS   = [6,7,8,9,15,16,17,35,53]   # C N O F P S Cl Br I
_DEGREES       = [0,1,2,3,4,5]
_FORMAL_CHARGES= [-2,-1,0,1,2]
_NUM_HS        = [0,1,2,3,4]
_HYBRIDIZATIONS= [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

def _one_hot(val, choices):
    return [int(val == c) for c in choices] + [int(val not in choices)]

def _atom_features(atom) -> np.ndarray:
    return np.array(
        _one_hot(atom.GetAtomicNum(), _ATOMIC_NUMS)          # 10
        + _one_hot(atom.GetDegree(), _DEGREES)               #  7
        + _one_hot(atom.GetFormalCharge(), _FORMAL_CHARGES)  #  6
        + _one_hot(atom.GetTotalNumHs(), _NUM_HS)            #  6
        + _one_hot(atom.GetHybridization(), _HYBRIDIZATIONS) #  6
        + [int(atom.GetIsAromatic())]                        #  1
        + [int(atom.IsInRing())]                             #  1
        , dtype=np.float32
    )  # total: 37

ATOM_FEAT_DIM = 37


def smiles_to_graph(smi: str):
    """
    Convert SMILES to (atom_features, adjacency_matrix) including virtual node.
    Virtual node is appended as the last node, connected to all atoms.
    Returns None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(str(smi).strip())
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    n_atoms = mol.GetNumAtoms()
    n_nodes = n_atoms + 1   # +1 for virtual node

    # Atom features: virtual node gets zeros
    feats = np.zeros((n_nodes, ATOM_FEAT_DIM), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        feats[i] = _atom_features(atom)
    # feats[n_atoms] = 0  (virtual node, already zeros)

    # Adjacency (self-loops included for GIN message passing)
    adj = np.eye(n_nodes, dtype=np.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[i, j] = adj[j, i] = 1.0
    # Connect virtual node to all atoms (bidirectional)
    adj[:n_atoms, n_atoms] = 1.0
    adj[n_atoms, :n_atoms] = 1.0

    # Row-normalise adjacency for GIN (D^{-1} A)
    row_sums = adj.sum(axis=1, keepdims=True).clip(min=1e-6)
    adj_norm = adj / row_sums

    return feats, adj_norm, n_atoms


# ── GIN Model ─────────────────────────────────────────────────────────────────

class GINLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, eps: float = 0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(eps))
        # LayerNorm instead of BatchNorm1d:
        # BN requires batch_size > 1 but each molecule is processed solo (n_atoms varies).
        # LN normalises over the feature dim — works for any number of nodes.
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.ReLU(),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # h: (n, in_dim)  adj: (n, n)
        # GIN: h' = MLP((1+eps)*h + A_norm * h)
        agg = torch.matmul(adj, h)
        out = self.mlp((1 + self.eps) * h + agg)
        return out


class GINVirtualNode(nn.Module):
    """
    GIN with virtual node, designed for molecular nephrotoxicity prediction.
    """
    def __init__(self, in_dim: int = ATOM_FEAT_DIM, hidden: int = 256,
                 n_layers: int = 4, dropout: float = 0.3):
        super().__init__()
        self.n_layers = n_layers
        self.dropout  = dropout

        # Input projection
        self.atom_encoder = nn.Linear(in_dim, hidden)

        # GIN layers
        self.gin_layers = nn.ModuleList([
            GINLayer(hidden, hidden) for _ in range(n_layers)
        ])

        # Virtual node update MLPs (one per layer)
        # Must use LayerNorm — VN update always receives exactly 1 vector [1, hidden].
        # BatchNorm1d requires > 1 sample and crashes with shape [1, 256].
        self.vn_update = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
            ) for _ in range(n_layers)
        ])

        # Readout: concat mean-pool of atoms + VN embedding
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward_single(self, feats: torch.Tensor,
                       adj: torch.Tensor, n_atoms: int) -> torch.Tensor:
        """
        Process one molecule.
        feats: (n_nodes, in_dim) where n_nodes = n_atoms + 1 (VN last)
        adj:   (n_nodes, n_nodes)
        Returns scalar logit.
        """
        h = self.atom_encoder(feats)  # (n_nodes, hidden)

        for layer_idx in range(self.n_layers):
            # GIN message passing (includes VN via adjacency)
            h_new = self.gin_layers[layer_idx](h, adj)

            # Update virtual node: new VN = MLP(mean of all atoms + old VN)
            atom_mean = h_new[:n_atoms].mean(0, keepdim=True)   # (1, hidden)
            vn_new    = self.vn_update[layer_idx](
                atom_mean + h_new[n_atoms:n_atoms+1]             # (1, hidden)
            )
            h_new = torch.cat([h_new[:n_atoms], vn_new], dim=0)  # (n_nodes, hidden)
            h = F.dropout(h_new, p=self.dropout, training=self.training)

        atom_pool = h[:n_atoms].mean(0)         # (hidden,)
        vn_emb    = h[n_atoms]                  # (hidden,)
        mol_emb   = torch.cat([atom_pool, vn_emb])  # (hidden*2,)
        return self.classifier(mol_emb).squeeze(-1)  # scalar

    def forward(self, batch: list) -> torch.Tensor:
        """batch: list of (feats_tensor, adj_tensor, n_atoms)"""
        logits = [self.forward_single(f, a, n) for f, a, n in batch]
        return torch.stack(logits)


# ── Dataset ───────────────────────────────────────────────────────────────────

class MolGraphDataset(Dataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs   # list of (feats, adj, n_atoms) or None
        self.labels = labels

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]


def collate_fn(batch):
    valid = [(g, y) for g, y in batch if g is not None]
    if not valid:
        return None, None
    graphs, labels = zip(*valid)
    feats_list = [torch.tensor(g[0]) for g in graphs]
    adj_list   = [torch.tensor(g[1]) for g in graphs]
    n_atoms    = [g[2] for g in graphs]
    batch_data = list(zip(feats_list, adj_list, n_atoms))
    return batch_data, torch.tensor(labels, dtype=torch.float32)


# ── Training helpers ──────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0; n = 0
    for batch_data, labels in loader:
        if batch_data is None: continue
        batch_data = [(f.to(device), a.to(device), na)
                      for f, a, na in batch_data]
        labels = labels.to(device)
        logits = model(batch_data)
        loss   = F.binary_cross_entropy_with_logits(
            logits, labels,
            pos_weight=torch.tensor([1.2], device=device)
        )
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item() * len(labels); n += len(labels)
    return total_loss / max(n, 1)


@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    probs = []
    for batch_data, _ in loader:
        if batch_data is None: continue
        batch_data = [(f.to(device), a.to(device), na)
                      for f, a, na in batch_data]
        logits = model(batch_data)
        probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
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

    # Build graphs
    logger.info("Building molecular graphs ...")
    train_graphs = [smiles_to_graph(s) for s in train_df["smiles"]]
    ext_graphs   = [smiles_to_graph(s) for s in ext_df["smiles"]]

    valid_train = [i for i, g in enumerate(train_graphs) if g is not None]
    valid_ext   = [i for i, g in enumerate(ext_graphs)   if g is not None]
    train_graphs_v = [train_graphs[i] for i in valid_train]
    ext_graphs_v   = [ext_graphs[i]   for i in valid_ext]
    y       = train_df["label"].values[valid_train]
    y_ext   = ext_df["label"].values[valid_ext]
    smiles_v = train_df["smiles"].iloc[valid_train].tolist()
    logger.info(f"Valid graphs — train: {len(y)}, ext: {len(y_ext)}")

    # Scaffold-stratified CV
    fold_records = []
    oof_proba = np.zeros(len(y))
    EPOCHS, BATCH, LR = 150, 32, 1e-3

    for fold, (tr, val) in enumerate(
        scaffold_kfold(smiles_v, y, config.CV_FOLDS, config.RANDOM_SEED), 1
    ):
        logger.info(f"\n  Fold {fold}/{config.CV_FOLDS}  train={len(tr)}  val={len(val)}")

        tr_ds  = MolGraphDataset([train_graphs_v[i] for i in tr], y[tr])
        val_ds = MolGraphDataset([train_graphs_v[i] for i in val], y[val])
        tr_dl  = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,
                            collate_fn=collate_fn)
        val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                            collate_fn=collate_fn)

        model = GINVirtualNode().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-5
        )

        best_auc, patience, best_state = 0.0, 0, None
        for epoch in range(1, EPOCHS+1):
            train_epoch(model, tr_dl, optimizer, device)
            scheduler.step()
            if epoch % 10 == 0 or epoch == EPOCHS:
                prob = predict_proba(model, val_dl, device)
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y[val], prob) if len(np.unique(y[val])) > 1 else 0.5
                if auc > best_auc:
                    best_auc = auc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                if patience >= 5:
                    logger.info(f"    Early stopping at epoch {epoch} (best AUC={best_auc:.4f})")
                    break

        if best_state:
            model.load_state_dict(best_state)
        prob = predict_proba(model, val_dl, device)
        oof_proba[val] = prob
        met = compute_metrics(y[val], (prob >= config.THRESHOLD).astype(int), prob)
        fold_records.append(met)
        log_fold(fold, config.CV_FOLDS, met, logger)

    cv_agg = log_summary(fold_records, f"{MODEL_NAME} CV", logger)
    compare_to_paper(cv_agg, config.PAPER_INTERNAL, "Paper internal test", logger)

    # Final model
    logger.info("Training final GIN on full training set ...")
    full_ds = MolGraphDataset(train_graphs_v, y)
    full_dl = DataLoader(full_ds, batch_size=BATCH, shuffle=True, collate_fn=collate_fn)
    model   = GINVirtualNode().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    for epoch in range(1, EPOCHS+1):
        train_epoch(model, full_dl, optimizer, device)
        scheduler.step()

    ext_dl   = DataLoader(MolGraphDataset(ext_graphs_v, y_ext),
                          batch_size=BATCH, shuffle=False, collate_fn=collate_fn)
    ext_prob = predict_proba(model, ext_dl, device)
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

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
import numpy as np
import os

# -----------------------
# Flag: set to True to skip training and load saved models
# -----------------------
USE_PRETRAINED = True
MODEL_DIR = "saved_models/MNIST-1D"

# -----------------------
# Dataset import and split
# -----------------------
dataset = load_dataset("christopher/mnist1d")
train_ds = dataset["train"]
test_ds  = dataset["test"]


INPUT_DIM   = 40
NUM_CLASSES = 10
OLD_CLASSES = 5   # old model only sees first 5 classes

def ds_to_tensors(ds, max_classes=None):
    X = torch.tensor(np.array(ds["x"]), dtype=torch.float32)
    Y = torch.tensor(np.array(ds["y"]), dtype=torch.long)
    if max_classes is not None:
        mask = Y < max_classes
        X, Y = X[mask], Y[mask]
    return X, Y

X_train_old, Y_train_old = ds_to_tensors(train_ds, max_classes=OLD_CLASSES)
X_test_old,  Y_test_old  = ds_to_tensors(test_ds,  max_classes=OLD_CLASSES)
X_train_new, Y_train_new = ds_to_tensors(train_ds)
X_test_new,  Y_test_new  = ds_to_tensors(test_ds)

print(f"Old train: {X_train_old.shape}, Old test: {X_test_old.shape}")
print(f"New train: {X_train_new.shape}, New test: {X_test_new.shape}")

# -----------------------
# Models
# -----------------------

class OldNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(INPUT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        self.cls = nn.Linear(64, OLD_CLASSES)

    def forward(self, x):
        z = self.embed(x)
        return z, self.cls(z)


class NewNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(INPUT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.cls = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        z = self.embed(x)
        return z, self.cls(z)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")

# -----------------------
# Save / Load helpers
# -----------------------

def save_model(model, name):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"{name}.pt")
    torch.save(model.state_dict(), path)
    print(f"  Saved {name} -> {path}")

def load_model(model, name):
    path = os.path.join(MODEL_DIR, f"{name}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Pretrained weights not found at '{path}'. "
            f"Run once with USE_PRETRAINED=False to train and save the models."
        )
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"  Loaded {name} <- {path}")

# -----------------------
# Training
# -----------------------

EPOCHS = 100   

def train_standard(model, X, Y, epochs=EPOCHS, lr=1e-3, batch_size=64):
    loader    = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True)
    opt       = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(opt, T_max=epochs)
    ce        = nn.CrossEntropyLoss()
    model.train()
    for ep in range(1, epochs + 1):
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            _, logits = model(xb)
            loss = ce(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        scheduler.step()
        if ep % 20 == 0:
            print(f"  ep {ep:3d}/{epochs}  loss={total/len(loader):.4f}")


def train_bct(new_model, old_model, X_new, Y_new, X_old, Y_old,
              epochs=EPOCHS, lr=1e-3, batch_size=64, lam=2.0):
    """
    BCT loss (Eq. 8):
      L_BCT = CE(new_cls(z_new(x)), y)                    <- main loss on T_new
            + lambda * CE(old_cls_frozen(z_new(x)), y)    <- influence loss on T_old

    The old classifier is frozen. Gradients from the influence loss
    flow only through new_model's embedding trunk.
    """
    loader_new = DataLoader(TensorDataset(X_new, Y_new), batch_size=batch_size, shuffle=True)
    loader_old = DataLoader(TensorDataset(X_old, Y_old), batch_size=batch_size, shuffle=True,
                            drop_last=True)

    opt       = Adam(new_model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(opt, T_max=epochs)
    ce        = nn.CrossEntropyLoss()

    # Freeze old model entirely
    old_model.eval()
    for p in old_model.parameters():
        p.requires_grad_(False)

    new_model.train()
    old_iter = iter(loader_old)

    for ep in range(1, epochs + 1):
        total = 0
        for xb_new, yb_new in loader_new:
            xb_new, yb_new = xb_new.to(device), yb_new.to(device)

            # -- Main loss (new classifier) --
            z_new, logits_new = new_model(xb_new)
            loss_main = ce(logits_new, yb_new)

            # -- Influence loss (old frozen classifier) --
            try:
                xb_old, yb_old = next(old_iter)
            except StopIteration:
                old_iter = iter(loader_old)
                xb_old, yb_old = next(old_iter)
            xb_old, yb_old = xb_old.to(device), yb_old.to(device)

            z_old, _ = new_model(xb_old)             # embed with NEW model
            logits_influence = old_model.cls(z_old)  # classify with OLD frozen head
            loss_influence = ce(logits_influence, yb_old)

            loss = loss_main + lam * loss_influence
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()

        scheduler.step()
        if ep % 20 == 0:
            print(f"  ep {ep:3d}/{epochs}  loss={total/len(loader_new):.4f}")


# -----------------------
# Evaluation  M(phi_query, phi_gallery; Q, D)
# -----------------------

@torch.no_grad()
def prototype_accuracy(phi_query, phi_gallery, X_q, Y_q, X_g, Y_g):
    phi_query.eval(); phi_gallery.eval()

    # Build gallery prototypes
    X_g, Y_g = X_g.to(device), Y_g.to(device)
    z_g, _   = phi_gallery(X_g)
    classes  = Y_g.unique().sort().values
    protos   = torch.stack([z_g[Y_g == c].mean(0) for c in classes])  # (C, D)

    # Embed queries and find nearest prototype
    z_q, _ = phi_query(X_q.to(device))
    dists  = torch.cdist(z_q, protos)            # (N_q, C)
    preds  = classes[dists.argmin(1)]
    return (preds == Y_q.to(device)).float().mean().item()



@torch.no_grad()
def top1_accuracy(phi_query, phi_gallery, X_q, Y_q, X_g, Y_g):
    """
    Top-1 accuracy using nearest-neighbour retrieval in embedding space.

    For each query, we find the single closest gallery sample (by L2 distance)
    and check whether its label matches the query label. This is stricter than
    prototype accuracy because it uses individual gallery embeddings rather than
    class-averaged prototypes, so it is sensitive to intra-class variance and
    outliers in the gallery.

    Boundary case: when phi_query == phi_gallery and the gallery contains the
    query samples, the nearest neighbour of a query would trivially be itself
    (distance 0). To avoid this self-match inflating the score, we mask out
    zero-distance matches, which handles the self-match case without requiring
    you to explicitly pass disjoint sets.
    """
    phi_query.eval(); phi_gallery.eval()

    z_q, _ = phi_query(X_q.to(device))           # (N_q, D)
    z_g, _ = phi_gallery(X_g.to(device))         # (N_g, D)
    Y_q    = Y_q.to(device)
    Y_g    = Y_g.to(device)

    dists = torch.cdist(z_q, z_g)                # (N_q, N_g)

    # Mask self-matches (distance == 0) by setting them to infinity
    self_mask        = dists == 0
    dists[self_mask] = float("inf")

    nn_idx  = dists.argmin(dim=1)                # (N_q,)
    nn_labels = Y_g[nn_idx]
    return (nn_labels == Y_q).float().mean().item()


@torch.no_grad()
def mean_average_precision(phi_query, phi_gallery, X_q, Y_q, X_g, Y_g):
    """
    Mean Average Precision (mAP) over all queries using embedding-space retrieval.

    For each query we rank all gallery samples by ascending L2 distance, then
    compute the Average Precision (AP) as the area under the precision-recall
    curve across the ranked list. mAP is the mean of AP over all queries.

    AP for a single query with R relevant items in the gallery:
        AP = (1/R) * sum_{k=1}^{N_g} P@k * rel(k)
    where P@k is precision at rank k and rel(k) = 1 if the k-th retrieved
    item shares the query label, 0 otherwise.

    Unlike top-1 or prototype accuracy, mAP rewards a model for pushing ALL
    relevant gallery items to the top of the ranked list, giving a richer
    signal about the quality of the learned embedding space.

    Self-matches (distance == 0) are masked out for the same reason as above.
    """
    phi_query.eval(); phi_gallery.eval()

    z_q, _ = phi_query(X_q.to(device))           # (N_q, D)
    z_g, _ = phi_gallery(X_g.to(device))         # (N_g, D)
    Y_q    = Y_q.to(device)
    Y_g    = Y_g.to(device)

    dists = torch.cdist(z_q, z_g)                # (N_q, N_g)

    # Mask self-matches
    self_mask        = dists == 0
    dists[self_mask] = float("inf")

    sorted_idx = dists.argsort(dim=1)            # (N_q, N_g), ascending distance

    ap_scores = []
    for i in range(len(Y_q)):
        ranked_labels = Y_g[sorted_idx[i]]       # gallery labels in retrieval order
        relevant      = (ranked_labels == Y_q[i]).float()
        R             = relevant.sum().item()
        if R == 0:                               # no relevant items → skip
            continue
        # Cumulative count of relevant items at each rank (1-indexed)
        cumulative   = relevant.cumsum(dim=0)
        ranks        = torch.arange(1, len(Y_g) + 1, device=device).float()
        precision_at = cumulative / ranks        # P@k for every k
        ap           = (precision_at * relevant).sum().item() / R
        ap_scores.append(ap)

    return float(np.mean(ap_scores))


# -----------------------
# Step 1: OLD model (5 classes)
# -----------------------
old = OldNet().to(device)

if USE_PRETRAINED:
    print("\n=== Loading OLD model (5 classes) ===")
    load_model(old, "old")
else:
    print("\n=== Training OLD model (5 classes) ===")
    train_standard(old, X_train_old, Y_train_old)
    save_model(old, "old")

# -----------------------
# Step 2: NEW* paragon (no BCT, full data)
# -----------------------
new_star = NewNet().to(device)

if USE_PRETRAINED:
    print("\n=== Loading NEW* paragon (10 classes, no BCT) ===")
    load_model(new_star, "new_star")
else:
    print("\n=== Training NEW* paragon (10 classes, no BCT) ===")
    train_standard(new_star, X_train_new, Y_train_new)
    save_model(new_star, "new_star")

# -----------------------
# Step 3: NEW-BCT (BCT aligned with old)
# -----------------------
new_bct = NewNet().to(device)

if USE_PRETRAINED:
    print("\n=== Loading NEW-BCT (10 classes, BCT) ===")
    load_model(new_bct, "new_bct")
else:
    print("\n=== Training NEW-BCT (10 classes, BCT) ===")
    train_bct(new_bct, old, X_train_new, Y_train_new, X_train_old, Y_train_old)
    save_model(new_bct, "new_bct")

# -----------------------
# Equation 9: Update Gain
# -----------------------
print("\n=== Evaluation top1 ===")

M_old_old   = top1_accuracy(old,      old,      X_test_old, Y_test_old, X_train_old, Y_train_old)
M_bct_old   = top1_accuracy(new_bct,  old,      X_test_old, Y_test_old, X_train_old, Y_train_old)
M_star_star = top1_accuracy(new_star, new_star, X_test_old, Y_test_old, X_train_old, Y_train_old)

print(f"  M(old,   old)   = {M_old_old:.4f}   <- lower bound")
print(f"  M(bct,   old)   = {M_bct_old:.4f}   <- backfill-free (Eq. 2 test)")
print(f"  M(new*,  new*)  = {M_star_star:.4f}   <- upper bound (paragon)")

denom = M_star_star - M_old_old
print("\n--- Equation 9: Update Gain ---")
if denom <= 0:
    print("Denominator <= 0: paragon not better than old model.")
elif M_bct_old <= M_old_old:
    print(f"Backward compatibility NOT achieved: M(bct,old) did not improve over M(old,old).")
else:
    G = (M_bct_old - M_old_old) / denom
    print(f"G = ({M_bct_old:.4f} - {M_old_old:.4f}) / ({M_star_star:.4f} - {M_old_old:.4f})")
    print(f"G = {G:.4f}  ({G*100:.1f}%)")
    print(f"Backward compatibility achieved!")
    print(f"BCT captures {G*100:.1f}% of paragon gain without backfilling the gallery.")


print("\n=== Evaluation prototype accuracy ===")

M_old_old   = prototype_accuracy(old,      old,      X_test_old, Y_test_old, X_train_old, Y_train_old)
M_bct_old   = prototype_accuracy(new_bct,  old,      X_test_old, Y_test_old, X_train_old, Y_train_old)
M_star_star = prototype_accuracy(new_star, new_star, X_test_old, Y_test_old, X_train_old, Y_train_old)

print(f"  M(old,   old)   = {M_old_old:.4f}   <- lower bound")
print(f"  M(bct,   old)   = {M_bct_old:.4f}   <- backfill-free (Eq. 2 test)")
print(f"  M(new*,  new*)  = {M_star_star:.4f}   <- upper bound (paragon)")

denom = M_star_star - M_old_old
print("\n--- Equation 9: Update Gain ---")
if denom <= 0:
    print("Denominator <= 0: paragon not better than old model.")
elif M_bct_old <= M_old_old:
    print(f"Backward compatibility NOT achieved: M(bct,old) did not improve over M(old,old).")
else:
    G = (M_bct_old - M_old_old) / denom
    print(f"G = ({M_bct_old:.4f} - {M_old_old:.4f}) / ({M_star_star:.4f} - {M_old_old:.4f})")
    print(f"G = {G:.4f}  ({G*100:.1f}%)")
    print(f"Backward compatibility achieved!")
    print(f"BCT captures {G*100:.1f}% of paragon gain without backfilling the gallery.")

# -----------------------
# Equation 9: Update Gain
# -----------------------
print("\n=== Evaluation mAp===")

M_old_old   = mean_average_precision(old,      old,      X_test_old, Y_test_old, X_train_old, Y_train_old)
M_bct_old   = mean_average_precision(new_bct,  old,      X_test_old, Y_test_old, X_train_old, Y_train_old)
M_star_star = mean_average_precision(new_star, new_star, X_test_old, Y_test_old, X_train_old, Y_train_old)

print(f"  M(old,   old)   = {M_old_old:.4f}   <- lower bound")
print(f"  M(bct,   old)   = {M_bct_old:.4f}   <- backfill-free (Eq. 2 test)")
print(f"  M(new*,  new*)  = {M_star_star:.4f}   <- upper bound (paragon)")

denom = M_star_star - M_old_old
print("\n--- Equation 9: Update Gain ---")
if denom <= 0:
    print("Denominator <= 0: paragon not better than old model.")
elif M_bct_old <= M_old_old:
    print(f"Backward compatibility NOT achieved: M(bct,old) did not improve over M(old,old).")
else:
    G = (M_bct_old - M_old_old) / denom
    print(f"G = ({M_bct_old:.4f} - {M_old_old:.4f}) / ({M_star_star:.4f} - {M_old_old:.4f})")
    print(f"G = {G:.4f}  ({G*100:.1f}%)")
    print(f"Backward compatibility achieved!")
    print(f"BCT captures {G*100:.1f}% of paragon gain without backfilling the gallery.")
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np

# -----------------------
# Dataset
# -----------------------
OLD_CLASSES = 40   # old model sees classes 0-39
NUM_CLASSES = 100  # new model sees all 100

# CIFAR-100 standard normalization
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

train_full = torchvision.datasets.CIFAR100(root="./data", train=True,  download=True, transform=transform_train)
test_full  = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)

def class_subset(dataset, max_classes):
    """Return a Subset containing only samples from classes 0..max_classes-1."""
    indices = [i for i, (_, y) in enumerate(dataset) if y < max_classes]
    return Subset(dataset, indices)

train_old = class_subset(train_full, OLD_CLASSES)
test_old  = class_subset(test_full,  OLD_CLASSES)

print(f"Old train: {len(train_old)} samples ({OLD_CLASSES} classes)")
print(f"New train: {len(train_full)} samples ({NUM_CLASSES} classes)")
print(f"Old test:  {len(test_old)} samples")
print(f"New test:  {len(test_full)} samples")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")

# -----------------------
# Models
#
# Both use the same ResNet-style backbone so embedding dim matches —
# this is required for the influence loss to feed new embeddings
# into the old classifier head.
#
# OldNet:  smaller (fewer filters), trained on 40 classes
# NewNet:  larger  (more filters),  trained on 100 classes
# -----------------------

class ResBlock(nn.Module):
    """Basic residual block with optional downsampling."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if (stride != 1 or in_ch != out_ch) else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


EMBED_DIM = 256   # shared embedding dimension — must match for influence loss


class OldNet(nn.Module):
    """Smaller CNN, trained on 40 classes only."""
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            # stem
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            # stage 1: 32x32 -> 16x16
            ResBlock(32,  64,  stride=2),
            # stage 2: 16x16 -> 8x8
            ResBlock(64,  128, stride=2),
            # stage 3: 8x8 -> 4x4
            ResBlock(128, EMBED_DIM, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.cls = nn.Linear(EMBED_DIM, OLD_CLASSES)

    def forward(self, x):
        z = self.backbone(x).flatten(1)
        return z, self.cls(z)


class NewNet(nn.Module):
    """Larger CNN, trained on all 100 classes."""
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            # stem
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            # stage 1: 32x32 -> 16x16
            ResBlock(64,  128, stride=2),
            ResBlock(128, 128),
            # stage 2: 16x16 -> 8x8
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256),
            # stage 3: 8x8 -> 4x4
            ResBlock(256, EMBED_DIM, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.cls = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def forward(self, x):
        z = self.backbone(x).flatten(1)
        return z, self.cls(z)


# -----------------------
# Training
# -----------------------

EPOCHS     = 100
BATCH_SIZE = 128
LR         = 1e-3

def train_standard(model, dataset, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE):
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           num_workers=2, pin_memory=True)
    opt       = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
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


def train_bct(new_model, old_model, train_new, train_old,
              epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, lam=2.0):
    """
    BCT loss (Eq. 8):
      L_BCT = CE(new_cls(z_new(x)), y)                  <- main loss on T_new
            + lambda * CE(old_cls_frozen(z_new(x)), y)  <- influence loss on T_old

    The old classifier head is frozen throughout.
    Gradients from the influence loss flow only through new_model's backbone.
    """
    loader_new = DataLoader(train_new, batch_size=batch_size, shuffle=True,
                            num_workers=2, pin_memory=True)
    loader_old = DataLoader(train_old, batch_size=batch_size, shuffle=True,
                            num_workers=2, pin_memory=True, drop_last=True)

    opt       = Adam(new_model.parameters(), lr=lr, weight_decay=1e-4)
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

            z_old, _ = new_model(xb_old)             # embed with NEW backbone
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
#
# Same nearest-prototype protocol as before:
#   - Build per-class mean embeddings from the gallery using phi_gallery
#   - Embed test queries with phi_query
#   - Assign each query to the nearest prototype
# -----------------------

@torch.no_grad()
def prototype_accuracy(phi_query, phi_gallery, test_dataset, gallery_dataset):
    phi_query.eval(); phi_gallery.eval()

    # Build gallery prototypes
    gallery_loader = DataLoader(gallery_dataset, batch_size=256, shuffle=False,
                                num_workers=2, pin_memory=True)
    all_z, all_y = [], []
    for xb, yb in gallery_loader:
        z, _ = phi_gallery(xb.to(device))
        all_z.append(z.cpu()); all_y.append(yb)
    all_z = torch.cat(all_z); all_y = torch.cat(all_y)

    classes = all_y.unique().sort().values
    protos  = torch.stack([all_z[all_y == c].mean(0) for c in classes]).to(device)

    # Embed test queries and find nearest prototype
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                             num_workers=2, pin_memory=True)
    correct, total = 0, 0
    for xb, yb in test_loader:
        z, _ = phi_query(xb.to(device))
        dists = torch.cdist(z, protos)
        preds = classes.to(device)[dists.argmin(1)]
        correct += (preds == yb.to(device)).sum().item()
        total   += len(yb)
    return correct / total


# -----------------------
# Step 1: Train OLD model (40 classes)
# -----------------------
print("\n=== Training OLD model (40 classes) ===")
old = OldNet().to(device)
train_standard(old, train_old)

# -----------------------
# Step 2: Train NEW* paragon (no BCT, all 100 classes)
# -----------------------
print("\n=== Training NEW* paragon (100 classes, no BCT) ===")
new_star = NewNet().to(device)
train_standard(new_star, train_full)

# -----------------------
# Step 3: Train NEW-BCT (BCT aligned with old)
# -----------------------
print("\n=== Training NEW-BCT (100 classes, BCT) ===")
new_bct = NewNet().to(device)
train_bct(new_bct, old, train_full, train_old)

# -----------------------
# Equation 9: Update Gain
#
# G(phi_new, phi_old; Q, D) =
#   [M(phi_new, phi_old) - M(phi_old, phi_old)]
#   / [M(phi*_new, phi*_new) - M(phi_old, phi_old)]
#
# Query set Q  = 40-class test set
# Gallery D    = 40-class train set  (the "indexed" gallery)
# -----------------------
print("\n=== Evaluation (40-class test set) ===")

M_old_old   = prototype_accuracy(old,      old,      test_old,  train_old)
M_bct_old   = prototype_accuracy(new_bct,  old,      test_old,  train_old)
M_star_star = prototype_accuracy(new_star, new_star, test_old,  train_old)

print(f"  M(old,   old)   = {M_old_old:.4f}   <- lower bound")
print(f"  M(bct,   old)   = {M_bct_old:.4f}   <- backfill-free (Eq. 2 test)")
print(f"  M(new*,  new*)  = {M_star_star:.4f}   <- upper bound (paragon)")

denom = M_star_star - M_old_old
print("\n--- Equation 9: Update Gain ---")
if denom <= 0:
    print("  Denominator <= 0: paragon not better than old model.")
elif M_bct_old <= M_old_old:
    print("  Backward compatibility NOT achieved: M(bct,old) did not improve over M(old,old).")
else:
    G = (M_bct_old - M_old_old) / denom
    print(f"  G = ({M_bct_old:.4f} - {M_old_old:.4f}) / ({M_star_star:.4f} - {M_old_old:.4f})")
    print(f"  G = {G:.4f}  ({G*100:.1f}%)")
    print(f"  Backward compatibility achieved!")
    print(f"  BCT captures {G*100:.1f}% of paragon gain without backfilling the gallery.")
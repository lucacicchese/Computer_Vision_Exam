import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# DATASET
# -----------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_set  = datasets.MNIST("./data", train=False, download=True, transform=transform)

X_train = train_set.data.view(-1,784).float()/255.
y_train = train_set.targets

X_test  = test_set.data.view(-1,784).float()/255.
y_test  = test_set.targets

# old dataset = 5 classi
mask_old = y_train < 5
X_old = X_train[mask_old]
y_old = y_train[mask_old]

old_loader = DataLoader(TensorDataset(X_old,y_old), batch_size=256, shuffle=True)
new_loader = DataLoader(TensorDataset(X_train,y_train), batch_size=256, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test,y_test), batch_size=1024)



# -----------------------
# MODELS
# -----------------------
class OldNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(784,16),
            nn.ReLU(),
            nn.Linear(16,64)
        )
        self.cls = nn.Linear(64,5)
    def forward(self,x):
        z = self.embed(x)
        return z, self.cls(z)

class NewNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(784,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64)
        )
        self.cls = nn.Linear(64,10)
    def forward(self,x):
        z = self.embed(x)
        return z, self.cls(z)

# -----------------------
# UTILS
# -----------------------
def train_epoch(model, loader, opt, loss_fn):
    model.train()
    for x,y in loader:
        x,y = x.to(device),y.to(device)
        opt.zero_grad()
        _,logits = model(x)
        loss = loss_fn(logits,y)
        loss.backward()
        opt.step()

def accuracy(model, loader):
    model.eval()
    c,t=0,0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device),y.to(device)
            _,logits = model(x)
            p = logits.argmax(1)
            c += (p==y).sum().item()
            t += y.size(0)
    return c/t

def accuracy_5(model, loader):
    model.eval()
    c,t=0,0
    with torch.no_grad():
        for x,y in loader:
            mask = y < 5
            if mask.sum()==0:
                continue
            x = x[mask].to(device)
            y = y[mask].to(device)
            _,logits = model(x)
            p = logits.argmax(1)
            c += (p==y).sum().item()
            t += y.size(0)
    return c/t


# -----------------------
# TRAIN OLD MODEL
# -----------------------
old = OldNet().to(device)
opt = torch.optim.Adam(old.parameters(),1e-3)

for _ in range(10):
    train_epoch(old, old_loader, opt, nn.CrossEntropyLoss())

acc_old_old = accuracy_5(old,test_loader)


old_w = old.cls.weight.detach().clone()
old_b = old.cls.bias.detach().clone()

# -----------------------
# TRAIN PARAGON MODEL
# -----------------------
paragon = NewNet().to(device)
opt = torch.optim.Adam(paragon.parameters(),1e-3)

for _ in range(50):
    train_epoch(paragon,new_loader,opt,nn.CrossEntropyLoss())

acc_star    = accuracy_5(paragon,test_loader)
# -----------------------
# TRAIN BCT MODEL
# -----------------------
bct = NewNet().to(device)
opt = torch.optim.Adam(bct.parameters(),1e-3)
lambda_par = 1.0

for _ in range(50):
    bct.train()
    for x,y in new_loader:
        x,y = x.to(device),y.to(device)

        z,logits = bct(x)
        loss_ce = F.cross_entropy(logits,y)

        old_logits = F.linear(z, old_w, old_b)

        mask = y<5
        if mask.sum()>0:
            loss_inf = F.cross_entropy(old_logits[mask], y[mask])
        else:
            loss_inf = 0.0

        loss = loss_ce + lambda_par*loss_inf

        opt.zero_grad()
        loss.backward()
        opt.step()

# -----------------------
# COMPATIBILITY ACC
# -----------------------
def compat_acc(new_model, loader):
    new_model.eval()
    c,t=0,0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device),y.to(device)
            z,_ = new_model(x)
            logits = F.linear(z, old_w, old_b)
            p = logits.argmax(1)
            mask = y<5
            c += (p[mask]==y[mask]).sum().item()
            t += mask.sum().item()
    return c/t

acc_new_old = compat_acc(bct,test_loader)

# -----------------------
# GAIN Eq(9)
# -----------------------
gain = (acc_new_old - acc_old_old) / (acc_star - acc_old_old)

print()
print("M(old,old) =",acc_old_old)
print("M(new,old) =",acc_new_old)
print("M(new*,new*) =",acc_star)
print("Gain =",gain)
print()

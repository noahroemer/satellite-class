from torchvision import transforms
from torchvision import datasets
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models
import torch
import torch.optim as optim
from torchmetrics.classification import MulticlassAccuracy
import matplotlib.pyplot as plt
from sat_main import train_step, test_step, collect_logits_targets
import math
from copy import deepcopy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Normalize
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Transform
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

data_dir = r"C:\Users\noahl\Downloads\sat-class\EuroSAT_RGB"

# Loading Data
dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

# splits (same ratios)
train_size = int(0.7 * len(dataset))
val_size   = int(0.15 * len(dataset))
test_size  = len(dataset) - train_size - val_size

train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
# Set correct transforms
val_ds.dataset.transform = eval_transform
test_ds.dataset.transform = eval_transform

batch_size = 32
resnet_train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
resnet_val_dataloader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
resnet_test_dataloader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

class_names = train_ds.dataset.classes
num_classes = len(class_names)  # EuroSAT = 10

def build_resnet(num_classes: int,
                 model_name: str = "resnet18",
                 pretrained: bool = True,
                 freeze_backbone: bool = True):
    # Create backbone
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    else:
        raise ValueError("Choose resnet18/resnet34/resnet50")

    # optionally freeze
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes) # logits

    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate model
model_2 = build_resnet(num_classes=num_classes,
                       model_name="resnet34",
                       pretrained=True,
                       freeze_backbone=True).to(device)

model_2.to(device)

loss_fn = nn.CrossEntropyLoss()
acc_fn = MulticlassAccuracy(num_classes=num_classes).to(device)

optimizer = optim.Adam(model_2.fc.parameters(), lr=3e-3, weight_decay=1e-4)

epochs = 50  # start small, then scale
print("torch.cuda.is_available():", torch.cuda.is_available())
print("Device variable:", device)
print("Model device:", next(model_2.parameters()).device)

X_test, y_test = next(iter(resnet_train_dataloader))
print("Batch device before:", X_test.device)
X_test, y_test = X_test.to(device), y_test.to(device)
print("Batch device after:", X_test.device)


# --- knobs you can tweak ---
EPOCHS_FROZEN    = 5          # phase 1
EPOCHS_FINETUNE  = 30       # phase 2
PATIENCE         = 5          # early-stopping patience on val acc
HEAD_LR          = 3e-3       # LR when only training the head (frozen backbone)
FINETUNE_LR      = 1e-4       # LR when unfreezing the whole model
WEIGHT_DECAY     = 1e-4

best_val_acc = 0.0
epochs_no_improve = 0
best_state = None

# ------------- PHASE 1: freeze backbone, train the head -------------
# (If you built the model with freeze_backbone=False, freeze it now)
for p in model_2.parameters():
    p.requires_grad = False
for p in model_2.fc.parameters():
    p.requires_grad = True

optimizer = optim.AdamW(model_2.fc.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)

print("\n=== Phase 1: training head (backbone frozen) ===")
for epoch in range(1, EPOCHS_FROZEN + 1):
    train_loss, train_acc = train_step(model=model_2,
                                       data_loader=resnet_train_dataloader,
                                       optimizer=optimizer,
                                       loss_fn=loss_fn,
                                       accuracy_fn=acc_fn,
                                       device=device)
    val_loss, val_acc = test_step(model=model_2,
                                  data_loader=resnet_val_dataloader,
                                  loss_fn=loss_fn,
                                  accuracy_fn=acc_fn,
                                  device=device)

    improved = val_acc > best_val_acc
    if improved:
        best_val_acc = val_acc
        best_state = deepcopy(model_2.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    print(f"[Frozen] Epoch {epoch:02d} | "
          f"train {train_loss:.4f}/{train_acc:.3f} | "
          f"val {val_loss:.4f}/{val_acc:.3f} | "
          f"best_val_acc {best_val_acc:.3f} (no_improve={epochs_no_improve})")

    if epochs_no_improve >= PATIENCE:
        print("Early stop (frozen phase) — no val improvement.")
        break

# Load best-before-finetune (optional but helpful)
if best_state is not None:
    model_2.load_state_dict(best_state)

# ------------- PHASE 2: unfreeze all, fine-tune at small LR ----------
for p in model_2.parameters():
    p.requires_grad = True   # unfreeze everything

# You can use AdamW or SGD; SGD uses slightly less memory
optimizer = optim.AdamW(model_2.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)

# Optional: reduce LR if val loss plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

print("\n=== Phase 2: fine-tuning (backbone unfrozen) ===")
epochs_no_improve = 0  # reset patience for fine-tune

train_losses, val_losses = [], []
train_accs,   val_accs   = [], []
if __name__ == "__main__":
    for epoch in range(1, EPOCHS_FINETUNE + 1):
        train_loss, train_acc = train_step(model=model_2,
                                        data_loader=resnet_train_dataloader,
                                        optimizer=optimizer,
                                        loss_fn=loss_fn,
                                        accuracy_fn=acc_fn,
                                        device=device)
        val_loss, val_acc = test_step(model=model_2,
                                    data_loader=resnet_val_dataloader,
                                    loss_fn=loss_fn,
                                    accuracy_fn=acc_fn,
                                    device=device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step(val_loss)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_state = deepcopy(model_2.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[Finetune] Epoch {epoch:02d} | "
            f"LR {current_lr:.2e} | "
            f"train {train_loss:.4f}/{train_acc:.3f} | "
            f"val {val_loss:.4f}/{val_acc:.3f} | "
            f"best_val_acc {best_val_acc:.3f} (no_improve={epochs_no_improve})")

        if epochs_no_improve >= PATIENCE:
            print("Early stop (fine-tune phase) — no val improvement.")
            break
    # restore best and save
    if best_state is not None:
        model_2.load_state_dict(best_state)
    torch.save(model_2.state_dict(), "resnet_eurosat_best.pt")
    print(f"Saved best model with val_acc={best_val_acc:.3f} → resnet_eurosat_best.pt")

    # Plots
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, train_losses, label="Train")
    plt.plot(epochs_range, val_losses,   label="Val")
    plt.title("Loss per Epoch - Resnet 34"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.savefig("loss_per_epoch_resnet.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,2)
    plt.plot(epochs_range, train_accs, label="Train")
    plt.plot(epochs_range, val_accs,   label="Val")
    plt.title("Accuracy per Epoch - ResNet34"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.savefig("acc_per_epoch_resnet.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Confusion Matrix
    # After training
    resnet_logits, resnet_targets = collect_logits_targets(model_2, resnet_test_dataloader, device)
    resnet_preds = resnet_logits.argmax(1)

    # Raw counts matrix
    cm = confusion_matrix(resnet_targets.numpy(), resnet_preds.numpy(), labels = np.arange(num_classes))

    # Normalized (per true class) – easier to read
    cm_norm = confusion_matrix(resnet_targets.numpy(), resnet_preds.numpy(),
                            labels=np.arange(num_classes), normalize="true")

    # Pretty plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Purples", values_format=".2f", colorbar=True)
    plt.title("EuroSAT - Confustion Matrix Normalized (Selfmade CNN)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
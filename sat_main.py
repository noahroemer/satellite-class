import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import MulticlassAccuracy
from tqdm.auto import tqdm
import os, random, numpy as np, torch
from models import SimpleCNN
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Set seed for reproducability 
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # safe even if no CUDA# Import the data 

# Path to image dataset
data_dir = r"C:\Users\noahl\Downloads\sat-class\EuroSAT_RGB"

# Define transforms: resize, tensor, normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5))
])

# Load dataset with ImageFolder
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Split into train, val, and test sets
train_size = int(.7 * len(dataset))
val_size = int(.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Split the dataset
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders
batch_size = 32 # Define batch size
train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

print(dataset.classes)  # Print class names
print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}")

print(train_ds[0])  # Print first sample from training set
print(train_ds[0][0].shape)  # Print shape of training set
print(train_ds[0][0].dtype)  # Data type of training set

class_names = train_ds.dataset.classes

'''
    We know have our image data loaded and stored as tensors.
    It is know ready to be trained.
'''



# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate model
input_size = 3*64*64
num_classes = 10

model_1 = SimpleCNN(input_shape=3,
                    hidden_units=32,
                    output_shape=num_classes)

# Set loss function, accuracy funtion, and optimizer
loss_fn = nn.CrossEntropyLoss()
acc_fn = MulticlassAccuracy(num_classes=10).to(device)
optimizer = torch.optim.Adam(params=model_1.parameters(), lr=1e-3, weight_decay=5e-4)


# Going to create training and test functions
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               optimizer: optim.Optimizer,
               loss_fn,
               accuracy_fn,
               device: torch.device = device):
    """
    Performs one full training epoch over the given data

    Parameters
    ----------------
    model: torch.nn.Module
        The neural network model to train,
    data_loader: torch.utils.data.DataLoader
        Dataloader providing input and target batches for training
      optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters (e.g., SGD, Adam).
    loss_fn : callable
        Loss function to minimize. Should take (y_pred, y_true) as input.
    accuracy_fn : callable
        Function to compute batch accuracy. Should take (y_true, y_pred) as input.
    device : torch.device, optional
        Device to run training on (CPU or GPU). Default is the global `device`.

    Returns
    -------
    tuple of (float, float)
        Average training loss and accuracy across all batches in this epoch.
    """
    # Set loss and accuracy values
    train_loss, train_acc = 0, 0

    # Ensure model on right device
    model.to(device)

    # Set model to train mode
    model.train()

    # Reset accuracy value
    accuracy_fn.reset()

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        # Forward pass
        logits = model(X)

        # Calculate loss and accuray
        loss = loss_fn(logits, y)
        train_loss += loss.item()
        train_acc = accuracy_fn(target=y,
                                preds=logits.argmax(1))

        # Apply changes
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # What's happening
        if batch % 1600 == 0:
            print(f"Looked at {batch * len(X)}/{len(data_loader.dataset)} samples.")

    train_loss /= len(data_loader)
    train_acc = accuracy_fn.compute().item()
    print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.2f}")
    return train_loss, train_acc



def test_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn,
               accuracy_fn,
               device: torch.device = device):
    """
    Evaluate a trained model on a given dataset.

    Runs the model in evaluation mode over all batches in `data_loader`,
    computes the average loss and accuracy, and prints the results.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to evaluate.
    data_loader : torch.utils.data.DataLoader
        Dataloader providing input and target batches for evaluation.
    loss_fn : callable
        Loss function used to calculate error between predictions and targets.
    accuracy_fn : torchmetrics.Metric or callable
        Accuracy metric that supports `.reset()` and `.compute()`. Should take
        predictions and targets as input.
    device : torch.device, optional
        Device to run evaluation on (CPU or GPU). Default is the global `device`.

    Returns
    -------
    tuple of (float, float)
        Average test loss and accuracy across all batches.
    """
    accuracy_fn.reset()

    test_loss, test_acc = 0, 0

    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            # Forward pass
            logits = model(X)

            # Loss and acc
            test_loss += loss_fn(logits, y).item()
            test_acc += accuracy_fn(target=y,
                               preds=logits.argmax(dim=1))

        test_loss /= len(data_loader)
        test_acc = accuracy_fn.compute().item()
        #print(f"Test Loss: {test_loss:.4f} | Test acc: {test_acc:.2f}")
        return test_loss, test_acc

# Set epochs
epochs = 50

# Helper function for evaluating our models
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn):
  """Returns a dictionary containing the results of a model predicting on data_loader"""
  loss, acc = 0, 0
  model.eval()
  with torch.inference_mode():
    accuracy_fn.reset()
    for X, y in data_loader:
      # Make preds
      X, y = X.to(device), y.to(device)
      y_pred = model(X)
      loss += loss_fn(y_pred, y)
      acc += accuracy_fn(target = y,
                         preds=y_pred.argmax(dim=1))
    # Scale loss and acc to find average loss/acc per batch

    loss /= len(data_loader)
    test_acc = accuracy_fn.compute().item()


  return {"model_name": model.__class__.__name__,
          "model_loss": loss.item(),
          "model_acc": acc}

def collect_logits_targets(model, data_loader, device):
  model.eval()
  all_logits = []
  all_targets = []
  with torch.no_grad():
    for X, y in data_loader:
      X, y = X.to(device), y.to(device)
      logits = model(X)
      all_logits.append(logits.cpu())
      all_targets.append(y.cpu())
  logits = torch.cat(all_logits)
  targets = torch.cat(all_targets)
  return logits, targets

patience = 7                     # stop if no val_loss improvement for 7 epochs
best_val = float("inf")
wait = 0
best_path = "best_model.pt"

train_losses_simple, train_accs_simple = [], []
val_losses_simple, val_accs_simple = [], []

if __name__ == "__main__":
    for epoch in tqdm(range(epochs)):
        train_step(model=model_1,
                data_loader=train_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                accuracy_fn=acc_fn,
                device=device)

        # validate on the validation set (not test) for early stopping
        val_loss, val_acc = test_step(model=model_1,
                                    data_loader=val_dataloader,
                                    loss_fn=loss_fn,
                                    accuracy_fn=acc_fn,
                                    device=device)
        print(f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}")

        # ---- EARLY STOPPING ----
        if val_loss < best_val - 1e-6:   # small delta to avoid float noise
            best_val = val_loss
            wait = 0
            torch.save(model_1.state_dict(), best_path)
            # print("Saved new best model.")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best val_loss={best_val:.4f}")
                break

    epochs_range_simple = range(1, len(train_losses_simple) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range_simple, train_losses_simple, label = "Train Loss")
    plt.plot(epochs_range_simple, val_losses_simple, label = "Val Loss")
    plt.title("Loss per Epoch - Selfmade CNN"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.savefig("loss_per_epoch_cnn.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range_simple, train_accs_simple, label = "Train Accs")
    plt.plot(epochs_range_simple, val_accs_simple, label = "Val Accs")
    plt.title("Accuracy per Epoch - Selfmade CNN"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.savefig("acc_per_epoch_cnn.png", dpi=300, bbox_inches="tight")
    plt.show()

        # After training
    logits, targets = collect_logits_targets(model_1, val_dataloader, device)
    preds = logits.argmax(1)

    # Raw counts matrix
    cm = confusion_matrix(targets.numpy(), preds.numpy(), labels = np.arange(num_classes))

    # Normalized (per true class) – easier to read
    cm_norm = confusion_matrix(targets.numpy(), preds.numpy(),
                            labels=np.arange(num_classes), normalize="true")

    # Pretty plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=True)
    plt.title("EuroSAT - Confustion Matrix Normalized (Selfmade CNN)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
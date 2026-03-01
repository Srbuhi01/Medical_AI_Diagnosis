import os
import time
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.cuda.amp import GradScaler
from dataset import ChestXrayDataset


# --- Early Stopping Class ---
class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0,
                 path=r'C:\Users\srbuh\Desktop\Medical_AI_Diagnosis\models\resnet50_best.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# --- Main Training Code ---

BATCH_SIZE = 8
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0

POS_WEIGHTS = [8.7, 39.4, 23.0, 47.7, 7.4, 43.6, 65.5, 492.9, 4.6, 18.4, 16.7, 32.5, 96.5, 20.3]
MODELS_DIR = r'C:\Users\srbuh\Desktop\Medical_AI_Diagnosis\models'


def train_model():
    print(f" Training on device: {DEVICE}")

    # Scaler for AMP
    scaler = GradScaler()

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(" Loading datasets...")
    train_dataset = ChestXrayDataset(
        data_dir=r'C:\Users\srbuh\Desktop\Medical_AI_Diagnosis\data\images',
        csv_file=r'C:\Users\srbuh\Desktop\Medical_AI_Diagnosis\data\Data_Entry_2017.csv',
        split_list_file=r'C:\Users\srbuh\Desktop\Medical_AI_Diagnosis\data\train_val_list.txt',
        transform=train_transform
    )

    val_dataset = ChestXrayDataset(
        data_dir=r'C:\Users\srbuh\Desktop\Medical_AI_Diagnosis\data\images',
        csv_file=r'C:\Users\srbuh\Desktop\Medical_AI_Diagnosis\data\Data_Entry_2017.csv',
        split_list_file=r'C:\Users\srbuh\Desktop\Medical_AI_Diagnosis\data\test_list.txt',
        transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(" Loading ResNet50...")
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 14)
    model = model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- RESUME LOGIC ---
    start_epoch = 0
    for e in range(NUM_EPOCHS, 0, -1):
        path = os.path.join(MODELS_DIR, f"resnet50_epoch_{e}.pth")
        if os.path.exists(path):
            print(f" Found checkpoint: {path}")
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            start_epoch = e
            print(f" Resuming training from Epoch {start_epoch + 1}")
            break

    if start_epoch == 0:
        print(" No checkpoint found. Starting from scratch.")

    pos_weights_tensor = torch.tensor(POS_WEIGHTS).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights_tensor)

    early_stopping = EarlyStopping(patience=3, verbose=True, path=os.path.join(MODELS_DIR, 'resnet50_best.pth'))

    history_path = os.path.join(MODELS_DIR, "training_log.csv")
    if os.path.exists(history_path) and start_epoch > 0:
        print(" Loading logs...")
        df_history = pd.read_csv(history_path)
        history = df_history.to_dict(orient='list')
    else:
        history = {'epoch': [], 'train_loss': [], 'val_loss': []}

    print("\n Starting Training (Safe Mode + Fixed Validation)...")

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 20)

        # --- Training ---
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, total=len(train_loader), leave=True)

        for images, labels in loop:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            # Training with AMP
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            loop.set_description(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
            loop.set_postfix(loss=loss.item())

            del images, labels, outputs, loss

        epoch_loss = running_loss / len(train_loader)
        print(f" Train Loss: {epoch_loss:.4f}")

        gc.collect()
        torch.cuda.empty_cache()

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        print(" Validating...")
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                # Validation with AMP
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                del images, labels, outputs, loss

        avg_val_loss = val_loss / len(val_loader)
        print(f" Val Loss: {avg_val_loss:.4f}")

        # --- Save History ---
        if not (len(history['epoch']) > 0 and history['epoch'][-1] == epoch + 1):
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(epoch_loss)
            history['val_loss'].append(avg_val_loss)

        df_history = pd.DataFrame(history)
        os.makedirs(MODELS_DIR, exist_ok=True)
        df_history.to_csv(history_path, index=False)

        checkpoint_path = os.path.join(MODELS_DIR, f"resnet50_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f" Checkpoint saved to {checkpoint_path}")

        # Early Stopping
        early_stopping(avg_val_loss, model)

        print(" Cooling down GPU for 10 seconds...")
        time.sleep(10)

        if early_stopping.early_stop:
            print("🛑 Early stopping triggered!")
            break

        gc.collect()
        torch.cuda.empty_cache()

    print("\n🎉 Training Complete!")


if __name__ == "__main__":
    train_model()
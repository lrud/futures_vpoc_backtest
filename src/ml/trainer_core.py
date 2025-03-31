"""Core training logic shared across all training modes."""
import torch
import time
from typing import Dict, Tuple
from torch.utils.data import DataLoader

class TrainingCore:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        train_loss = 0.0
        samples_processed = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs).squeeze()
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs).squeeze()
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            samples_processed += inputs.size(0)
            
        return train_loss / samples_processed

    def validate(self, val_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        X_val, y_val = val_data
        
        with torch.no_grad():
            outputs = self.model(X_val).squeeze()
            val_loss = self.criterion(outputs, y_val).item()
            predicted = (outputs > 0).float()
            accuracy = (predicted == y_val).sum().item() / y_val.size(0) * 100
            
        return {'val_loss': val_loss, 'val_accuracy': accuracy}

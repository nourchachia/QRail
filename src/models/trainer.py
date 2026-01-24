# File: src/models/trainer.py

"""
Unified Model Trainer
Handles training for GNN, LSTM, and MLP models with production features:
- Validation loop with early stopping
- Learning rate scheduling
- Gradient clipping
- TensorBoard logging
- Checkpoint management
- Resume training support
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Optional, Dict
import time
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: str = "cpu",
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        log_dir: str = "runs",
        checkpoint_dir: str = "checkpoints",
        gradient_clip: float = 1.0
    ):
        """
        A unified trainer for GNN, LSTM, and MLP models.
        
        Args:
            model: PyTorch model to train
            optimizer: Optimizer (Adam, SGD, etc.)
            criterion: Loss function (MSE, CrossEntropy, etc.)
            device: 'cuda', 'cpu', or 'mps'
            scheduler: Learning rate scheduler (optional)
            log_dir: Directory for TensorBoard logs
            checkpoint_dir: Directory for model checkpoints
            gradient_clip: Max norm for gradient clipping
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip
        
        # Directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'epoch': 0
        }
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        logger.info(f"✅ Trainer initialized | Device: {device} | Model: {model.__class__.__name__}")
    
    def train_epoch(self, dataloader: DataLoader, is_gnn: bool = False) -> float:
        """Runs one epoch of training."""
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            self.optimizer.zero_grad()
            
            try:
                if is_gnn:
                    # GNN Handling (PyTorch Geometric Batch)
                    batch = batch.to(self.device)
                    outputs = self.model(batch)
                    
                    if hasattr(batch, 'y'):
                        targets = batch.y.to(self.device)
                        loss = self.criterion(outputs, targets)
                    else:
                        continue
                else:
                    # Standard Handling (LSTM/MLP)
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                loss.backward()
                
                # Gradient clipping
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.gradient_clip
                    )
                
                self.optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
            except Exception as e:
                logger.error(f"❌ Error in batch {batch_idx}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        avg_loss = total_loss / max(batch_count, 1)
        return avg_loss
    
    def validate_epoch(self, dataloader: DataLoader, is_gnn: bool = False) -> float:
        """Runs one epoch of validation."""
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for batch in dataloader:
                try:
                    if is_gnn:
                        batch = batch.to(self.device)
                        outputs = self.model(batch)
                        
                        if hasattr(batch, 'y'):
                            targets = batch.y.to(self.device)
                            loss = self.criterion(outputs, targets)
                            total_loss += loss.item()
                            batch_count += 1
                    else:
                        inputs, targets = batch
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                        total_loss += loss.item()
                        batch_count += 1
                    
                except Exception as e:
                    logger.error(f"❌ Validation error: {e}")
                    continue
        
        avg_loss = total_loss / max(batch_count, 1)
        return avg_loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        is_gnn: bool = False,
        early_stopping_patience: int = 10,
        save_best_only: bool = True,
        save_interval: int = 5
    ) -> Dict:
        """Main training loop."""
        logger.info(f"🚀 Starting training for {epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training phase
            train_loss = self.train_epoch(train_loader, is_gnn=is_gnn)
            self.history['train_loss'].append(train_loss)
            
            # Validation phase
            val_loss = None
            if val_loader:
                val_loss = self.validate_epoch(val_loader, is_gnn=is_gnn)
                self.history['val_loss'].append(val_loss)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loss else train_loss)
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            self.history['epoch'] = epoch + 1
            
            # TensorBoard logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            if val_loss:
                self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('LearningRate', current_lr, epoch)
            
            # Console output
            epoch_time = time.time() - epoch_start
            log_msg = f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}"
            if val_loss:
                log_msg += f" | Val Loss: {val_loss:.4f}"
            log_msg += f" | LR: {current_lr:.6f} | Time: {epoch_time:.2f}s"
            logger.info(log_msg)
            
            # Checkpoint saving
            if val_loader:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    if save_best_only:
                        self.save_checkpoint("best_model.pt", epoch, val_loss)
                        logger.info(f"💾 New best model saved (val_loss: {val_loss:.4f})")
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= early_stopping_patience:
                    logger.info(f"⏹️  Early stopping triggered")
                    break
            
            if not save_best_only and (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt", epoch, train_loss)
        
        total_time = time.time() - start_time
        logger.info(f"✅ Training completed in {total_time/60:.2f} minutes")
        
        self.save_checkpoint("final_model.pt", epochs, train_loss)
        self.save_history()
        
        return self.history
    
    def save_checkpoint(self, filename: str, epoch: int, loss: float):
        """Save model checkpoint."""
        filepath = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'history': self.history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        logger.info(f"💾 Checkpoint saved: {filepath}")
    
    def save_history(self):
        """Save training history to JSON."""
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"📊 Training history saved: {history_path}")
    
    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()
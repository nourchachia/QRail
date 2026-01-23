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
from typing import Optional, Dict, Callable
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
            gradient_clip: Max norm for gradient clipping (prevents exploding gradients)
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
        """
        Runs one epoch of training.
        
        Args:
            dataloader: PyTorch DataLoader or Geometric DataLoader
            is_gnn: If True, handles PyTorch Geometric batch objects
        
        Returns:
            Average loss for the epoch
        """
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
                    # Handle both single Data objects and Batch objects
                    if hasattr(batch, 'y'):
                        targets = batch.y
                    elif hasattr(batch, '__getitem__') and isinstance(batch, (list, tuple)):
                        # If batch is a list/tuple, extract targets
                        targets = batch[1] if len(batch) > 1 else None
                    else:
                        # For unsupervised/embedding tasks, use outputs as targets (self-supervised)
                        targets = outputs.detach()  # Use model output as target for embedding learning
                else:
                    # Standard Handling (LSTM/MLP)
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    outputs = self.model(inputs)
                
                # Compute loss
                if targets is not None:
                    loss = self.criterion(outputs, targets)
                else:
                    # If no targets, skip this batch (or use a different loss)
                    continue
                
                loss.backward()
                
                # Gradient clipping (prevents exploding gradients in RNNs)
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
        """
        Runs one epoch of validation (no gradient updates).
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for batch in dataloader:
                try:
                    if is_gnn:
                        batch = batch.to(self.device)
                        outputs = self.model(batch)
                        # Handle both single Data objects and Batch objects
                        if hasattr(batch, 'y'):
                            targets = batch.y
                        elif hasattr(batch, '__getitem__') and isinstance(batch, (list, tuple)):
                            targets = batch[1] if len(batch) > 1 else None
                        else:
                            targets = None
                    else:
                        inputs, targets = batch
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        outputs = self.model(inputs)
                    
                    if targets is not None:
                        loss = self.criterion(outputs, targets)
                        total_loss += loss.item()
                        batch_count += 1
                    
                except Exception as e:
                    logger.error(f"❌ Validation error: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
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
        """
        Main training loop with validation and early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs to train
            is_gnn: Whether model is a GNN (affects batch handling)
            early_stopping_patience: Stop if no improvement for N epochs
            save_best_only: Only save checkpoints when validation improves
            save_interval: Save checkpoint every N epochs (if not save_best_only)
        
        Returns:
            Training history dictionary
        """
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
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    if save_best_only:
                        self.save_checkpoint("best_model.pt", epoch, val_loss)
                        logger.info(f"💾 New best model saved (val_loss: {val_loss:.4f})")
                else:
                    self.patience_counter += 1
                
                # Early stopping check
                if self.patience_counter >= early_stopping_patience:
                    logger.info(f"⏹️  Early stopping triggered (no improvement for {early_stopping_patience} epochs)")
                    break
            
            # Periodic checkpoint
            if not save_best_only and (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt", epoch, train_loss)
        
        total_time = time.time() - start_time
        logger.info(f"✅ Training completed in {total_time/60:.2f} minutes")
        
        # Save final model
        self.save_checkpoint("final_model.pt", epochs, train_loss)
        self.save_history()
        
        return self.history
    
    def save_checkpoint(self, filename: str, epoch: int, loss: float):
        """Save model checkpoint with metadata."""
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
    
    def load_checkpoint(self, filepath: str) -> Dict:
        """Load checkpoint and resume training."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        
        logger.info(f"✅ Checkpoint loaded from {filepath}")
        return checkpoint
    
    def save_history(self):
        """Save training history to JSON."""
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"📊 Training history saved: {history_path}")
    
    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()


# Example Usage
if __name__ == "__main__":

    from cascade.lstm_encoder import LSTMEncoder

    print("=" * 70)
    print("UNIFIED TRAINER TEST - Neural Rail Conductor")
    print("=" * 70)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device}")
    
    # Create model
    model = LSTMEncoder(input_size=4, hidden_size=64, num_layers=2)
    
    # Create dummy data
    train_inputs = torch.randn(100, 10, 4)  # 100 samples
    train_targets = torch.randn(100, 64)
    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    val_inputs = torch.randn(20, 10, 4)
    val_targets = torch.randn(20, 64)
    val_dataset = torch.utils.data.TensorDataset(val_inputs, val_targets)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Initialize trainer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    criterion = nn.MSELoss()
    
    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        log_dir="runs/lstm_test",
        checkpoint_dir="checkpoints/lstm_test"
    )
    
    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,
        is_gnn=False,
        early_stopping_patience=5
    )
    
    print("\n📊 Training Summary:")
    print(f"   Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"   Final Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"   Best Val Loss: {trainer.best_val_loss:.4f}")
    
    trainer.close()
    
    print("\n" + "=" * 70)
    print("✅ Training test completed! Check 'runs/' for TensorBoard logs.")
    print("   Command: tensorboard --logdir=runs")
    print("=" * 70)
"""
LSTM Cascade Encoder Training Script
Trains the LSTM model on historical incident data to predict delay propagation patterns.
Generates synthetic telemetry sequences from incident metadata with realistic delay patterns.

Features:
- Loads 1000 incidents with rich metadata (type, severity, cascade depth, etc.)
- Generates synthetic telemetry based on incident archetype and characteristics
- Three delay patterns: Static Blockage (spike), Ripple Delay (gradual), Bottleneck Cascade (oscillating)
- Self-supervised learning (model learns sequence patterns)
- Early stopping and learning rate scheduling
- TensorBoard logging and checkpoint management
- Gradient clipping for stable RNN training

Data Structure:
- Input: [batch, 10 timesteps, 4 features]
  * Feature 0: Normalized delay (0-1)
  * Feature 1: Progress through route (0-1)
  * Feature 2: Normalized speed (0-1)
  * Feature 3: Hub proximity (binary)
- Output: [batch, 64] embedding vector
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import time
import json
import logging
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cascade.lstm_encoder import LSTMEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IncidentDataset(Dataset):
    """
    PyTorch Dataset for loading LSTM training data from incidents.json.
    Generates synthetic telemetry sequences based on incident metadata.
    Each sample is a telemetry sequence [10 timesteps, 4 features].
    """
    def __init__(self, incidents_path: str, split: str = "train", sequence_length: int = 10):
        """
        Args:
            incidents_path: Path to incidents.json
            split: "train" or "test" (uses existing split from data generator)
            sequence_length: Number of timesteps in sequence (default 10)
        """
        self.incidents_path = Path(incidents_path)
        self.split = split
        self.sequence_length = sequence_length
        
        # Load incidents
        with open(self.incidents_path, 'r') as f:
            data = json.load(f)
        
        self.incidents = data.get(split, [])
        logger.info(f"✅ Loaded {len(self.incidents)} {split} incidents from {incidents_path}")
    
    def __len__(self):
        return len(self.incidents)
    
    def _generate_telemetry_sequence(self, incident: Dict) -> torch.Tensor:
        """
        Generate synthetic telemetry sequence from incident metadata.
        Creates a realistic delay propagation pattern based on incident characteristics.
        
        Returns:
            [sequence_length, 4] tensor with features:
            - Feature 0: Normalized delay (0-1, scaled by MAX_DELAY=60min)
            - Feature 1: Progress through route (0-1)
            - Feature 2: Normalized speed (0-1, scaled by MAX_SPEED=160km/h)
            - Feature 3: Hub proximity (0=no, 1=yes)
        """
        MAX_DELAY = 60.0  # minutes
        MAX_SPEED = 160.0  # km/h
        
        # Extract incident characteristics
        cascade_depth = incident.get('cascade_depth', 1)
        severity = incident.get('severity_level', 3)
        estimated_delay = incident.get('estimated_delay_minutes', 30)
        is_peak = incident.get('is_peak', False)
        is_junction = incident.get('is_junction', False)
        location_type = incident.get('location_type', 'segment')
        archetype = incident.get('archetype', 'Unknown')
        
        # Determine delay pattern based on archetype
        if archetype == "Static Blockage":
            # Sudden spike, then plateau
            delay_pattern = "spike"
        elif archetype == "Ripple Delay":
            # Gradual increase (cascade effect)
            delay_pattern = "gradual"
        elif archetype == "Bottleneck Cascade":
            # Oscillating pattern
            delay_pattern = "oscillating"
        else:
            delay_pattern = "linear"
        
        sequence = []
        base_delay = min(estimated_delay / MAX_DELAY, 1.0)
        
        for t in range(self.sequence_length):
            progress = t / (self.sequence_length - 1)  # 0 to 1
            
            # Generate delay based on pattern
            if delay_pattern == "spike":
                # Sudden increase at t=3, then plateau
                if t < 3:
                    delay = base_delay * (t / 3) * 0.3
                else:
                    delay = base_delay * (0.8 + 0.2 * (t - 3) / (self.sequence_length - 3))
            
            elif delay_pattern == "gradual":
                # Exponential growth (cascade)
                delay = base_delay * (1 - np.exp(-3 * progress))
            
            elif delay_pattern == "oscillating":
                # Oscillating delays (bottleneck clearing/blocking)
                delay = base_delay * (0.5 + 0.5 * np.sin(progress * 2 * np.pi))
            
            else:  # linear
                delay = base_delay * progress
            
            # Add severity multiplier
            delay *= (severity / 5.0)
            
            # Add peak hour factor (congestion amplifies delays)
            if is_peak:
                delay *= 1.2
            
            # Clip to valid range
            delay = min(delay, 1.0)
            
            # Speed decreases as delay increases (realistic constraint)
            speed = max(0.3, 1.0 - delay * 0.7)  # Speed 30-100% of max
            
            # Hub proximity indicator
            is_hub = 1.0 if (is_junction or location_type in ['major_hub', 'regional']) else 0.0
            # At hub stations (beginning or end), set hub flag
            if t < 2 or t >= self.sequence_length - 2:
                is_hub = max(is_hub, 0.5)
            
            sequence.append([delay, progress, speed, is_hub])
        
        return torch.tensor(sequence, dtype=torch.float32)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
            sequence: [sequence_length, 4] tensor of normalized telemetry
            metadata: Dict with incident metadata for optional use
        """
        incident = self.incidents[idx]
        
        # Generate synthetic telemetry from incident metadata
        sequence = self._generate_telemetry_sequence(incident)
        
        # Extract useful metadata
        metadata = {
            'incident_id': incident.get('incident_id'),
            'cascade_depth': incident.get('cascade_depth', 0),
            'severity_level': incident.get('severity_level', 3),
            'estimated_delay_minutes': incident.get('estimated_delay_minutes', 0),
            'archetype': incident.get('archetype', 'Unknown'),
            'is_peak': incident.get('is_peak', False),
        }
        
        return sequence, metadata


class LSTMTrainer:
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
        LSTM-specific trainer for cascade delay prediction.
        
        Args:
            model: LSTMEncoder instance
            optimizer: Optimizer (Adam recommended)
            criterion: Loss function (MSE for reconstruction)
            device: 'cuda' or 'cpu'
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
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Runs one epoch of training.
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        
        for batch_idx, (sequences, metadata) in enumerate(dataloader):
            self.optimizer.zero_grad()
            
            try:
                sequences = sequences.to(self.device)
                
                # Forward pass: [batch, 10, 4] -> [batch, 64]
                embeddings = self.model(sequences)
                
                # For self-supervised learning, predict sequence summary statistics
                # Target is derived from the sequence itself
                target_summary = sequences.mean(dim=1)  # [batch, 4]
                
                # Project embedding to match target dimension
                prediction = embeddings[:, :4]  # Take first 4 dims of embedding
                
                loss = self.criterion(prediction, target_summary)
                loss.backward()
                
                # Gradient clipping (critical for RNNs)
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
    
    def validate_epoch(self, dataloader: DataLoader) -> float:
        """
        Runs one epoch of validation (no gradient updates).
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for sequences, metadata in dataloader:
                try:
                    sequences = sequences.to(self.device)
                    
                    embeddings = self.model(sequences)
                    
                    target_summary = sequences.mean(dim=1)
                    prediction = embeddings[:, :4]
                    
                    loss = self.criterion(prediction, target_summary)
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
            early_stopping_patience: Stop if no improvement for N epochs
            save_best_only: Only save checkpoints when validation improves
            save_interval: Save checkpoint every N epochs
        
        Returns:
            Training history dictionary
        """
        logger.info(f"🚀 Starting LSTM training for {epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training phase
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validation phase
            val_loss = None
            if val_loader:
                val_loss = self.validate_epoch(val_loader)
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


# Main Training Script
if __name__ == "__main__":
    print("=" * 70)
    print("LSTM Cascade Encoder Training - Neural Rail Conductor")
    print("=" * 70)
    
    # Configuration
    INCIDENTS_PATH = "data/processed/incidents.json"
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    EARLY_STOPPING_PATIENCE = 10
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Device: {device}")
    
    # Load datasets (no enricher needed - generates synthetic telemetry)
    print(f"\n📂 Loading incident data from {INCIDENTS_PATH}...")
    train_dataset = IncidentDataset(INCIDENTS_PATH, split="train", sequence_length=10)
    test_dataset = IncidentDataset(INCIDENTS_PATH, split="test", sequence_length=10)
    
    # Split train into train/val (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    print(f"   Train: {len(train_subset)} samples")
    print(f"   Val:   {len(val_subset)} samples")
    print(f"   Test:  {len(test_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Initialize model
    print(f"\n🧠 Creating LSTM model (hidden={HIDDEN_SIZE}, layers={NUM_LAYERS})...")
    model = LSTMEncoder(
        input_size=4,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=64,
        bidirectional=False,
        dropout=0.2,
        use_attention=True
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=5, 
        factor=0.5
    )
    criterion = nn.MSELoss()
    
    # Initialize trainer
    trainer = LSTMTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        log_dir="runs/lstm_cascade_encoder",
        checkpoint_dir="checkpoints/lstm",
        gradient_clip=1.0
    )
    
    # Train model
    print("\n" + "=" * 70)
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        save_best_only=True
    )
    
    # Final evaluation on test set
    print("\n" + "=" * 70)
    print("📊 Final Evaluation on Test Set")
    print("=" * 70)
    test_loss = trainer.validate_epoch(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Training summary
    print("\n" + "=" * 70)
    print("📊 Training Summary")
    print("=" * 70)
    print(f"   Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"   Final Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"   Best Val Loss: {trainer.best_val_loss:.4f}")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Total Epochs: {history['epoch']}")
    
    trainer.close()
    
    print("\n" + "=" * 70)
    print("✅ Training completed successfully!")
    print("=" * 70)
    print(f"📁 Checkpoints saved to: checkpoints/lstm/")
    print(f"📈 TensorBoard logs: runs/lstm_cascade_encoder/")
    print(f"   View with: tensorboard --logdir=runs/lstm_cascade_encoder")
    print("=" * 70)

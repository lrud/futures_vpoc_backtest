"""
Distributed training utilities for futures ML models.
Optimized for AMD ROCm architecture.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler

from src.ml.model import AMDOptimizedFuturesModel
from src.core.data import FuturesDataManager
from src.config.settings import TRAINING_DIR
from src.ml.feature_engineering import prepare_features_and_labels


class AMDFuturesTensorParallel:
    """
    Distributed training implementation for futures prediction models.
    Optimized for AMD GPUs using ROCm and tensor parallelism.
    """
    
    def __init__(self, session_type='RTH', contract='ES'):
        """
        Initialize distributed training manager.
        
        Parameters:
        -----------
        session_type : str
            Type of trading session ('RTH' or 'ETH')
        contract : str
            Futures contract to use (e.g., 'ES' for S&P 500)
        """
        self.session_type = session_type
        self.contract = contract
        self.data_manager = FuturesDataManager()
        self.model = None
        self.feature_columns = None
        self.scaler = None
    
    def _setup_distributed_env(self):
        """Configure environment variables for distributed training."""
        # Set PyTorch distributed environment variables
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        
        # ROCm-specific environment variables
        os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
        os.environ['HIP_VISIBLE_DEVICES'] = '0,1'  # Set available GPUs
        
        # Performance optimization for ROCm
        os.environ['GPU_MAX_HEAP_SIZE'] = '100'
        os.environ['GPU_MAX_ALLOC_PERCENT'] = '100'
        os.environ['GPU_SINGLE_ALLOC_PERCENT'] = '100'
    
    def train_ddp(self, num_epochs=50, batch_size=64, learning_rate=1e-3, 
                 use_feature_selection=True, max_features=15):
        """
        Launch distributed training across multiple GPUs.
        
        Parameters:
        -----------
        num_epochs : int
            Number of training epochs
        batch_size : int
            Batch size per GPU
        learning_rate : float
            Initial learning rate
        use_feature_selection : bool
            Whether to perform feature selection
        max_features : int
            Maximum number of features to use if feature selection is enabled
        """
        # Setup distributed environment
        self._setup_distributed_env()
        
        # Determine number of available GPUs
        world_size = torch.cuda.device_count()
        if world_size < 1:
            print("No GPUs available. Falling back to CPU training.")
            world_size = 1
        
        # Prepare data for training - IMPORTANT: This must match the ML_TEST approach
        # The original code doesn't pass session_type or contract to load_futures_data
        data = self.data_manager.load_futures_data()
        
        # Feature engineering
        X, y, self.feature_columns, self.scaler = prepare_features_and_labels(
            data, 
            use_feature_selection=use_feature_selection,
            max_features=max_features
        )
        
        # Launch distributed processes
        print(f"Starting distributed training with {world_size} processes")
        mp.spawn(
            self._ddp_worker,
            args=(world_size, X, y, num_epochs, batch_size, learning_rate),
            nprocs=world_size,
            join=True
        )
    
    def _ddp_worker(self, rank, world_size, X, y, num_epochs, batch_size, learning_rate):
        """
        Worker function for each distributed process.
        
        Parameters:
        -----------
        rank : int
            Process rank
        world_size : int
            Total number of processes
        X : numpy.ndarray
            Feature matrix
        y : numpy.ndarray
            Target vector
        num_epochs : int
            Number of training epochs
        batch_size : int
            Batch size per GPU
        learning_rate : float
            Initial learning rate
        """
        try:
            print(f"[Worker {rank}] Initializing process group")
            # Use gloo backend if CUDA is not available
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(
                backend=backend,
                init_method='env://',
                rank=rank,
                world_size=world_size
            )
            
            # Set device for this process
            device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                torch.cuda.set_device(device)
            
            # Create model instance
            input_dim = X.shape[1]
            model = AMDOptimizedFuturesModel(input_dim=input_dim)
            model = model.to(device)
            
            # Wrap model for distributed training
            model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
            
            # Prepare optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            
            # Loss function
            criterion = torch.nn.MSELoss()
            
            # Mixed precision
            scaler = GradScaler()
            
            # Create PyTorch datasets and dataloaders
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)
            
            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
            
            # Create distributed sampler
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank
            )
            
            train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=train_sampler,
                pin_memory=True
            )
            
            # Training loop
            for epoch in range(num_epochs):
                model.train()
                train_sampler.set_epoch(epoch)
                running_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass with mixed precision
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                    
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    running_loss += loss.item()
                
                # Average loss across all processes
                avg_loss = running_loss / len(train_loader)
                dist.all_reduce(torch.tensor(avg_loss).to(device))
                avg_loss = avg_loss / world_size
                
                # Update scheduler
                if rank == 0:
                    scheduler.step(avg_loss)
                    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
                
                # Save checkpoint at intervals
                if rank == 0 and (epoch + 1) % 10 == 0:
                    self._save_checkpoint(model.module, epoch, optimizer, scaler)
            
            # Save final model if this is the main process
            if rank == 0:
                self._save_final_model(model.module)
            
            print(f"[Worker {rank}] Training complete")
            
        except Exception as e:
            print(f"[Worker {rank}] Error: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Cleanup
            if dist.is_initialized():
                print(f"[Worker {rank}] Cleaning up distributed group")
                dist.destroy_process_group()
    
    def _save_checkpoint(self, model, epoch, optimizer, scaler):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(TRAINING_DIR, f"es_futures_model_epoch_{epoch+1}.pt")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'amp_scaler': scaler,
            'feature_columns': self.feature_columns,
            'feature_scaler': self.scaler
        }, checkpoint_path)
        
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def _save_final_model(self, model):
        """Save the final trained model."""
        final_model_path = os.path.join(TRAINING_DIR, "es_futures_model_final.pt")
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_columns': self.feature_columns,
            'feature_scaler': self.scaler,
        }, final_model_path)
        
        print(f"Final model saved to {final_model_path}")
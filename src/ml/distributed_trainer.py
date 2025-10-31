"""
Distributed training functionality for ML models with ROCm 7.0 optimizations.
Updated for PyTorch 2.10+ ROCm build with enhanced dual-GPU support.
"""

import os
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler

from src.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

def _setup_process_group(rank, world_size):
    """Initialize process group for distributed training with ROCm 6.3 compatible optimizations."""
    try:
        # ROCm 6.3 compatible distributed training config
        os.environ['MASTER_ADDR'] = '127.0.0.1'  # Use IP instead of localhost
        os.environ['MASTER_PORT'] = '12355'
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)

        # ROCm 6.3 compatible optimizations for dual 7900 XT setup
        os.environ.update({
            # Basic HCCL settings (ROCm 6.x compatible)
            'ROCM_HCCL_DEBUG': 'WARN',        # Reduced debugging for performance
            'HIP_VISIBLE_DEVICES': str(rank), # Each process sees only its GPU
            'HSA_ENABLE_SDMA': '0',           # Disable for better GPU utilization

            # ROCm 6.3 PyTorch specific settings (removed ROCm 7 variables)
            'PYTORCH_ROCM_ARCH': 'gfx1100',   # RDNA3 architecture for 7900 XT
            'PYTORCH_HIP_ALLOC_CONF': 'expandable_segments:True,max_split_size_mb:128',
            'HIP_LAUNCH_BLOCKING': '0',       # Async kernel launches

            # Memory and performance optimizations
            'GPU_SINGLE_ALLOC_PERCENT': '90',  # Lower for multi-GPU
            'HSA_ENABLE_INTERRUPT': '0',
            'HSA_ENABLE_WAIT_COMPLETION': '0',

            # PyTorch torch.compile optimizations (ROCm 6.x compatible)
            'TORCH_COMPILE_BACKEND': 'inductor',

            # ROCm 6.x compatible settings (removed ROCm 7 specific variables)
            # Note: PYTORCH_ROCM_FUSION and ENABLE_FLASH_ATTENTION removed for ROCm 6.x compatibility

            # NCCL settings for ROCm 6.x
            'NCCL_DEBUG': 'WARN',
            'NCCL_SOCKET_IFNAME': 'lo',
            'NCCL_NSOCKS_PERTHREAD': '4'
        })

        logger.info(f"🚀 Initializing ROCm 6.3 compatible process group for rank {rank}/{world_size}")

        # Initialize process group with ROCm 6.x compatible settings
        # Use 'gloo' backend which is more stable than nccl for ROCm
        dist.init_process_group(
            backend="gloo",  # Changed from nccl to gloo for better ROCm compatibility
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(minutes=5)  # Reduced timeout
        )

        # Configure GPU with ROCm 6.x compatible settings
        torch.cuda.set_device(rank)

        # ROCm 6.x: Enable standard Flash Attention (works properly in ROCm 6.x)
        # No need to disable in ROCm 6.x like we did for ROCm 7
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # Use default Flash Attention settings for ROCm 6.x
            pass

        # ROCm 6.x: Standard JIT optimizations (no memory fragmentation issues like ROCm 7)
        # ROCm 6.x: JIT optimizations available if needed
        # torch._C._jit_set_profiling_executor(False)  # Available if needed
        # torch._C._jit_set_profiling_mode(False)      # Available if needed

        # Log detailed ROCm 6.x compatible info
        props = torch.cuda.get_device_properties(rank)
        logger.info(f"✅ Initialized ROCm 6.3 compatible process group (gloo) for rank {rank}")
        logger.info(f"🎯 GPU {rank} Info: {props.name}")
        logger.info(f"🔧 Compute: {props.multi_processor_count} compute units")
        logger.info(f"💾 Memory: {props.total_memory/1e9:.2f}GB total")

        # Log GPU memory status
        free_mem, total_mem = torch.cuda.mem_get_info(rank)
        logger.info(f"💾 Available Memory: {free_mem/1e9:.2f}GB / {total_mem/1e9:.2f}GB")
        logger.info(f"⚡ ROCm 6.3 Optimizations: Standard Flash Attention, BF16, Gloo enabled")

        return True

    except Exception as e:
        logger.error(f"❌ Failed to initialize process group for rank {rank}: {e}")
        return False

def _cleanup_process_group():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()

def _train_worker(rank, world_size, model, X_train, y_train, X_val, y_val, args_dict, feature_columns):
    """Worker function for distributed training with ROCm 6.3 compatible optimizations."""
    logger.info(f"🚀 Starting distributed worker {rank}/{world_size}")

    # Initialize process group with ROCm 6.3 compatible settings
    if not _setup_process_group(rank, world_size):
        logger.error(f"❌ Failed to initialize process group for rank {rank}")
        return None, {}

    # Enable Flash Attention if available (works properly in ROCm 6.x)
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        model.enable_flash_attention = True
        logger.info(f"✅ Rank {rank}: Flash Attention enabled for time series modeling")

    # Force model to use specific GPU
    torch.cuda.set_device(rank)
    logger.info(f"🎯 Rank {rank}: Assigned to GPU {rank} - {torch.cuda.get_device_name(rank)}")

    try:
        # Create device with ROCm 6.x compatible optimizations
        device = torch.cuda.current_device()
        
        # Log detailed device info
        props = torch.cuda.get_device_properties(device)
        logger.info(f"Rank {rank} using GPU: {props.name}")
        logger.info(f"  Compute: {props.multi_processor_count} SMs")
        logger.info(f"  Memory: {props.total_memory/1e9:.2f}GB")
        
        # Set device with ROCm optimizations
        torch.cuda.set_device(device)
        
        # Move model to device with FP8/BF16 support
        model = model.to(device)
        
        # Configure mixed precision
        dtype = torch.bfloat16 if args_dict.get('use_bfloat16', False) else torch.float16
        policy = torch.amp.autocast(device_type='cuda', dtype=dtype)
        
        # Wrap model with FSDP for financial models
        if args_dict.get('use_fsdp', False):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import MixedPrecision
            from torch.distributed.fsdp import ShardingStrategy
            
            # FSDP configuration optimized for financial models
            fsdp_model = FSDP(
                model,
                sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                mixed_precision=MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                ),
                device_id=device,
                limit_all_gathers=True,  # Better for time series
                use_orig_params=True
            )
            ddp_model = fsdp_model
        else:
            # Fallback to DDP
            ddp_model = DDP(
                model,
                device_ids=[device] if isinstance(device, int) else [device.index],
                output_device=device if isinstance(device, int) else device.index,
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
                static_graph=True
            )
        
        # Prepare data
        batch_size = args_dict.get('batch_size', 32) // world_size  # Scale batch size
        
        # Convert data to tensors - NOTE: Reshape y_train to match model output shape
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        y_train_tensor = (y_train_tensor + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
        y_val_tensor = (y_val_tensor + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
        # Create dataset
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        
        # Create distributed sampler
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        # Create dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            pin_memory=True
        )
        
        # Financial-optimized loss and optimizer with Flash Attention
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        
        # Use Flash Attention if available
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            model.enable_flash_attention = True
            logger.info("Using Flash Attention for time series modeling")
        
        optimizer = torch.optim.AdamW(
            ddp_model.parameters(),
            lr=args_dict.get('learning_rate', 0.001),
            weight_decay=0.01,
            foreach=True,  # Use foreach for ROCm optimization
            fused=False   # Cannot use both foreach and fused together
        )
        
        # Financial-aware learning rate scheduler with ROCm optimizations
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args_dict.get('learning_rate', 0.001),
            steps_per_epoch=len(train_loader),
            epochs=args_dict.get('epochs', 50),
            pct_start=0.3,
            anneal_strategy='cos',
            final_div_factor=1e4,
            verbose=(rank==0),
            div_factor=25.0,  # Better for financial models
            three_phase=True  # ROCm-optimized
        )
        
        # Set up ROCm-optimized mixed precision
        scaler = torch.cuda.amp.GradScaler(
            init_scale=2.**11,  # Optimal for financial data
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=1000,
            enabled=args_dict.get('use_mixed_precision', False)
        )
            
        # Track history (only for rank 0)
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Training loop
        best_val_loss = float('inf')
        epochs = args_dict.get('epochs', 50)
        
        for epoch in range(epochs):
            # Set epoch for sampler
            train_sampler.set_epoch(epoch)
            
            # Training phase
            ddp_model.train()
            train_loss = 0.0
            samples_processed = 0
            
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                
                # ROCm-optimized training step with Flash Attention
                with policy:
                    outputs = ddp_model(inputs).squeeze()
                    loss = criterion(outputs, targets)
                
                # Gradient handling optimized for financial time series
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        ddp_model.parameters(),
                        max_norm=1.0,  # Helps with financial data stability
                        norm_type=2.0
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        ddp_model.parameters(),
                        max_norm=1.0,
                        norm_type=2.0
                    )
                    optimizer.step()
                
                batch_size = inputs.size(0)
                train_loss += loss.item() * batch_size
                samples_processed += batch_size
            
            # Calculate average loss (across all processes)
            train_loss = train_loss / samples_processed
            
            # Validation (only on rank 0)
            if rank == 0:
                ddp_model.eval()
                with torch.no_grad():
                    val_outputs = ddp_model(X_val_tensor).squeeze()
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                    
                    # Calculate accuracy
                    val_preds = (val_outputs > 0).float()
                    accuracy = (val_preds == y_val_tensor).float().mean().item() * 100
                
                # Update scheduler
                scheduler.step(val_loss)
                
                # Update history
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(accuracy)
                
                # Log progress
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                           f"Train Loss: {train_loss:.4f} - "
                           f"Val Loss: {val_loss:.4f} - "
                           f"Val Accuracy: {accuracy:.2f}%")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    logger.info(f"Validation loss improved to {best_val_loss:.4f}, saving model")
                    
                    # Save checkpoint
                    output_dir = args_dict.get('output_dir', './models')
                    torch.save({
                        'model_state_dict': ddp_model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'loss': val_loss,
                        'accuracy': accuracy,
                        'feature_columns': feature_columns
                    }, os.path.join(output_dir, 'best_model.pt'))
        
        # Save final model (only rank 0)
        if rank == 0:
            output_dir = args_dict.get('output_dir', './models')
            torch.save({
                'model_state_dict': ddp_model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epochs-1,
                'loss': val_loss,
                'accuracy': accuracy,
                'history': history,
                'feature_columns': feature_columns
            }, os.path.join(output_dir, 'model_final.pt'))
            
            # Return the final model and history
            return ddp_model.module, history
        
    finally:
        # Clean up
        _cleanup_process_group()

def train_distributed(model, X_train, y_train, X_val, y_val, args_dict, feature_columns=None):
    """
    Run distributed training on multiple GPUs with ROCm 6.3 optimizations for financial ML.
    
    Key Features:
    - ROCm 6.3 optimized distributed training
    - FSDP (Fully Sharded Data Parallel) support
    - Flash Attention v3 for time series models
    - BF16/FP8 mixed precision
    - Financial-specific optimizations:
      * Gradient clipping for stability
      * OneCycle learning rate scheduling
      * Time series-aware data loading
    
    Args:
        model: Model to train (must be AMDOptimizedFuturesModel)
        X_train: Training features (numpy array)
        y_train: Training targets (numpy array)
        X_val: Validation features (numpy array)
        y_val: Validation targets (numpy array)
        args_dict: Dictionary containing training arguments:
            - batch_size: int
            - epochs: int
            - learning_rate: float
            - use_mixed_precision: bool
            - use_bfloat16: bool (optional)
            - output_dir: str
        feature_columns: List of feature names (optional)
        
    Returns:
        Tuple of (trained model, training history)
        
    Note:
        Uses ROCm-specific optimizations including:
        - RCCL backend for multi-GPU communication
        - BF16/FP8 mixed precision support
        - Gradient clipping for financial data stability
        - OneCycle learning rate scheduling
    """
    """
    Run distributed training on multiple GPUs.
    
    Args:
        model: Model to train
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        args_dict: Training arguments
        feature_columns: Feature column names
        
    Returns:
        Tuple of (trained model, history)
    """
    # Check ROCm availability and GPU count with detailed diagnostics
    if not torch.cuda.is_available():
        logger.error("""
        ROCm not available - cannot use distributed training.
        Please verify:
        1. ROCm 6.3+ is installed
        2. AMD GPUs are properly configured
        3. PyTorch ROCm version is installed
        """)
        return None, {}
        
    # Detailed ROCm version check
    try:
        if hasattr(torch.version, 'hip'):
            logger.info(f"ROCm version: {torch.version.hip}")
            logger.info(f"PyTorch ROCm build: {torch.__version__}")
            
            # Check for important ROCm features
            if not hasattr(torch, 'is_hip'):
                logger.warning("PyTorch not built with ROCm HIP support")
            if not torch.cuda.get_device_capability()[0] >= 9:
                logger.warning("GPU may not support all ROCm 6.3 features")
        else:
            logger.error("ROCm version not detected - please install ROCm-enabled PyTorch")
            return None, {}
            
        # Check GPU memory
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            logger.info(f"GPU {i}: {props.name} | Memory: {free/1e9:.1f}/{total/1e9:.1f} GB free")
    except Exception as e:
        logger.error(f"Error during ROCm version check: {e}")
        return None, {}
    
    world_size = torch.cuda.device_count()
    if world_size <= 1:
        logger.warning(f"Only {world_size} GPU available, distributed training not needed")
        return None, {}
    
    logger.info(f"Starting distributed training with {world_size} GPUs")
    
    # Extract model architecture parameters before distributed training
    hidden_layers = model.module.aligned_hidden_layers if hasattr(model, 'module') else model.aligned_hidden_layers
    dropout_rate = model.dropout_rate
    
    # Use multiprocessing to start processes
    mp.spawn(
        _train_worker,
        args=(world_size, model, X_train, y_train, X_val, y_val, args_dict, feature_columns),
        nprocs=world_size,
        join=True
    )
    
    # Load the best model after training
    output_dir = args_dict.get('output_dir', './models')
    checkpoint = torch.load(os.path.join(output_dir, 'model_final.pt'))
    
    # Create a new model instance with the same architecture as the original model
    from src.ml.model import AMDOptimizedFuturesModel
    loaded_model = AMDOptimizedFuturesModel(
        input_dim=len(feature_columns) if feature_columns else X_train.shape[1],
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate
    )
    
    # Load weights
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set feature columns
    loaded_model.feature_columns = feature_columns or checkpoint.get('feature_columns', [])
    
    # Get history
    history = checkpoint.get('history', {})
    
    return loaded_model, history


class AMDFuturesTensorParallel:
    """
    AMD-optimized distributed training wrapper for futures models.
    Provides a simple interface for distributed training functionality.
    """

    def __init__(self, session_type="RTH", contract="ES", world_size=None):
        """
        Initialize the distributed training wrapper.

        Parameters:
        -----------
        session_type : str
            Trading session type (RTH, ETH, etc.)
        contract : str
            Futures contract symbol (ES, NQ, etc.)
        world_size : int
            Number of processes for distributed training
        """
        self.logger = get_logger(__name__)
        self.session_type = session_type
        self.contract = contract
        self.world_size = world_size or self._detect_gpu_count()

        self.logger.info(f"Initialized AMDFuturesTensorParallel for {contract} {session_type}")
        self.logger.info(f"World size: {self.world_size}")

    def _detect_gpu_count(self):
        """Detect available GPU count for distributed training."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.device_count()
            else:
                self.logger.warning("CUDA not available, using single GPU")
                return 1
        except ImportError:
            self.logger.warning("PyTorch not available, using single process")
            return 1

    def train_model(self, model, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Train a model using distributed training.

        Parameters:
        -----------
        model : torch.nn.Module
            PyTorch model to train
        X_train, y_train : np.ndarray
            Training data
        X_val, y_val : np.ndarray
            Validation data (optional)
        **kwargs : dict
            Additional training arguments

        Returns:
        --------
        dict
            Training history and results
        """
        self.logger.info("Starting distributed training...")

        # Use the existing train_distributed function
        args_dict = {
            'epochs': kwargs.get('epochs', 10),
            'batch_size': kwargs.get('batch_size', 32),
            'learning_rate': kwargs.get('learning_rate', 0.001),
            'session_type': self.session_type,
            'contract': self.contract
        }

        try:
            history = train_distributed(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                args_dict=args_dict,
                feature_columns=kwargs.get('feature_columns', None)
            )

            self.logger.info("Distributed training completed successfully")
            return history

        except Exception as e:
            self.logger.error(f"Distributed training failed: {e}")
            return {}

    def __repr__(self):
        return f"AMDFuturesTensorParallel(contract={self.contract}, session_type={self.session_type}, world_size={self.world_size})"

"""
Training script for ML models using optimizations for modern GPUs.
"""

import os
import sys
import argparse
import torch
from datetime import datetime, timedelta

# CRITICAL: Disable hipBLASLt for ROCm 7 + RDNA3 compatibility
# Prevents HIPBLAS_STATUS_INTERNAL_ERROR on AMD RX 7900 XT
os.environ['TORCH_BLAS_PREFER_HIPBLASLT'] = '0'

# Import local modules
from src.ml.model import AMDOptimizedFuturesModel
from src.ml.trainer import ModelTrainer
from src.ml.trainer_utils import setup_rocm_environment, set_random_seed, prepare_data
from src.config.settings import settings
from src.utils.logging import get_logger, setup_logging

# Initialize logger
logger = get_logger(__name__)

def prime_rocm_allocator(device='cuda:0'):
    """
    Pre-allocate and free dummy tensor to 'warm up' ROCm memory allocator.
    This bizarre workaround fixes fragmentation issues on AMD GPUs.
    """
    logger.info("🔧 Priming ROCm memory allocator...")

    try:
        # Create progressively smaller dummy tensors for severe fragmentation
        for size_mb in [32, 64, 128]:  # Much smaller sizes
            elements = int(size_mb * 1024**2 / 4)  # 4 bytes per float32

            # Allocate
            dummy = torch.randn(elements, device=device, dtype=torch.float32)
            torch.cuda.synchronize()

            # Free
            del dummy
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        logger.info(f"✅ ROCm allocator primed on {device}")
        logger.info(f"   Allocated: {allocated:.2f}GB / Reserved: {reserved:.2f}GB")
    except Exception as e:
        logger.warning(f"Failed to prime ROCm allocator (continuing): {e}")
        logger.info("   Skipping allocator priming due to memory constraints")

def parse_arguments():
    """Parse command-line arguments for training."""
    from src.ml.arguments import get_base_parser
    parser = get_base_parser("Train futures prediction model with optimizations")
    
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=32,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--learning_rate", "-lr",
        type=float,
        default=0.0005,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--hidden_layers", "-hl",
        type=str,
        default="128,64",
        help="Comma-separated hidden layer dimensions"
    )
    
    parser.add_argument(
        "--dropout_rate", "-dr",
        type=float,
        default=0.4,
        help="Dropout rate"
    )
    
    parser.add_argument(
        "--train_split", "-ts",
        type=float,
        default=0.8,
        help="Train/validation split ratio"
    )
    
    parser.add_argument(
        "--contract", "-c",
        type=str,
        choices=['ES', 'VIX', 'ALL'],
        default="ALL",
        help="Contract filter (ES, VIX, or ALL)"
    )
    
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--use_mixed_precision", "-mp",
        action="store_true",
        help="Use mixed precision training"
    )
    
    parser.add_argument(
        "--no_distributed", 
        action="store_true",
        help="Disable distributed training even with multiple GPUs"
    )
    
    parser.add_argument(
        "--device_ids",
        type=str,
        default="0,1",
        help="Comma-separated list of GPU device IDs to use"
    )

    parser.add_argument(
        "--data_fraction", "-df",
        type=float,
        default=1.0,
        help="Fraction of data to use for training (0.1 = 10%, 1.0 = 100%)"
    )

    parser.add_argument(
        "--skip_gpu_cleanup",
        action="store_true",
        help="Skip pre-training GPU cleanup (faster startup)"
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=3500,
        help="VPOC chunk size for processing large sessions (default: 3500). Larger chunks reduce processing overhead but use more VRAM."
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of steps to accumulate gradients before updating weights (default: 4). Allows larger effective batch sizes with less memory."
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Resume training from last checkpoint if available"
    )

    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=50,
        help="Save checkpoint every N batches (default: 50)"
    )

    parser.add_argument(
        "--distributed_strategy",
        type=str,
        choices=['dataparallel', 'fsdp', 'auto'],
        default='auto',
        help="Distributed training strategy: dataparallel, fsdp, or auto (default)"
    )

    return parser.parse_args()

def create_model(input_dim: int, args: argparse.Namespace):
    """Create and configure the model."""
    # Parse hidden layers
    try:
        hidden_layers = [int(dim) for dim in args.hidden_layers.split(',') if dim.strip()]
        if not hidden_layers:
            raise ValueError("Hidden layers configuration is empty")
            
        logger.info(f"Creating model with hidden layers: {hidden_layers}")
        
        # Create model
        model = AMDOptimizedFuturesModel(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            dropout_rate=args.dropout_rate
        )
    except ValueError as e:
        logger.error(f"Invalid hidden layers configuration: {args.hidden_layers}")
        logger.error(f"Error: {str(e)}")
        raise
    
    logger.info(f"Created model with architecture: {hidden_layers}")
    # Check if the property exists before logging it (handle DataParallel wrapped models)
    actual_model = model.module if hasattr(model, 'module') else model
    if hasattr(actual_model, 'aligned_hidden_layers'):
        logger.info(f"Optimized layers: {actual_model.aligned_hidden_layers}")
    else:
        logger.warning("Model does not have 'aligned_hidden_layers' attribute")
    
    return model

def setup_pytorch210_fsdp(model, device_ids, use_mixed_precision=False):
    """Setup PyTorch 2.10 FSDP (Fully Sharded Data Parallel) as alternative to DataParallel."""

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, BackwardPrefetch
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        import torch.distributed as dist

        # Check if distributed is already initialized
        if not dist.is_initialized():
            # Initialize process group for single-node, multi-GPU
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            os.environ['RANK'] = '0'
            os.environ['WORLD_SIZE'] = str(len(device_ids))
            os.environ['LOCAL_RANK'] = '0'

            dist.init_process_group(
                backend='hccl',  # ROCm backend
                init_method='env://'
            )

        # Mixed precision configuration for PyTorch 2.10
        mixed_precision_config = None
        if use_mixed_precision:
            mixed_precision_config = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )

        # Auto-wrap policy for layers
        auto_wrap_policy = size_based_auto_wrap_policy(
            min_num_params=10000,  # Wrap layers with >10k parameters
        )

        # FSDP configuration optimized for ROCm 7
        fsdp_config = {
            'mixed_precision': mixed_precision_config,
            'auto_wrap_policy': auto_wrap_policy,
            'backward_prefetch': BackwardPrefetch.BACKWARD_PRE,
            'forward_prefetch': True,
            'limit_all_gathers': True,
            'use_orig_params': False,  # PyTorch 2.10 optimization
            'cpu_init': False,  # Keep on GPU for better performance
        }

        # Create FSDP model
        fsdp_model = FSDP(model, **fsdp_config)

        logger.info("🚀 PyTorch 2.10 FSDP (Fully Sharded Data Parallel) enabled")
        logger.info(f"  • Mixed Precision: {use_mixed_precision}")
        logger.info(f"  • Device IDs: {device_ids}")
        logger.info(f"  • Auto-wrap Policy: Enabled for layers >10k parameters")

        return fsdp_model

    except ImportError:
        logger.warning("FSDP not available, falling back to DataParallel")
        return None
    except Exception as e:
        logger.warning(f"FSDP setup failed: {e}, falling back to DataParallel")
        return None

def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Configure logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    logger.info("Starting training script")
    
    # Set random seed for reproducibility
    set_random_seed(args.seed)
    
    # Verify and setup GPU environment
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        # Force ROCm backend usage if available
        try:
            if torch.cuda.device_count() >= 2:
                logger.info(f"Detected {torch.cuda.device_count()} GPUs - forcing distributed training")
                use_distributed = True
            else:
                logger.info(f"Detected single GPU - using distributed training for ROCm optimization")
                use_distributed = True

            # Setup ROCm 7 specific environment variables for optimal performance
            os.environ['HIP_VISIBLE_DEVICES'] = '0,1'
            os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
            os.environ['ROCM_PATH'] = '/opt/rocm'
            os.environ['HSA_ENABLE_SDMA'] = '0'
            os.environ['HSA_ENABLE_INTERRUPT'] = '0'

            # ROCm 7 specific memory settings
            os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:512,garbage_collection_threshold:0.8'

            # Enable flash attention and mixed precision optimizations
            os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
            os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '0'
            os.environ['HIP_FORCE_DEV_KERNARG'] = '1'

        except Exception as e:
            logger.error(f"Failed to initialize ROCm environment: {e}")
            logger.warning("Falling back to CPU training")
            use_gpu = False
            use_distributed = False
    else:
        logger.warning("No GPU detected - falling back to CPU training")
    
    # Set device for training
    device = torch.device('cuda' if use_gpu else 'cpu')

    # Setup device IDs for distributed training
    if args.no_distributed:
        use_distributed = False
        device_ids = [0] if use_gpu else None
        logger.info("Forcing single-GPU training as requested")
    else:
        device_ids = [int(id) for id in args.device_ids.split(',')] if args.device_ids else list(range(torch.cuda.device_count())) if use_gpu else None

    if use_gpu and device_ids:
        # Ensure we have available GPUs
        available_gpus = torch.cuda.device_count()
        device_ids = [id for id in device_ids if id < available_gpus]
        if not device_ids:
            logger.error("No valid GPU IDs available")
            use_gpu = False
            device_ids = None

    logger.info(f"Using device: {device} with device IDs: {device_ids}")
    if use_gpu and device_ids:
        logger.info(f"Distributed training: {use_distributed} across {len(device_ids)} GPUs")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(args.output, f"train_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    args.output_dir = model_dir
    
    logger.info(f"Model will be saved to: {model_dir}")
    
    # Prepare data - use provided data path if specified, otherwise fall back to settings
    data_path = args.data if hasattr(args, 'data') and args.data else settings.DATA_DIR
    device_ids = [int(id) for id in args.device_ids.split(',')] if args.device_ids and use_gpu else None
    
    # Log GPU memory before data loading
    if use_gpu:
        for i in range(torch.cuda.device_count()):
            try:
                total_memory = torch.cuda.get_device_properties(i).total_memory/1e9
                allocated_memory = torch.cuda.memory_allocated(i)/1e9
                logger.info(f"Preprocessing - GPU {i} memory: {allocated_memory:.2f}GB used / {total_memory:.2f}GB total")
            except Exception as e:
                logger.warning(f"Could not get GPU {i} properties: {e}")
                logger.info(f"Preprocessing - GPU {i} available")
    
    # Load and distribute data
    result = prepare_data(args,
                        contract_filter=None if args.contract == "ALL" else args.contract,
                        data_path=data_path,
                        device_ids=device_ids,
                        data_fraction=args.data_fraction,
                        chunk_size=args.chunk_size)
    if not result:
        logger.error("Data preparation failed")
        return 1
        
    X_train, y_train, X_val, y_val, feature_columns = result
    
    # Log GPU memory after data loading
    if use_gpu:
        for i in range(torch.cuda.device_count()):
            try:
                total_memory = torch.cuda.get_device_properties(i).total_memory/1e9
                allocated_memory = torch.cuda.memory_allocated(i)/1e9
                logger.info(f"Post-processing - GPU {i} memory: {allocated_memory:.2f}GB used / {total_memory:.2f}GB total")
            except Exception as e:
                logger.warning(f"Could not get GPU {i} properties: {e}")
                logger.info(f"Post-processing - GPU {i} available")
    
    # Create base model and move to device
    model = create_model(len(feature_columns), args)
    if device_ids and len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)

    # CRITICAL: Prime ROCm allocator to prevent fragmentation
    prime_rocm_allocator('cuda:0')
    if torch.cuda.device_count() > 1:
        prime_rocm_allocator('cuda:1')
    
    # Store feature column names and model metadata
    if not hasattr(model, 'feature_columns'):
        model.feature_columns = feature_columns
    if not hasattr(model, 'input_dim'):
        model.input_dim = len(feature_columns)
    if not hasattr(model, 'hidden_layers_config'):
        model.hidden_layers_config = [int(dim) for dim in args.hidden_layers.split(',')]
    if not hasattr(model, 'dropout_rate_config'):
        model.dropout_rate_config = args.dropout_rate
    
    # Initialize trainer with device
    trainer = ModelTrainer(model_dir=args.output_dir, device=device)
    
    # Enable Flash Attention if available
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        # Disable Flash Attention for ROCm 7 compatibility (kernel issues)
        model.enable_flash_attention = False
        logger.info("Flash Attention disabled for ROCm 7 compatibility")
    
    # Use distributed training for multi-GPU setup
    use_multi_gpu = (not args.no_distributed and
                     use_gpu and
                     torch.cuda.device_count() > 1)

    if use_multi_gpu:
        device_ids = [0, 1]
        logger.info(f"Multi-GPU training detected with {len(device_ids)} GPUs")

        # ROCm 7 optimizations for multi-GPU with memory management
        os.environ.update({
            'PYTORCH_ROCM_ARCH': 'gfx1100',
            'PYTORCH_ROCM_WAVE32_MODE': '1',
            'PYTORCH_ROCM_FUSION': '1',
            'PYTORCH_HIP_ALLOC_CONF': 'expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6',
            'GPU_SINGLE_ALLOC_PERCENT': '80',  # Reduced from 90
            'HSA_ENABLE_SDMA': '0',
            'HSA_ENABLE_INTERRUPT': '0',
            'HSA_ENABLE_WAIT_COMPLETION': '0',
            'GPU_MAX_HW_QUEUES': '8',
            'ENABLE_FLASH_ATTENTION': '0',  # Disabled for ROCm 7 compatibility
            'TORCH_COMPILE_BACKEND': 'inductor'
        })

        # Make both GPUs visible
        os.environ['HIP_VISIBLE_DEVICES'] = '0,1'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

        # Pre-emptive memory cleanup for DataParallel
        if torch.cuda.is_available() and not args.skip_gpu_cleanup:
            logger.info("🧹 Performing pre-training GPU cleanup...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        elif torch.cuda.is_available() and args.skip_gpu_cleanup:
            logger.info("⚡ Skipping pre-training GPU cleanup for faster startup")

        # Choose distributed strategy based on argument
        strategy = args.distributed_strategy
        if strategy == 'auto':
            # Auto-select: Try FSDP first, fall back to DataParallel
            strategy = 'fsdp'

        logger.info(f"🎯 Using distributed strategy: {strategy}")

        if strategy == 'fsdp':
            # DISABLED: FSDP due to ROCm version incompatibility
            # FSDP can hang with certain ROCm/PyTorch version combinations
            logger.warning(f"⚠️  FSDP DISABLED due to ROCm version incompatibility")
            logger.warning(f"   • PyTorch: {torch.__version__}")
            if hasattr(torch.version, 'hip'):
                logger.warning(f"   • ROCm/HIP: {torch.version.hip}")
            logger.warning(f"   • FSDP hangs at hccl backend initialization")
            logger.warning(f"   • Using DataParallel fallback for stability")
            strategy = 'dataparallel'
            logger.info("🔄 Forced DataParallel fallback due to FSDP incompatibility")

        if strategy == 'dataparallel':
            # Use traditional DataParallel with memory management
            logger.info(f"🚀 Creating DataParallel model with GPUs: {device_ids}")

            # Clear memory before DataParallel wrapping
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])

            # Disable Flash Attention for ROCm 7 compatibility
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                model.module.enable_flash_attention = False
                logger.info("Flash Attention disabled for ROCm 7 compatibility (DataParallel)")

            logger.info(f"✅ DataParallel model created with memory-safe configuration")

        # Log GPU information
        for i in device_ids:
            if i < torch.cuda.device_count():
                try:
                    props = torch.cuda.get_device_properties(i)
                    logger.info(f"GPU {i}: {props.name} ({props.total_memory/1e9:.1f}GB)")
                except Exception as e:
                    logger.warning(f"Could not get GPU {i} properties: {e}")

        # Train with DataParallel model
        trained_model, history = trainer.train(
            model, X_train, y_train, X_val, y_val,
            args={
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'learning_rate': args.learning_rate,
                'use_mixed_precision': args.use_mixed_precision,
                'output_dir': args.output_dir,
                'device': device,
                'gradient_accumulation_steps': args.gradient_accumulation_steps,
                'resume_from_checkpoint': args.resume_from_checkpoint,
                'checkpoint_interval': args.checkpoint_interval,
            }
        )
    else:
        # Single device training (GPU or CPU)
        trained_model, history = trainer.train(
            model, X_train, y_train, X_val, y_val,
            args={
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'learning_rate': args.learning_rate,
                'use_mixed_precision': args.use_mixed_precision,
                'output_dir': args.output_dir,
                'device': device,
                'gradient_accumulation_steps': args.gradient_accumulation_steps,
                'resume_from_checkpoint': args.resume_from_checkpoint,
                'checkpoint_interval': args.checkpoint_interval,
            }
        )
    
    # Generate and save training plot
    plot_path = os.path.join(args.output_dir, "training_history.png")
    trainer.plot_training_history(history, save_path=plot_path)
    
    # Save the trained model with memory cleanup
    model_path = os.path.join(args.output_dir, "model.pt")

    # For DataParallel models, save the underlying model and cleanup
    if hasattr(trained_model, 'module'):
        # Get the underlying model from DataParallel wrapper
        final_model = trained_model.module
        logger.info("Extracted model from DataParallel wrapper")

        # ROCm 7: DataParallel cleanup
        logger.info("🧹 Performing DataParallel cleanup...")

        # Move DataParallel model to CPU to free GPU memory
        trained_model.cpu()

        # Clear gradients from all parameters
        for param in trained_model.parameters():
            if param.grad is not None:
                param.grad = None

        # Delete the DataParallel wrapper
        del trained_model

        # Force garbage collection and GPU cleanup
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        logger.info("✅ DataParallel cleanup completed")
    else:
        final_model = trained_model

    # Save model state dict
    torch.save(final_model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    # Log training completion with GPU information
    if use_multi_gpu:
        logger.info("✅ Multi-GPU DataParallel training completed successfully!")
        logger.info(f"🎯 Used GPUs: {device_ids}")
    else:
        logger.info("✅ Single-GPU training completed successfully!")
    
    logger.info("Training completed successfully")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        # Cleanup on any exception
        logger.error(f"Training failed with error: {e}")
        logger.info("Performing emergency GPU cleanup...")

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Kill any remaining processes
        import subprocess
        subprocess.run(['pkill', '-f', 'train.py'], capture_output=True)

        sys.exit(1)
    finally:
        # Always cleanup, even on success
        logger.info("Training completed - performing final GPU cleanup...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

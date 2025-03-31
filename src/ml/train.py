"""
Training script for ML models using optimizations for modern GPUs.
"""

import os
import sys
import argparse
import torch
from datetime import datetime

# Import local modules
from src.ml.model import AMDOptimizedFuturesModel
from src.ml.trainer import ModelTrainer
from src.ml.trainer_utils import setup_rocm_environment, set_random_seed, prepare_data
from src.config.settings import settings
from src.utils.logging import get_logger, setup_logging

# Initialize logger
logger = get_logger(__name__)

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
        if not setup_rocm_environment():
            logger.warning("Falling back to CPU training")
            use_gpu = False
        else:
            # Log detailed GPU information
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                logger.info(f"  Memory: {torch.cuda.get_device_properties(i).total_memory/1e9:.2f}GB")
                logger.info(f"  Compute: {torch.cuda.get_device_properties(i).multi_processor_count} SMs")
    else:
        logger.warning("No GPU detected - falling back to CPU training")
    
    # Set device for training
    device = torch.device('cuda' if use_gpu else 'cpu')
    device_ids = [int(id) for id in args.device_ids.split(',')] if args.device_ids else None
    logger.info(f"Using device: {device} with device IDs: {device_ids}")
    
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
            logger.info(f"Preprocessing - GPU {i} memory: {torch.cuda.memory_allocated(i)/1e9:.2f}GB used / {torch.cuda.get_device_properties(i).total_memory/1e9:.2f}GB total")
    
    # Load and distribute data
    result = prepare_data(args, 
                        contract_filter=None if args.contract == "ALL" else args.contract, 
                        data_path=data_path, 
                        device_ids=device_ids)
    if not result:
        logger.error("Data preparation failed")
        return 1
        
    X_train, y_train, X_val, y_val, feature_columns = result
    
    # Log GPU memory after data loading
    if use_gpu:
        for i in range(torch.cuda.device_count()):
            logger.info(f"Post-processing - GPU {i} memory: {torch.cuda.memory_allocated(i)/1e9:.2f}GB used / {torch.cuda.get_device_properties(i).total_memory/1e9:.2f}GB total")
    
    # Create base model and move to device
    model = create_model(len(feature_columns), args)
    if device_ids and len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)
    
    # Store feature column names and model metadata
    model.feature_columns = feature_columns
    model.input_dim = len(feature_columns)
    model.hidden_layers = [int(dim) for dim in args.hidden_layers.split(',')]
    model.dropout_rate = args.dropout_rate
    
    # Initialize trainer with device
    trainer = ModelTrainer(model_dir=args.output_dir, device=device)
    
    # Enable Flash Attention if available
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        model.enable_flash_attention = True
        logger.info("Flash Attention v3 enabled for model")
    
    # Determine if distributed training should be used
    use_distributed = (not args.no_distributed and 
                      use_gpu and 
                      torch.cuda.device_count() > 1)
    if use_distributed:
        logger.info(f"Using distributed training with {torch.cuda.device_count()} GPUs")
        # Import distributed only if needed
        from src.ml.distributed_trainer import train_distributed
        
        # Set ROCm environment variables for optimal performance
        os.environ['ROCM_HCCL_DEBUG'] = '2'
        os.environ['HIP_VISIBLE_DEVICES'] = ','.join(map(str, range(torch.cuda.device_count())))
        os.environ['HSA_ENABLE_SDMA'] = '0'
        os.environ['GPU_MAX_HW_QUEUES'] = '8'
        os.environ['PYTORCH_ROCM_WAVE32_MODE'] = '1'
        os.environ['HSA_ENABLE_INTERRUPT'] = '0'
        os.environ['HSA_ENABLE_WAIT_COMPLETION'] = '0'
        os.environ['NCCL_NSOCKS_PERTHREAD'] = '8'
        
        # Verify and initialize PyTorch distributed backend
        if not hasattr(torch, 'distributed') or not torch.distributed.is_available():
            logger.warning("PyTorch distributed not available, falling back to single GPU")
            use_distributed = False
        else:
            try:
                torch.distributed.init_process_group(
                    backend='rccl',
                    init_method='env://',
                    world_size=torch.cuda.device_count(),
                    rank=0,
                    timeout=datetime.timedelta(minutes=10)
                )
                
                logger.info(f"Initialized ROCm distributed backend (rccl) with {torch.cuda.device_count()} GPUs")
            except Exception as e:
                logger.error(f"Failed to initialize distributed training: {e}")
                use_distributed = False
                # Fall back to single-GPU training
                trained_model, history = trainer.train(
                    model, X_train, y_train, X_val, y_val,
                    args={
                        'batch_size': args.batch_size,
                        'epochs': args.epochs,
                        'learning_rate': args.learning_rate,
                        'use_mixed_precision': args.use_mixed_precision,
                        'output_dir': args.output_dir,
                        'device': device,
                    }
                )
                return 0
                
            # Log detailed GPU information
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {props.name}")
                logger.info(f"  Memory: {props.total_memory/1e9:.2f}GB")
                logger.info(f"  Compute: {props.multi_processor_count} SMs")
                logger.info(f"  ROCm Capability: {props.major}.{props.minor}")
        
            # Scale batch size for multi-GPU training
            effective_batch_size = args.batch_size
            logger.info(f"Using batch size of {effective_batch_size} per GPU (total {effective_batch_size * torch.cuda.device_count()} across {torch.cuda.device_count()} GPUs)")
            
            trained_model, history = train_distributed(
                model, X_train, y_train, X_val, y_val,
                args_dict={
                    'batch_size': effective_batch_size,
                    'epochs': args.epochs,
                    'learning_rate': args.learning_rate,
                    'use_mixed_precision': args.use_mixed_precision,
                    'use_bfloat16': True,  # Force BF16 for ROCm
                    'output_dir': args.output_dir,
                    'device': device,
                },
                feature_columns=feature_columns
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
            }
        )
    
    # Generate and save training plot
    plot_path = os.path.join(args.output_dir, "training_history.png")
    trainer.plot_training_history(history, save_path=plot_path)
    
    # Save the trained model
    model_path = os.path.join(args.output_dir, "model.pt")
    torch.save(trained_model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    logger.info("Training completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())

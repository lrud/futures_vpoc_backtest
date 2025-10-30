# ES Futures VPOC Strategy - Machine Learning Training Guide

## Overview

This guide provides comprehensive instructions for training machine learning models to enhance the VPOC trading strategy. The system uses AMD GPU-optimized neural networks to filter and improve signal quality.

## Training Architecture

### Model Overview

The system uses a sophisticated neural network architecture specifically optimized for AMD GPUs:

```python
class AMDOptimizedFuturesModel(nn.Module):
    """ROCm 7.0 optimized neural network for futures trading signals"""

    def __init__(self, input_size=50, hidden_sizes=[256, 128, 64], output_size=1):
        super().__init__()

        # Optimized layer sizes for Wave32 mode
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        # Build layers with AMD-specific optimizations
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                self._create_optimized_layer(layer_sizes[i], layer_sizes[i+1])
            )
```

### Key Features

- **Memory Alignment**: 64-byte alignment for optimal AMD GPU performance
- **Wave32 Optimization**: Efficient utilization of RDNA3 architecture
- **Mixed Precision**: BF16 training for improved performance
- **Flash Attention**: Efficient attention mechanisms for sequence data
- **Layer Normalization**: Enhanced stability over batch normalization

## Training Setup

### 1. Environment Preparation

```bash
# Activate the environment
source futures-vpoc-env/bin/activate

# Set GPU environment variables for ROCm 7 optimization
export HIP_VISIBLE_DEVICES=0,1
export PYTORCH_ROCM_ARCH=gfx1100
export GPU_MAX_HW_QUEUES=8

# Enable memory optimizations (required for current VRAM bug)
export PYTORCH_HIP_ALLOC_CONF='expandable_segments:True,max_split_size_mb:128'
```

### 2. Known VRAM Issue

**⚠️ Current Bug**: The system exhibits a VRAM reporting issue where:
- `rocm-smi` shows 99% VRAM usage
- PyTorch reports minimal actual allocation (often < 2MB)
- Training fails with OOM errors despite showing "free" memory

**Workaround**: Use smaller batch sizes (8-16) and the memory allocation settings above. This appears to be a ROCm 7 memory fragmentation issue.

### 2. Data Preparation

#### Feature Engineering Pipeline

```python
from src.ml.feature_engineering import FeatureEngineer

# Initialize feature engineer
feature_engineer = FeatureEngineer(
    lookback_periods=[5, 10, 20, 50],
    target_horizon=1,
    volatility_window=20
)

# Load and prepare data
data = feature_engineer.load_data('DATA/ES/5min/')
features, targets = feature_engineer.prepare_features(data)

# Split data
train_size = int(0.8 * len(features))
val_size = int(0.1 * len(features))
test_size = len(features) - train_size - val_size

train_features = features[:train_size]
train_targets = targets[:train_size]
val_features = features[train_size:train_size+val_size]
val_targets = targets[train_size:train_size+val_size]
test_features = features[train_size+val_size:]
test_targets = targets[train_size+val_size:]
```

#### Feature Categories

1. **Price Features**:
   - Returns over multiple timeframes
   - Momentum indicators
   - Price position relative to VPOC
   - Value Area relationships

2. **Volume Features**:
   - Volume profile metrics
   - VPOC migration patterns
   - Volume-weighted price levels
   - Volume trend analysis

3. **Volatility Features**:
   - GARCH-style volatility estimates
   - Realized volatility
   - Volatility regime indicators
   - Range-based volatility measures

4. **Technical Indicators**:
   - RSI over multiple periods
   - MACD signals
   - Bollinger Band positions
   - Moving average relationships

### 3. Training Configuration

#### Hyperparameter Setup

```python
# training_config.py
TRAINING_CONFIG = {
    # Model Architecture
    'model': {
        'input_size': 50,
        'hidden_sizes': [256, 128, 64],
        'output_size': 1,
        'dropout_rate': 0.2,
        'activation': 'silu',
        'use_layer_norm': True,
    },

    # Training Parameters
    'training': {
        'batch_size': 64,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_epochs': 100,
        'patience': 15,  # Early stopping patience
        'min_delta': 1e-6,
    },

    # Optimization
    'optimizer': {
        'type': 'AdamW',
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
    },

    # Scheduler
    'scheduler': {
        'type': 'cosine',
        'T_max': 100,
        'eta_min': 1e-6,
    },

    # Data Loading
    'data': {
        'num_workers': 8,
        'pin_memory': True,
        'persistent_workers': True,
        'prefetch_factor': 4,
    },

    # Mixed Precision
    'mixed_precision': True,
    'precision': 'bf16',  # For ROCm optimization

    # GPU Configuration
    'device_ids': [0, 1],
    'compile_model': True,
    'compile_mode': 'max-autotune',
}
```

## Training Execution

### Quick Start Training Command

```bash
# Environment setup
export PYTHONPATH=/workspace
export HIP_VISIBLE_DEVICES=0,1
export PYTORCH_ROCM_ARCH=gfx1100
export PYTORCH_HIP_ALLOC_CONF='expandable_segments:True,max_split_size_mb:128'

# Train with merged ES/VIX data (recommended working command)
python src/ml/train.py \
    --data DATA/MERGED/merged_es_vix_test.csv \
    --output TRAINING/ \
    --epochs 30 \
    --batch_size 16 \
    --learning_rate 0.0002 \
    --hidden_layers 192,128,64 \
    --use_mixed_precision
```

### 1. Single GPU Training

```python
# single_gpu_train.py
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.ml.trainer import ModelTrainer
from src.ml.model import AMDOptimizedFuturesModel

def train_single_gpu():
    # Initialize trainer
    trainer = ModelTrainer(model_dir='TRAINING/single_gpu/')

    # Create model
    model = AMDOptimizedFuturesModel(
        input_size=50,
        hidden_sizes=[256, 128, 64],
        output_size=1
    )

    # Compile model for ROCm optimization
    if torch.cuda.is_available() and hasattr(torch, 'compile'):
        model = torch.compile(model, mode='max-autotune')
        print("Model compiled with ROCm optimizations")

    # Prepare data
    train_dataset = TensorDataset(train_features, train_targets)
    val_dataset = TensorDataset(val_features, val_targets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,  # Reduced due to VRAM bug
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # Train model
    history = trainer.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        learning_rate=1e-4,
        save_best=True,
        model_name='vpoc_enhanced_model'
    )

    return history, model

if __name__ == "__main__":
    history, model = train_single_gpu()
```

### 2. Multi-GPU Distributed Training

```python
# distributed_train.py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from src.ml.distributed_trainer import DistributedTrainer

def train_worker(rank, world_size):
    """Training worker for distributed training"""

    # Initialize process group
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        rank=rank,
        world_size=world_size
    )

    # Set device
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # Create distributed trainer
    trainer = DistributedTrainer(
        rank=rank,
        world_size=world_size,
        device=device
    )

    # Train model
    trainer.train_distributed()

def main():
    world_size = torch.cuda.device_count()
    print(f"Starting distributed training on {world_size} GPUs")

    mp.spawn(
        train_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
```

### 3. Advanced Training Features

#### Mixed Precision Training

```python
from torch.cuda.amp import GradScaler, autocast

class MixedPrecisionTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.scaler = GradScaler()

    def train_step(self, data, target, optimizer, criterion):
        optimizer.zero_grad()

        with autocast(dtype=torch.bfloat16):  # ROCm optimized
            output = self.model(data)
            loss = criterion(output, target)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.scaler.step(optimizer)
        self.scaler.update()

        return loss.item()
```

#### Gradient Accumulation

```python
class GradientAccumulationTrainer:
    def __init__(self, model, accumulation_steps=4):
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.current_step = 0

    def train_step(self, data, target, optimizer, criterion):
        self.current_step += 1

        # Forward pass
        with autocast(dtype=torch.bfloat16):
            output = self.model(data)
            loss = criterion(output, target) / self.accumulation_steps

        # Backward pass
        loss.backward()

        # Update weights every accumulation_steps
        if self.current_step % self.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            self.current_step = 0

        return loss.item() * self.accumulation_steps
```

## Model Evaluation

### 1. Performance Metrics

```python
# evaluation.py
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            with autocast(dtype=torch.bfloat16):
                output = model(data)
                loss = criterion(output, target.float())

            total_loss += loss.item()

            # Store predictions and targets
            predictions = torch.sigmoid(output).cpu().numpy()
            targets = target.cpu().numpy()

            all_predictions.extend(predictions)
            all_targets.extend(targets)

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Calculate metrics
    binary_predictions = (all_predictions > 0.5).astype(int)

    metrics = {
        'loss': total_loss / len(test_loader),
        'accuracy': accuracy_score(all_targets, binary_predictions),
        'precision': precision_score(all_targets, binary_predictions),
        'recall': recall_score(all_targets, binary_predictions),
        'f1_score': f1_score(all_targets, binary_predictions),
        'auc_roc': roc_auc_score(all_targets, all_predictions),
    }

    return metrics, all_predictions, all_targets

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    if 'train_accuracy' in history:
        ax2.plot(history['train_accuracy'], label='Training Accuracy')
        ax2.plot(history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)

    plt.tight_layout()
    plt.savefig('TRAINING/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
```

### 2. Model Interpretability

```python
# interpretability.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

class ModelInterpretability:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names

    def calculate_feature_importance(self, test_loader, device='cuda'):
        """Calculate permutation feature importance"""
        self.model.eval()

        # Collect all test data
        all_data = []
        all_targets = []

        with torch.no_grad():
            for data, target in test_loader:
                all_data.append(data.cpu())
                all_targets.append(target.cpu())

        all_data = torch.cat(all_data, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Define scoring function
        def score_func(model, X, y):
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

                predictions = torch.sigmoid(self.model(X_tensor))
                return -np.mean((predictions.cpu().numpy() - y)**2)  # Negative MSE

        # Calculate permutation importance
        result = permutation_importance(
            self.model, all_data.numpy(), all_targets.numpy(),
            scoring=score_func,
            n_repeats=10,
            random_state=42
        )

        # Sort features by importance
        importance_scores = result.importances_mean
        sorted_indices = np.argsort(importance_scores)[::-1]

        return {
            'importance_scores': importance_scores,
            'sorted_indices': sorted_indices,
            'feature_names': [self.feature_names[i] for i in sorted_indices]
        }

    def plot_feature_importance(self, importance_data):
        """Plot feature importance"""
        plt.figure(figsize=(12, 8))

        y_pos = np.arange(len(importance_data['feature_names']))
        plt.barh(y_pos, importance_data['importance_scores'][::-1])
        plt.yticks(y_pos, importance_data['feature_names'][::-1])
        plt.xlabel('Importance Score')
        plt.title('Feature Importance for VPOC Enhancement Model')
        plt.tight_layout()
        plt.savefig('TRAINING/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
```

## Hyperparameter Optimization

### 1. Grid Search

```python
# hyperparameter_search.py
import itertools
from src.ml.trainer import ModelTrainer

def grid_search_hyperparameters():
    """Perform grid search for hyperparameter optimization"""

    # Define hyperparameter grid
    param_grid = {
        'learning_rate': [1e-3, 5e-4, 1e-4, 5e-5],
        'batch_size': [32, 64, 128],
        'hidden_sizes': [
            [128, 64],
            [256, 128, 64],
            [512, 256, 128],
        ],
        'dropout_rate': [0.1, 0.2, 0.3],
        'weight_decay': [1e-4, 1e-5, 1e-6],
    }

    # Generate all combinations
    param_combinations = []
    keys = param_grid.keys()
    values = param_grid.values()

    for combination in itertools.product(*values):
        param_combinations.append(dict(zip(keys, combination)))

    best_score = 0
    best_params = None
    results = []

    for i, params in enumerate(param_combinations):
        print(f"Testing combination {i+1}/{len(param_combinations)}: {params}")

        # Create model with current parameters
        model = AMDOptimizedFuturesModel(
            input_size=50,
            hidden_sizes=params['hidden_sizes'],
            output_size=1,
            dropout_rate=params['dropout_rate']
        )

        # Train model
        trainer = ModelTrainer(model_dir=f'TRAINING/grid_search/run_{i}/')

        history = trainer.train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=50,  # Reduced for grid search
            learning_rate=params['learning_rate'],
            batch_size=params['batch_size'],
            weight_decay=params['weight_decay'],
        )

        # Evaluate model
        val_accuracy = max(history['val_accuracy'])

        results.append({
            'params': params,
            'val_accuracy': val_accuracy,
            'final_loss': history['val_loss'][-1]
        })

        # Update best parameters
        if val_accuracy > best_score:
            best_score = val_accuracy
            best_params = params

    print(f"\nBest parameters: {best_params}")
    print(f"Best validation accuracy: {best_score:.4f}")

    return results, best_params
```

### 2. Bayesian Optimization

```python
# bayesian_optimization.py
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

def bayesian_optimization():
    """Perform Bayesian optimization for hyperparameters"""

    # Define search space
    space = [
        Real(1e-5, 1e-2, prior='log-uniform', name='learning_rate'),
        Integer(32, 256, name='batch_size'),
        Integer(64, 512, name='hidden_size_1'),
        Integer(32, 256, name='hidden_size_2'),
        Real(0.1, 0.5, name='dropout_rate'),
        Real(1e-6, 1e-3, prior='log-uniform', name='weight_decay'),
    ]

    @use_named_args(space)
    def objective(**params):
        """Objective function to minimize (negative accuracy)"""

        # Create model
        hidden_sizes = [params['hidden_size_1'], params['hidden_size_2']]

        model = AMDOptimizedFuturesModel(
            input_size=50,
            hidden_sizes=hidden_sizes,
            output_size=1,
            dropout_rate=params['dropout_rate']
        )

        # Train model
        trainer = ModelTrainer(model_dir='TRAINING/bayesian_opt/')

        history = trainer.train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=50,
            learning_rate=params['learning_rate'],
            batch_size=params['batch_size'],
            weight_decay=params['weight_decay'],
        )

        # Return negative accuracy for minimization
        val_accuracy = max(history['val_accuracy'])
        return -val_accuracy

    # Run optimization
    result = gp_minimize(
        objective,
        space,
        n_calls=50,
        n_initial_points=10,
        random_state=42
    )

    # Extract best parameters
    best_params = {
        'learning_rate': result.x[0],
        'batch_size': result.x[1],
        'hidden_size_1': result.x[2],
        'hidden_size_2': result.x[3],
        'dropout_rate': result.x[4],
        'weight_decay': result.x[5],
    }

    print(f"Best parameters: {best_params}")
    print(f"Best validation accuracy: {-result.fun:.4f}")

    return result, best_params
```

## Model Deployment

### 1. Model Export

```python
# model_export.py
import torch
import torch.jit

def export_model_for_inference(model, export_path='models/vpoc_model.pt'):
    """Export trained model for production inference"""

    model.eval()

    # Create sample input for tracing
    sample_input = torch.randn(1, 50).cuda()

    # Export with TorchScript
    traced_model = torch.jit.trace(model, sample_input)
    traced_model.save(export_path)

    print(f"Model exported to {export_path}")

    # Test exported model
    loaded_model = torch.jit.load(export_path)
    test_output = loaded_model(sample_input)
    print(f"Exported model test: {test_output.shape}")

def export_onnx_model(model, export_path='models/vpoc_model.onnx'):
    """Export model to ONNX format"""

    model.eval()
    sample_input = torch.randn(1, 50).cuda()

    torch.onnx.export(
        model,
        sample_input,
        export_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"ONNX model exported to {export_path}")
```

### 2. Inference Optimization

```python
# inference_optimizer.py
import torch
from src.ml.model import AMDOptimizedFuturesModel

class OptimizedInference:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self.load_and_optimize_model(model_path)

    def load_and_optimize_model(self, model_path):
        """Load and optimize model for inference"""

        # Load model
        model = AMDOptimizedFuturesModel(
            input_size=50,
            hidden_sizes=[256, 128, 64],
            output_size=1
        )

        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Optimize for inference
        model.eval()

        # Compile with torch.compile for ROCm optimization
        if hasattr(torch, 'compile'):
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled for optimized inference")

        # Move to device
        model = model.to(self.device)

        return model

    def predict_batch(self, features):
        """Batch prediction with optimizations"""

        with torch.no_grad():
            # Convert to tensor
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float32)

            features = features.to(self.device)

            # Mixed precision inference
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = self.model(features)
                probabilities = torch.sigmoid(logits)

            return probabilities.cpu().numpy()

    def predict_single(self, features):
        """Single prediction with optimizations"""

        # Add batch dimension
        if features.ndim == 1:
            features = features.unsqueeze(0)

        predictions = self.predict_batch(features)
        return predictions[0]
```

## Troubleshooting Training Issues

### Common Problems and Solutions

1. **CUDA Out of Memory**:
   ```python
   # Reduce batch size
   batch_size = 32  # Try smaller values

   # Enable gradient checkpointing
   from torch.utils.checkpoint import checkpoint

   # Clear cache periodically
   torch.cuda.empty_cache()
   ```

2. **Slow Training**:
   ```python
   # Enable mixed precision
   scaler = GradScaler()

   # Use larger batch sizes
   batch_size = 128

   # Optimize data loading
   num_workers = 8
   pin_memory = True
   persistent_workers = True
   ```

3. **Poor Convergence**:
   ```python
   # Adjust learning rate
   learning_rate = 1e-4

   # Add learning rate scheduler
   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

   # Increase model capacity
   hidden_sizes = [512, 256, 128]
   ```

4. **Overfitting**:
   ```python
   # Add dropout
   dropout_rate = 0.3

   # Add weight decay
   weight_decay = 1e-4

   # Use early stopping
   patience = 15
   ```

This comprehensive ML training guide provides all the necessary information to effectively train, evaluate, and deploy machine learning models for the ES Futures VPOC trading strategy with full ROCm 7.0 optimization.
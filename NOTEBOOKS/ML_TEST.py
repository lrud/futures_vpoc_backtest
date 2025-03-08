#!/opt/conda/envs/py_3.12/bin/python

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.device_mesh import init_device_mesh

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Dynamic path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

# Import project-specific modules
from DATA_LOADER import load_futures_data
from STRATEGY import calculate_volume_profile, find_vpoc
from MATH import VPOCMathAnalysis

class AMDOptimizedFuturesModel(nn.Module):
    """
    Advanced Neural Network Optimized for AMD 7900 XT GPUs
    Leveraging ROCm-specific performance characteristics
    """
    def __init__(self, input_dim, hidden_layers=[256, 128, 64], dropout_rate=0.3):
        super().__init__()
        
        # Input layer with adaptive initialization
        self.input_layer = nn.Linear(input_dim, hidden_layers[0])
        nn.init.kaiming_normal_(self.input_layer.weight)  # Better for SiLU activation
        
        # Dynamic hidden layers with advanced regularization
        layers = []
        prev_dim = hidden_layers[0]
        
        for i, hidden_dim in enumerate(hidden_layers[1:]):
            # Gradually increase dropout for deeper layers
            layer_dropout = dropout_rate * (1 + i * 0.2)  # Increase dropout slightly in deeper layers
            
            layer_block = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.GroupNorm(min(8, hidden_dim), hidden_dim),  # Ensure groups don't exceed features
                nn.SiLU(),  # Efficient activation function
                nn.Dropout(layer_dropout)
            )
            layers.append(layer_block)
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.ModuleList(layers)
        
        # Output layer with precision initialization
        self.output_layer = nn.Linear(hidden_layers[-1], 1)
        nn.init.zeros_(self.output_layer.bias)
        
        # Add layer norm before output for better stability
        self.final_norm = nn.LayerNorm(hidden_layers[-1])

def forward(self, x):
    x = self.input_layer(x)
    x = F.silu(x)  # Apply activation to input layer
    
    # Efficient layer traversal
    for layer in self.hidden_layers:
        x = layer(x)
    
    # Apply final normalization
    x = self.final_norm(x)
    
    return self.output_layer(x)

class AMDFuturesTensorParallel:
    """
    Distributed Machine Learning Predictor for Futures Trading
    Optimized for AMD 7900 XTs with Advanced ROCm Features
    """
    def __init__(self, 
                data_path=None, 
                session_type='RTH', 
                contract='ES',
                target_column='close_change_pct',
                lookback_periods=[10, 20, 50]):
        """
        Initialize predictor with ROCm-specific optimizations
        
        Parameters:
        -----------
        data_path : str, optional
            Path to futures data CSV
        session_type : str, optional
            Trading session type
        contract : str, optional
            Futures contract (default: ES)
        target_column : str, optional
            Column to predict
        lookback_periods : list, optional
            Feature lookback periods
        """
        # Configure ROCm environment
        os.environ['ROCM_HOME'] = '/opt/rocm'
        os.environ['HIP_VISIBLE_DEVICES'] = '0,1'
        os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1100'
        
        # Store parameters
        self.session_type = session_type
        self.contract_prefix = contract
        self.target_column = target_column
        self.lookback_periods = lookback_periods
        
        # GPU and Performance Diagnostics
        self.num_gpus = torch.cuda.device_count()
        self._log_gpu_details()
        
        # Data Loading and Preprocessing
        self.raw_data = self._load_and_preprocess_data(
            data_path or os.path.join(PROJECT_ROOT, 'DATA'),
            session_type,
            contract
        )
        
        # Feature Engineering
        self.features_df = self._generate_features(lookback_periods)
        
        if self.features_df.empty:
            raise ValueError("No valid features were generated. Check data filtering and processing.")
        
        # Print feature statistics
        print("\nFeature Statistics:")
        print(f"Total sessions with complete features: {len(self.features_df)}")
        print(f"Date range: {self.features_df['date'].min()} to {self.features_df['date'].max()}")
        print(f"Feature columns: {[col for col in self.features_df.columns if col != 'date']}")
        
        # Prepare Features and Targets
        feature_columns = [col for col in self.features_df.columns 
                        if col not in ['date', target_column]]
        
        # Store feature columns for later use
        self.feature_columns = feature_columns
        
        # Convert to numpy arrays for training
        self.X = self.features_df[feature_columns].values
        self.y = self.features_df[target_column].values
        
        # Print training data shapes
        print(f"\nTraining data prepared:")
        print(f"X shape: {self.X.shape}")
        print(f"y shape: {self.y.shape}")
    def _log_gpu_details(self):
        """
        Log detailed GPU information for AMD 7900 XTs
        """
        print(f"🔥 ROCm Detected: {self.num_gpus} AMD 7900 XTs")
        for i in range(self.num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Total Memory: {props.total_memory / (1024**3):.2f} GB")
            # Remove or replace the clock_rate property
            # print(f"  Clock Rate: {props.clock_rate / 1e6:.2f} GHz")
            # Alternative properties you can display:
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Multi Processor Count: {props.multi_processor_count}")
    
    def _load_and_preprocess_data(self, data_path, session_type, contract):
        """
        Load data using the existing data_loader module
        """
        # Load data using project's data loader
        data = load_futures_data(data_path)
        
        # Print contract information to debug
        print(f"\nAvailable contracts: {data['contract'].unique()}")
        
        # Identify contracts that match our pattern
        es_contracts = [c for c in data['contract'].unique() if str(c).startswith('ES')]
        print(f"ES contracts found: {es_contracts}")
        
        # Don't filter by contract name, just return the data and do filtering later
        # This lets us work with all the data without making assumptions about contract naming
        return data
    
    def _generate_features(self, lookback_periods):
        """
        Generate comprehensive ML features from futures data
        
        Parameters:
        -----------
        lookback_periods : list
            Periods for feature calculation
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with engineered features
        """
        # Start with session filtering only
        filtered_data = self.raw_data
        print(f"Total data without session filtering: {len(filtered_data)} rows")
        
        # Now use partial matching to include all ES contracts
        es_data = filtered_data[filtered_data['contract'].str.startswith('ES')]
        print(f"ES contract data: {len(es_data)} rows")
        
        if es_data.empty:
            print("⚠️ Warning: No ES contract data found after filtering")
            return pd.DataFrame()
        
        # Ensure date column is in datetime format
        es_data['date'] = pd.to_datetime(es_data['date'])
        
        features_list = []
        
        # Process data by date and generate features
        for date, session_data in es_data.groupby('date'):
            try:
                # Skip dates with insufficient data
                if len(session_data) < 10:
                    continue
                    
                # Volume Profile Analysis
                volume_profile = calculate_volume_profile(session_data)
                vpoc = find_vpoc(volume_profile)
                
                # Calculate session statistics with error handling
                session_high = session_data['high'].max()
                session_low = session_data['low'].min()
                session_open = session_data['open'].iloc[0]
                session_close = session_data['close'].iloc[-1]
                session_volume = session_data['volume'].sum()
                
                # Safely calculate price change percentage
                close_change_pct = (session_close - session_open) / max(session_open, 0.0001) * 100
                
                # Calculate ranges with safety checks
                price_range = max(session_high - session_low, 0.0001)
                range_pct = price_range / max(session_open, 0.0001) * 100
                
                # Create basic feature dictionary
                session_features = {
                    'date': date,
                    'vpoc': float(vpoc),  # Ensure float type
                    'total_volume': float(session_volume),
                    'price_range': float(price_range),
                    'range_pct': float(range_pct),
                    'close_change_pct': float(close_change_pct),
                    'session_high': float(session_high),
                    'session_low': float(session_low),
                    'session_open': float(session_open),
                    'session_close': float(session_close)
                }
                
                # Get average bid/ask spread if available
                if 'spread' in session_data.columns:
                    session_features['avg_spread'] = float(session_data['spread'].mean())
                
                # Add volume-weighted features with safety
                vwap = (session_data['close'] * session_data['volume']).sum() / max(session_data['volume'].sum(), 0.0001)
                session_features['vwap'] = float(vwap)
                session_features['close_to_vwap_pct'] = float((session_close - vwap) / max(vwap, 0.0001) * 100)
                
                features_list.append(session_features)
                
            except Exception as e:
                print(f"⚠️ Error processing session {date}: {e}")
                continue
        
        # Convert list to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Sort by date
        if not features_df.empty:
            features_df = features_df.sort_values('date')
            
            # Add lagged features across sessions
            for period in lookback_periods:
                # Skip if we don't have enough data
                if len(features_df) <= period:
                    continue
                    
                # Price momentum (close-to-close change over period)
                features_df[f'price_mom_{period}d'] = features_df['session_close'].pct_change(period) * 100
                
                # Volatility (std dev of price changes)
                features_df[f'volatility_{period}d'] = features_df['close_change_pct'].rolling(period).std()
                
                # Volume trend
                features_df[f'volume_trend_{period}d'] = features_df['total_volume'].pct_change(period) * 100
                
                # VPOC migration
                features_df[f'vpoc_change_{period}d'] = features_df['vpoc'].diff(period)
                features_df[f'vpoc_pct_change_{period}d'] = features_df['vpoc'].pct_change(period) * 100
                
                # Range evolution
                features_df[f'range_change_{period}d'] = features_df['range_pct'].pct_change(period) * 100
            
            # Extensive cleaning and safety checks
            # Remove infinite values
            features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Drop rows with NaN values
            features_df.dropna(inplace=True)
            
            # Additional diagnostic print
            print("\nFeature Statistics AFTER Cleaning:")
            for col in features_df.columns:
                if col != 'date':
                    print(f"{col}:")
                    print(f"  Mean: {features_df[col].mean()}")
                    print(f"  Min: {features_df[col].min()}")
                    print(f"  Max: {features_df[col].max()}")
                    print(f"  Has NaN: {features_df[col].isna().any()}")
        
        print(f"Generated features for {len(features_df)} sessions")
        
        return features_df
    
    def _augment_with_noise(self, features_df):
        """
        Simple data augmentation by adding noise to existing features
        """
        original_count = len(features_df)
        print(f"Original feature count: {original_count}")
        
        # Create a copy with small random noise
        noise_features = features_df.copy()
        for col in noise_features.columns:
            if col != 'date' and pd.api.types.is_numeric_dtype(noise_features[col]):
                std = noise_features[col].std() * 0.03  # 3% of standard deviation
                noise_features[col] = noise_features[col] + np.random.normal(0, std, len(noise_features))
        
        # Combine original and noisy features
        augmented_df = pd.concat([features_df, noise_features], ignore_index=True)
        print(f"Augmented feature count: {len(augmented_df)}")
        
        return augmented_df
    
    def train_ddp(self, 
                num_epochs=50, 
                batch_size=64, 
                learning_rate=1e-3):
        """
        Launch training with Distributed Data Parallel and ROCm optimizations
        """
        print("[Checkpoint] Starting DDP training")
        
        # Distribute training across available GPUs
        world_size = min(self.num_gpus, 2)
        print(f"[Checkpoint] Using {world_size} GPUs for DDP")
        
        # Set environment variables for the spawned processes
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        
        # Launch distributed training
        print("[Checkpoint] Spawning worker processes")
        mp.spawn(
            self._ddp_worker, 
            args=(
                world_size, 
                self.X, 
                self.y, 
                num_epochs, 
                batch_size, 
                learning_rate
            ), 
            nprocs=world_size
        )
        
        print("[Checkpoint] Training completed")

    def _ddp_worker(self, 
                local_rank, 
                world_size, 
                X, y, 
                num_epochs, 
                batch_size, 
                learning_rate):
        """
        Distributed Data Parallel Training Worker with ROCm Optimizations
        """
        print(f"[Checkpoint] Worker {local_rank}: Starting DDP worker")
        
        try:
            # Initialize distributed environment
            print(f"[Checkpoint] Worker {local_rank}: Initializing process group")
            dist.init_process_group(
                backend='nccl', 
                init_method='env://',
                world_size=world_size, 
                rank=local_rank
            )
            
            # Set the device for this process
            torch.cuda.set_device(local_rank)
            
            print(f"[Checkpoint] Worker {local_rank}: Scaling features")
            # Prepare data with half-precision support
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            print(f"[Checkpoint] Worker {local_rank}: Moving data to GPU")
            # Convert to PyTorch tensors with half-precision option
            X_tensor = torch.tensor(X_scaled, dtype=torch.float16).cuda(local_rank)
            y_tensor = torch.tensor(y, dtype=torch.float16).cuda(local_rank)
            
            print(f"[Checkpoint] Worker {local_rank}: Creating model")
            # Create model and move to GPU
            model = AMDOptimizedFuturesModel(
                input_dim=X_scaled.shape[1]
            ).cuda(local_rank)
            
            print(f"[Checkpoint] Worker {local_rank}: Wrapping model in DDP")
            # Wrap model in DDP
            ddp_model = torch.nn.parallel.DistributedDataParallel(
                model, 
                device_ids=[local_rank],
                output_device=local_rank
            )
            
            # Add diagnostic check to verify model is on correct device
            if local_rank == 0:
                print("[DDP Check] Checking model device")
                for name, param in ddp_model.named_parameters():
                    print(f"Parameter {name} on device: {param.device}")
            
            # Mixed precision training
            scaler = torch.cuda.amp.GradScaler()
            
            print(f"[Checkpoint] Worker {local_rank}: Setting up optimizer")
            # Loss and optimizer with adaptive learning
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(
                ddp_model.parameters(), 
                lr=learning_rate,
                weight_decay=1e-5,
                eps=1e-8
            )
            
            # Learning rate scheduler
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=learning_rate,
                total_steps=num_epochs,
                pct_start=0.3,  # Spend 30% of time warming up
                anneal_strategy='cos'
            )
            print(f"[Checkpoint] Worker {local_rank}: Starting training loop")
            # Training loop with mixed precision
            for epoch in range(num_epochs):
                # Set model to training mode
                ddp_model.train()
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Mixed precision forward and backward passes
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = ddp_model(X_tensor).squeeze()
                    loss = criterion(outputs, y_tensor)
                
                # Scale gradients and perform backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # Update learning rate
                scheduler.step()
                
                # Logging (only on main process)
                if local_rank == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}")
                    print(f"Training Loss: {loss.item():.4f}")
                    
                    # Memory utilization check
                    if epoch % 5 == 0:
                        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
                        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
                        print(f"[Memory] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
                
                # Save checkpoint every 10 epochs
                if epoch % 10 == 0 and local_rank == 0:
                    checkpoint_path = os.path.join(PROJECT_ROOT, "TRAINING", f"es_futures_model_epoch_{epoch}.pt")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': ddp_model.module.state_dict(),  # Save the inner model
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                    }, checkpoint_path)
                    print(f"[Checkpoint] Saved model at epoch {epoch} to {checkpoint_path}")
            
            # Save final model on main process
            if local_rank == 0:
                print("[Checkpoint] Training complete, saving final model")
                final_model_path = os.path.join(PROJECT_ROOT, "TRAINING", "es_futures_model_final.pt")
                torch.save({
                    'model_state_dict': ddp_model.module.state_dict(),  # Save the inner model
                    'feature_columns': self.feature_columns if hasattr(self, 'feature_columns') else None,
                    'scaler': scaler,
                }, final_model_path)
            
        except Exception as e:
            print(f"[ERROR] Worker {local_rank} failed with exception: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            if dist.is_initialized():
                print(f"[Checkpoint] Worker {local_rank}: Cleaning up distributed group")
                dist.destroy_process_group()

def main():
    """
    Main execution for AMD 7900 XT Tensor Parallel Futures ML
    With comprehensive ROCm optimizations
    """
    print("[Checkpoint] Starting main execution")
    
    TRAINING_DIR = os.path.join(PROJECT_ROOT, "TRAINING")
    os.makedirs(TRAINING_DIR, exist_ok=True)
    print(f"[Checkpoint] Created TRAINING directory: {TRAINING_DIR}")
    
    # Initialize predictor
    print("[Checkpoint] Initializing AMDFuturesTensorParallel")
    predictor = AMDFuturesTensorParallel(
        session_type='RTH',
        contract='ES'
    )
    
    # Start training
    print("[Checkpoint] Starting tensor parallel training")
    predictor.train_ddp(
        num_epochs=50,
        batch_size=64,
        learning_rate=1e-3
    )
    
    print("[Checkpoint] Execution complete")

if __name__ == "__main__":
    # Diagnostic information
    print("Python Executable:", sys.executable)
    print("Python Version:", sys.version)
    print("Current Working Directory:", os.getcwd())
    
    # Check torch and GPU availability
    print("\nTorch GPU Check:")
    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Device Count:", torch.cuda.device_count())
    
    main()
"""
Advanced training utilities for stable and robust ML training.
Includes learning rate schedules, noise injection, and monitoring.
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union, Any
from torch.optim.lr_scheduler import _LRScheduler
import warnings
warnings.filterwarnings('ignore')

from ..utils.logging import get_logger


class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by cosine decay.
    Helps prevent training instability in early epochs.
    """
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]


class CyclicalScheduler(_LRScheduler):
    """
    Cyclical learning rate scheduler for better optimization.
    Implements triangular policy as described in Smith (2017).
    """
    def __init__(self, optimizer, base_lr, max_lr, step_size_up, mode='triangular', gamma=1., scale_fn=None, scale_mode='cycle', last_epoch=-1):

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_up
        self.mode = mode
        self.gamma = gamma
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode

        if self.scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2.**(x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        cycle = math.floor(1 + self.last_epoch / (self.step_size_up + self.step_size_down))
        x = 1 + self.last_epoch - cycle * (self.step_size_up + self.step_size_down)

        if x <= self.step_size_up:
            scale_factor = x / self.step_size_up
        else:
            scale_factor = (x - self.step_size_up) / self.step_size_down

        base_lrs = [base_lr for base_lr in self.base_lrs]

        if self.scale_mode == 'cycle':
            scale_factor = scale_factor * self.scale_fn(cycle)
        else:
            scale_factor = scale_factor * self.scale_fn(self.last_epoch)

        return [base_lr + (max_lr - base_lr) * scale_factor for base_lr, max_lr in zip(base_lrs, self.max_lrs)]


class NoiseInjector:
    """
    Injects noise into training process for regularization and robustness.
    """
    def __init__(self, noise_type='gaussian', noise_std=0.01, probability=0.1):
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.probability = probability
        self.logger = get_logger(__name__)

    def inject_input_noise(self, x):
        """Inject noise into input features."""
        if self.training and np.random.random() < self.probability:
            if self.noise_type == 'gaussian':
                noise = torch.randn_like(x) * self.noise_std
                return x + noise
            elif self.noise_type == 'uniform':
                noise = (torch.rand_like(x) - 0.5) * 2 * self.noise_std
                return x + noise
            elif self.noise_type == 'dropout':
                mask = torch.rand_like(x) > self.probability
                return x * mask / (1 - self.probability)
        return x

    def inject_weight_noise(self, model):
        """Inject noise into model weights."""
        if self.training and np.random.random() < self.probability:
            with torch.no_grad():
                for param in model.parameters():
                    if param.requires_grad and len(param.shape) > 1:  # Only for weight matrices
                        if self.noise_type == 'gaussian':
                            noise = torch.randn_like(param) * self.noise_std * 0.01
                            param.add_(noise)
        return model


class GradientClipper:
    """
    Advanced gradient clipping for training stability.
    """
    def __init__(self, clip_type='norm', clip_value=1.0, clip_norm_type=2):
        self.clip_type = clip_type
        self.clip_value = clip_value
        self.clip_norm_type = clip_norm_type
        self.logger = get_logger(__name__)

    def clip_gradients(self, model):
        """Apply gradient clipping to model parameters."""
        if self.clip_type == 'norm':
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_value, norm_type=self.clip_norm_type)
        elif self.clip_type == 'value':
            torch.nn.utils.clip_grad_value_(model.parameters(), self.clip_value)
        elif self.clip_type == 'adaptive':
            self._adaptive_clipping(model)

    def _adaptive_clipping(self, model):
        """Adaptive gradient clipping based on gradient statistics."""
        total_norm = 0
        param_count = 0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(self.clip_norm_type)
                total_norm += param_norm.item() ** self.clip_norm_type
                param_count += 1

        total_norm = total_norm ** (1. / self.clip_norm_type)
        adaptive_clip = self.clip_value * (1 + 0.1 * math.log(param_count + 1))

        if total_norm > adaptive_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), adaptive_clip, norm_type=self.clip_norm_type)
            self.logger.debug(f"Applied adaptive clipping: {total_norm:.4f} -> {adaptive_clip:.4f}")


class TrainingMonitor:
    """
    Comprehensive training monitoring for early detection of issues.
    """
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.loss_history = []
        self.lr_history = []
        self.gradient_norms = []
        self.logger = get_logger(__name__)

    def update(self, loss, lr, gradient_norm=None):
        """Update monitoring metrics."""
        self.loss_history.append(loss)
        self.lr_history.append(lr)
        if gradient_norm is not None:
            self.gradient_norms.append(gradient_norm)

        # Keep only recent history
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)
            self.lr_history.pop(0)
            if len(self.gradient_norms) > self.window_size:
                self.gradient_norms.pop(0)

    def check_training_health(self) -> Dict[str, bool]:
        """Check for training issues and return health status."""
        health_status = {
            'loss_explosion': False,
            'loss_plateau': False,
            'gradient_vanishing': False,
            'gradient_explosion': False,
            'lr_issues': False
        }

        if len(self.loss_history) < 10:
            return health_status

        # Check for loss explosion
        recent_losses = self.loss_history[-10:]
        if len(recent_losses) > 1:
            loss_change = (recent_losses[-1] - recent_losses[0]) / recent_losses[0]
            if loss_change > 1.0:  # 100% increase
                health_status['loss_explosion'] = True
                self.logger.warning(f"⚠️ Loss explosion detected: {loss_change:.2%}")

            # Check for loss plateau
            if abs(loss_change) < 0.01:  # Less than 1% change
                health_status['loss_plateau'] = True
                self.logger.info("ℹ️ Loss plateau detected")

        # Check gradient issues
        if len(self.gradient_norms) > 5:
            recent_grads = self.gradient_norms[-5:]
            avg_grad_norm = np.mean(recent_grads)

            if avg_grad_norm < 1e-6:
                health_status['gradient_vanishing'] = True
                self.logger.warning(f"⚠️ Gradient vanishing detected: {avg_grad_norm:.2e}")

            elif avg_grad_norm > 10.0:
                health_status['gradient_explosion'] = True
                self.logger.warning(f"⚠️ Gradient explosion detected: {avg_grad_norm:.2f}")

        # Check learning rate issues
        if len(self.lr_history) > 5:
            recent_lrs = self.lr_history[-5:]
            if np.std(recent_lrs) / np.mean(recent_lrs) < 0.01:
                health_status['lr_issues'] = True
                self.logger.info("ℹ️ Learning rate may be too stable")

        return health_status

    def get_statistics(self) -> Dict[str, float]:
        """Get current training statistics."""
        if not self.loss_history:
            return {}

        return {
            'current_loss': self.loss_history[-1],
            'avg_loss': np.mean(self.loss_history),
            'loss_std': np.std(self.loss_history),
            'current_lr': self.lr_history[-1] if self.lr_history else 0,
            'avg_grad_norm': np.mean(self.gradient_norms) if self.gradient_norms else 0
        }


class EarlyStopping:
    """
    Advanced early stopping with patience and minimum delta.
    """
    def __init__(self, patience=10, min_delta=1e-6, restore_best_weights=True, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        self.logger = get_logger(__name__)

    def __call__(self, score, model):
        """Check if training should stop."""
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    self.logger.info("Restored best weights")

    def _is_improvement(self, score):
        """Check if the current score is an improvement."""
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta

    def save_checkpoint(self, model):
        """Save model checkpoint."""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()


class AdamWWithWarmup:
    """
    AdamW optimizer with integrated warmup scheduler.
    """
    def __init__(self, model_params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8, warmup_steps=1000):
        self.optimizer = torch.optim.AdamW(
            model_params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps
        )
        self.warmup_steps = warmup_steps
        self.base_lr = lr
        self.current_step = 0
        self.logger = get_logger(__name__)

    def step(self):
        """Perform optimization step with warmup."""
        self.current_step += 1

        # Apply warmup
        if self.current_step <= self.warmup_steps:
            warmup_factor = self.current_step / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.base_lr * warmup_factor

        self.optimizer.step()

    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        """Access optimizer param groups."""
        return self.optimizer.param_groups


class ComprehensiveTrainingMonitor:
    """
    Comprehensive training monitoring system with advanced metrics and alerts.
    """
    def __init__(self, window_size=100, alert_thresholds=None):
        self.window_size = window_size
        self.logger = get_logger(__name__)

        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'loss_explosion_factor': 2.0,      # 2x increase in loss
            'gradient_explosion_limit': 10.0,  # Gradient norm > 10
            'gradient_vanishing_limit': 1e-6,  # Gradient norm < 1e-6
            'loss_nan_tolerance': 0,            # Any NaN in loss
            'accuracy_plateau_epochs': 20,     # No improvement for 20 epochs
            'memory_usage_limit_gb': 14,       # GPU memory > 14GB
            'lr_minimum_limit': 1e-8           # Learning rate too small
        }

        # Monitoring data
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': [],
            'gradient_norms': [],
            'memory_usage': [],
            'batch_times': [],
            'epoch_times': []
        }

        # Alert tracking
        self.alerts = {
            'loss_explosion': [],
            'gradient_explosion': [],
            'gradient_vanishing': [],
            'nan_values': [],
            'memory_issues': [],
            'convergence_issues': []
        }

        # Stability metrics
        self.stability_metrics = {
            'loss_variance': [],
            'gradient_variance': [],
            'loss_trend': [],
            'training_efficiency': []
        }

        # Health status
        self.health_status = {
            'overall_health': 'good',
            'last_check': None,
            'issues_detected': [],
            'recommendations': []
        }

    def update_metrics(self, metrics: Dict[str, float]):
        """Update monitoring metrics."""
        timestamp = len(self.metrics_history['train_loss'])

        # Update metric histories
        for metric_name, value in metrics.items():
            if metric_name in self.metrics_history:
                self.metrics_history[metric_name].append(value)
                # Keep only recent history
                if len(self.metrics_history[metric_name]) > self.window_size:
                    self.metrics_history[metric_name].pop(0)

        # Calculate stability metrics
        self._calculate_stability_metrics()

        # Check for alerts
        self._check_alerts(metrics)

        # Update health status
        self._update_health_status()

    def _calculate_stability_metrics(self):
        """Calculate stability and convergence metrics."""
        if len(self.metrics_history['train_loss']) < 10:
            return

        # Loss variance (stability measure)
        recent_losses = self.metrics_history['train_loss'][-20:]
        loss_variance = np.var(recent_losses)
        self.stability_metrics['loss_variance'].append(loss_variance)

        # Gradient variance
        if len(self.metrics_history['gradient_norms']) >= 10:
            recent_grads = self.metrics_history['gradient_norms'][-20:]
            grad_variance = np.var(recent_grads)
            self.stability_metrics['gradient_variance'].append(grad_variance)

        # Loss trend (convergence measure)
        if len(recent_losses) >= 10:
            x = np.arange(len(recent_losses))
            slope, _, _, _, _ = stats.linregress(x, recent_losses)
            self.stability_metrics['loss_trend'].append(slope)

        # Training efficiency (loss reduction per time)
        if len(self.metrics_history['epoch_times']) >= 2:
            recent_times = self.metrics_history['epoch_times'][-5:]
            recent_losses = self.metrics_history['train_loss'][-5:]
            if len(recent_times) > 1 and len(recent_losses) > 1:
                time_efficiency = abs(recent_losses[-1] - recent_losses[0]) / sum(recent_times)
                self.stability_metrics['training_efficiency'].append(time_efficiency)

    def _check_alerts(self, current_metrics: Dict[str, float]):
        """Check for training issues and generate alerts."""
        current_time = len(self.metrics_history['train_loss'])

        # Loss explosion check
        if len(self.metrics_history['train_loss']) >= 10:
            recent_losses = self.metrics_history['train_loss'][-10:]
            avg_recent = np.mean(recent_losses[:-5])  # Average of older 5
            current_loss = np.mean(recent_losses[-5:])   # Average of recent 5

            if avg_recent > 0 and current_loss > avg_recent * self.alert_thresholds['loss_explosion_factor']:
                alert = {
                    'timestamp': current_time,
                    'severity': 'high',
                    'message': f"Loss explosion detected: {current_loss:.4f} vs {avg_recent:.4f} (factor {current_loss/avg_recent:.2f})"
                }
                self.alerts['loss_explosion'].append(alert)
                self.logger.warning(f"🚨 ALERT: {alert['message']}")

        # Gradient explosion/vanishing check
        if 'gradient_norm' in current_metrics:
            grad_norm = current_metrics['gradient_norm']

            if grad_norm > self.alert_thresholds['gradient_explosion_limit']:
                alert = {
                    'timestamp': current_time,
                    'severity': 'high',
                    'message': f"Gradient explosion: {grad_norm:.4f}"
                }
                self.alerts['gradient_explosion'].append(alert)
                self.logger.warning(f"🚨 ALERT: {alert['message']}")

            elif grad_norm < self.alert_thresholds['gradient_vanishing_limit']:
                alert = {
                    'timestamp': current_time,
                    'severity': 'medium',
                    'message': f"Gradient vanishing: {grad_norm:.2e}"
                }
                self.alerts['gradient_vanishing'].append(alert)
                self.logger.warning(f"⚠️ ALERT: {alert['message']}")

        # NaN/Inf values check
        for metric_name, value in current_metrics.items():
            if np.isnan(value) or np.isinf(value):
                alert = {
                    'timestamp': current_time,
                    'severity': 'critical',
                    'message': f"NaN/Inf detected in {metric_name}: {value}"
                }
                self.alerts['nan_values'].append(alert)
                self.logger.error(f"💀 CRITICAL ALERT: {alert['message']}")

        # Memory usage check
        if 'memory_usage_gb' in current_metrics:
            memory_usage = current_metrics['memory_usage_gb']
            if memory_usage > self.alert_thresholds['memory_usage_limit_gb']:
                alert = {
                    'timestamp': current_time,
                    'severity': 'high',
                    'message': f"High memory usage: {memory_usage:.2f}GB"
                }
                self.alerts['memory_issues'].append(alert)
                self.logger.warning(f"🚨 ALERT: {alert['message']}")

        # Learning rate check
        if 'learning_rate' in current_metrics:
            lr = current_metrics['learning_rate']
            if lr < self.alert_thresholds['lr_minimum_limit']:
                alert = {
                    'timestamp': current_time,
                    'severity': 'medium',
                    'message': f"Learning rate too small: {lr:.2e}"
                }
                self.alerts['convergence_issues'].append(alert)
                self.logger.warning(f"⚠️ ALERT: {alert['message']}")

    def _update_health_status(self):
        """Update overall training health status."""
        current_time = len(self.metrics_history['train_loss'])

        # Count recent alerts
        recent_alerts = 0
        critical_issues = []

        for alert_type, alert_list in self.alerts.items():
            recent_alerts += len([a for a in alert_list if current_time - a['timestamp'] < 50])
            if alert_list and alert_list[-1]['severity'] == 'critical':
                critical_issues.append(alert_type)

        # Determine overall health
        if critical_issues:
            self.health_status['overall_health'] = 'critical'
        elif recent_alerts > 10:
            self.health_status['overall_health'] = 'poor'
        elif recent_alerts > 5:
            self.health_status['overall_health'] = 'fair'
        elif recent_alerts > 0:
            self.health_status['overall_health'] = 'good'
        else:
            self.health_status['overall_health'] = 'excellent'

        self.health_status['last_check'] = current_time
        self.health_status['issues_detected'] = [f"{k}: {len(v)}" for k, v in self.alerts.items() if v]

        # Generate recommendations
        self.health_status['recommendations'] = self._generate_health_recommendations()

    def _generate_health_recommendations(self) -> List[str]:
        """Generate health-based recommendations."""
        recommendations = []

        # Based on alerts
        if self.alerts['loss_explosion']:
            recommendations.append("Reduce learning rate or add gradient clipping")

        if self.alerts['gradient_explosion']:
            recommendations.append("Implement stronger gradient clipping")

        if self.alerts['gradient_vanishing']:
            recommendations.append("Check for dead neurons or reduce regularization")

        if self.alerts['nan_values']:
            recommendations.append("Check for numerical instability in loss computation")

        if self.alerts['memory_issues']:
            recommendations.append("Reduce batch size or enable gradient checkpointing")

        if self.alerts['convergence_issues']:
            recommendations.append("Adjust learning rate schedule or check model architecture")

        # Based on stability metrics
        if self.stability_metrics['loss_variance']:
            recent_var = self.stability_metrics['loss_variance'][-1]
            if recent_var > 0.1:
                recommendations.append("High loss variance - consider more conservative parameters")

        if self.stability_metrics['loss_trend']:
            recent_trend = self.stability_metrics['loss_trend'][-1]
            if recent_trend > 0:
                recommendations.append("Loss increasing - consider reducing learning rate")

        if not recommendations:
            recommendations.append("Training appears stable - continue monitoring")

        return recommendations

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive training monitoring report."""
        current_time = len(self.metrics_history['train_loss'])

        report = {
            'timestamp': current_time,
            'health_status': self.health_status.copy(),
            'recent_metrics': {},
            'stability_summary': {},
            'alerts_summary': {},
            'recommendations': self.health_status['recommendations'].copy()
        }

        # Recent metrics
        for metric_name, history in self.metrics_history.items():
            if history:
                report['recent_metrics'][metric_name] = {
                    'current': history[-1],
                    'average': np.mean(history[-10:]) if len(history) >= 10 else np.mean(history),
                    'trend': 'stable' if len(history) < 10 else 'improving' if history[-1] < np.mean(history[-10:]) else 'degrading'
                }

        # Stability summary
        for metric_name, history in self.stability_metrics.items():
            if history:
                report['stability_summary'][metric_name] = {
                    'current': history[-1],
                    'average': np.mean(history[-10:]) if len(history) >= 10 else np.mean(history)
                }

        # Alerts summary
        for alert_type, alert_list in self.alerts.items():
            report['alerts_summary'][alert_type] = {
                'total_count': len(alert_list),
                'recent_count': len([a for a in alert_list if current_time - a['timestamp'] < 50])
            }
            if alert_list:
                report['alerts_summary'][alert_type]['latest'] = alert_list[-1]

        return report

    def log_comprehensive_status(self):
        """Log comprehensive training status."""
        report = self.get_comprehensive_report()

        self.logger.info("=" * 80)
        self.logger.info("🏥 COMPREHENSIVE TRAINING HEALTH REPORT")
        self.logger.info("=" * 80)

        # Overall health
        health = report['health_status']
        health_emoji = {
            'excellent': '🟢', 'good': '🟡', 'fair': '🟠', 'poor': '🔴', 'critical': '💀'
        }
        self.logger.info(f"Overall Health: {health_emoji.get(health['overall_health'], '❓')} {health['overall_health'].upper()}")

        # Recent metrics
        self.logger.info("📊 Recent Metrics:")
        for metric_name, metric_data in report['recent_metrics'].items():
            if metric_data:
                trend_emoji = {'improving': '📈', 'stable': '➡️', 'degrading': '📉'}
                self.logger.info(f"  • {metric_name}: {metric_data['current']:.6f} ({trend_emoji.get(metric_data['trend'], '❓')} {metric_data['trend']})")

        # Alerts summary
        total_recent_alerts = sum(alert_data['recent_count'] for alert_data in report['alerts_summary'].values())
        if total_recent_alerts > 0:
            self.logger.info(f"🚨 Recent Alerts: {total_recent_alerts}")
            for alert_type, alert_data in report['alerts_summary'].items():
                if alert_data['recent_count'] > 0:
                    self.logger.info(f"  • {alert_type}: {alert_data['recent_count']} recent")

        # Recommendations
        if report['recommendations']:
            self.logger.info("💡 Recommendations:")
            for i, rec in enumerate(report['recommendations'], 1):
                self.logger.info(f"  {i}. {rec}")

        self.logger.info("=" * 80)

    def save_training_log(self, filepath: str):
        """Save comprehensive training log to file."""
        report = self.get_comprehensive_report()

        try:
            import json
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"📝 Training log saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save training log: {e}")

    def create_training_dashboard(self, save_path: Optional[str] = None) -> Optional[str]:
        """Create comprehensive training dashboard visualization."""
        try:
            fig, axes = plt.subplots(3, 2, figsize=(16, 12))
            fig.suptitle('Comprehensive Training Dashboard', fontsize=16)

            # Training and validation loss
            if self.metrics_history['train_loss']:
                axes[0, 0].plot(self.metrics_history['train_loss'], label='Train Loss', alpha=0.7)
                if self.metrics_history['val_loss']:
                    axes[0, 0].plot(self.metrics_history['val_loss'], label='Val Loss', alpha=0.7)
                axes[0, 0].set_title('Loss Progression')
                axes[0, 0].set_xlabel('Iteration')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)

            # Learning rate schedule
            if self.metrics_history['learning_rate']:
                axes[0, 1].plot(self.metrics_history['learning_rate'])
                axes[0, 1].set_title('Learning Rate Schedule')
                axes[0, 1].set_xlabel('Iteration')
                axes[0, 1].set_ylabel('Learning Rate')
                axes[0, 1].set_yscale('log')
                axes[0, 1].grid(True, alpha=0.3)

            # Gradient norms
            if self.metrics_history['gradient_norms']:
                axes[1, 0].plot(self.metrics_history['gradient_norms'])
                axes[1, 0].axhline(y=self.alert_thresholds['gradient_explosion_limit'], color='r', linestyle='--', alpha=0.7, label='Explosion Limit')
                axes[1, 0].axhline(y=self.alert_thresholds['gradient_vanishing_limit'], color='orange', linestyle='--', alpha=0.7, label='Vanishing Limit')
                axes[1, 0].set_title('Gradient Norms')
                axes[1, 0].set_xlabel('Iteration')
                axes[1, 0].set_ylabel('Gradient Norm')
                axes[1, 0].set_yscale('log')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

            # Validation accuracy
            if self.metrics_history['val_accuracy']:
                axes[1, 1].plot(self.metrics_history['val_accuracy'])
                axes[1, 1].set_title('Validation Accuracy')
                axes[1, 1].set_xlabel('Iteration')
                axes[1, 1].set_ylabel('Accuracy (%)')
                axes[1, 1].grid(True, alpha=0.3)

            # Stability metrics
            if self.stability_metrics['loss_variance']:
                axes[2, 0].plot(self.stability_metrics['loss_variance'])
                axes[2, 0].set_title('Loss Variance (Stability)')
                axes[2, 0].set_xlabel('Iteration')
                axes[2, 0].set_ylabel('Variance')
                axes[2, 0].grid(True, alpha=0.3)

            # Training efficiency
            if self.stability_metrics['training_efficiency']:
                axes[2, 1].plot(self.stability_metrics['training_efficiency'])
                axes[2, 1].set_title('Training Efficiency')
                axes[2, 1].set_xlabel('Iteration')
                axes[2, 1].set_ylabel('Loss Reduction / Time')
                axes[2, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"📊 Training dashboard saved to {save_path}")
                plt.close()
                return save_path
            else:
                plt.show()
                plt.close()
                return None

        except Exception as e:
            self.logger.error(f"Failed to create training dashboard: {e}")
            return None
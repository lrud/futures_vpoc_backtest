#!/bin/bash
# Safe Training Wrapper with Memory Protection

set -e

# Configuration
MAX_MEMORY_MB=20000  # Maximum memory per GPU
CHECK_INTERVAL=10    # Check memory every N seconds
CLEANUP_ATTEMPTS=3   # Maximum cleanup attempts

LOG_DIR="/workspace/logs"
TRAINING_SCRIPT="/workspace/src/ml/train.py"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Function to log messages
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_DIR/safe_train.log"
}

# Function to check GPU memory before training
check_pre_training_memory() {
    log "🔍 Checking pre-training GPU memory status..."

    # Get current memory usage
    memory_info=$(rocm-smi --showmeminfo vram --csv)

    while IFS= read -r line; do
        if [[ $line == GPU* ]]; then
            gpu_id=$(echo "$line" | cut -d',' -f1)
            used_mb=$(echo "$line" | cut -d',' -f2)
            total_mb=$(echo "$line" | cut -d',' -f3)
            usage_percent=$(echo "$line" | cut -d',' -f4 | sed 's/%//g')

            log "GPU $gpu_id: $used_mb / $total_mb MB (${usage_percent}% used)"

            # Check if memory usage is too high
            if (( $(echo "$usage_percent > 20" | bc -l) )); then
                log "⚠️  WARNING: GPU $gpu_id has high memory usage (${usage_percent}%)"

                # Try to cleanup first
                attempt_cleanup

                # Check again
                new_usage=$(rocm-smi --showmeminfo vram --csv | grep "$gpu_id" | cut -d',' -f4 | sed 's/%//g')

                if (( $(echo "$new_usage > 20" | bc -l) )); then
                    log "❌ ERROR: GPU memory still too high after cleanup (${new_usage}%)"
                    log "❌ Please reboot the system or check for zombie processes"
                    exit 1
                fi
            fi
        fi
    done <<< "$memory_info"
}

# Function to attempt memory cleanup
attempt_cleanup() {
    log "🧹 Attempting GPU memory cleanup..."

    # Kill any remaining training processes
    pkill -f "$TRAINING_SCRIPT" 2>/dev/null || true
    pkill -f "python.*train" 2>/dev/null || true

    # Clear ROCm caches
    sleep 2

    # Check for orphaned processes
    orphaned=$(rocm-smi --showall | grep "PID.*UNKNOWN" | awk '{print $3}' | sort -u)
    if [ -n "$orphaned" ]; then
        log "Found orphaned KFD processes: $orphaned"
        for pid in $orphaned; do
            kill -9 "$pid" 2>/dev/null || true
        done
    fi

    # Wait for cleanup
    sleep 5
}

# Function to monitor memory during training
monitor_training_memory() {
    local training_pid=$1

    log "👀 Starting memory monitoring for training PID: $training_pid"

    while kill -0 "$training_pid" 2>/dev/null; do
        # Check memory usage
        memory_info=$(rocm-smi --showmeminfo vram --csv)

        while IFS= read -r line; do
            if [[ $line == GPU* ]]; then
                gpu_id=$(echo "$line" | cut -d',' -f1)
                used_mb=$(echo "$line" | cut -d',' -f2)
                total_mb=$(echo "$line" | cut -d',' -f3)
                usage_percent=$(echo "$line" | cut -d',' -f4 | sed 's/%//g')

                # Alert if memory usage is getting high
                if (( $(echo "$usage_percent > 90" | bc -l) )); then
                    log "⚠️  HIGH MEMORY ALERT: GPU $gpu_id at ${usage_percent}% (${used_mb}MB)"

                    # If we get too close to max, kill training to prevent corruption
                    if (( $(echo "$usage_percent > 98" | bc -l) )); then
                        log "🚨 CRITICAL: GPU memory at ${usage_percent}% - killing training to prevent corruption"
                        kill -TERM "$training_pid" 2>/dev/null || true
                        sleep 5
                        kill -KILL "$training_pid" 2>/dev/null || true
                        return 1
                    fi
                fi
            fi
        done <<< "$memory_info"

        sleep "$CHECK_INTERVAL"
    done

    log "✅ Training process completed or terminated"
    return 0
}

# Function to post-cleanup after training
post_training_cleanup() {
    log "🧹 Performing post-training cleanup..."

    # Kill any remaining processes
    pkill -f "$TRAINING_SCRIPT" 2>/dev/null || true

    # Clear caches
    if command -v python3 &> /dev/null; then
        python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print('GPU cache cleared')
" 2>/dev/null || true
    fi

    log "✅ Post-training cleanup completed"
}

# Main execution
main() {
    log "🚀 Starting safe training wrapper..."

    # Pre-training memory check
    check_pre_training_memory

    log "✅ Pre-training memory check passed"

    # Start memory monitor in background
    "$WORKSPACE/scripts/gpu_memory_monitor.sh" monitor &
    local monitor_pid=$!

    # Start training
    log "🎯 Starting training: $TRAINING_SCRIPT $*"

    # Start training and capture PID
    PYTHONPATH=/workspace python "$TRAINING_SCRIPT" "$@" &
    local training_pid=$!

    log "Training started with PID: $training_pid"

    # Monitor memory during training
    if monitor_training_memory "$training_pid"; then
        # Training completed successfully
        log "✅ Training completed successfully"

        # Wait for training to finish and get exit code
        wait "$training_pid"
        local exit_code=$?

        if [ $exit_code -eq 0 ]; then
            log "🎉 Training finished successfully!"
        else
            log "❌ Training failed with exit code: $exit_code"
        fi
    else
        # Training was killed due to memory issues
        log "❌ Training was terminated due to memory constraints"

        # Kill the training process if still running
        kill -TERM "$training_pid" 2>/dev/null || true
        sleep 2
        kill -KILL "$training_pid" 2>/dev/null || true

        wait "$training_pid" 2>/dev/null || true
    fi

    # Kill memory monitor
    kill "$monitor_pid" 2>/dev/null || true

    # Post-training cleanup
    post_training_cleanup

    log "🏁 Safe training wrapper completed"
}

# Run main function with all arguments
main "$@"
#!/bin/bash
# GPU Memory Monitor and Cleanup Script
# Prevents GPU memory leaks from orphaned processes

set -e

LOG_FILE="/workspace/logs/gpu_monitor.log"
ALERT_THRESHOLD=90  # Alert when GPU memory usage > 90%
KILL_THRESHOLD=95   # Kill processes when GPU memory usage > 95%

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Function to check GPU memory usage
check_gpu_memory() {
    log_message "Checking GPU memory usage..."

    # Get memory usage percentages
    memory_usage=$(rocm-smi --showmeminfo vram --csv | grep -A1 "GPU" | tail -n +2 | awk -F',' '{print $3}' | sed 's/%//g')

    gpu_id=0
    for usage in $memory_usage; do
        if (( $(echo "$usage > $ALERT_THRESHOLD" | bc -l) )); then
            log_message "⚠️  ALERT: GPU $gpu_id memory usage is ${usage}% (threshold: ${ALERT_THRESHOLD}%)"

            if (( $(echo "$usage > $KILL_THRESHOLD" | bc -l) )); then
                log_message "🚨 CRITICAL: GPU $gpu0 memory usage is ${usage}% - initiating cleanup..."
                cleanup_gpu_memory
            fi
        fi
        gpu_id=$((gpu_id + 1))
    done
}

# Function to cleanup GPU memory
cleanup_gpu_memory() {
    log_message "🧹 Starting GPU memory cleanup..."

    # Find and kill Python processes with GPU memory
    python_processes=$(rocm-smi --showpids --csv | grep -i "python" | awk -F',' '{print $2}' | head -5)

    if [ -n "$python_processes" ]; then
        log_message "Found Python processes with GPU memory: $python_processes"
        for pid in $python_processes; do
            if kill -0 "$pid" 2>/dev/null; then
                log_message "Killing process $pid..."
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
    fi

    # Check for orphaned KFD processes
    kfd_processes=$(rocm-smi --showall | grep "PID.*UNKNOWN" | awk '{print $3}' | sort -u)

    if [ -n "$kfd_processes" ]; then
        log_message "🔥 Found orphaned KFD processes: $kfd_processes"
        for pid in $kfd_processes; do
            # Try to kill even if they don't show in ps
            kill -9 "$pid" 2>/dev/null || true
        done

        # Force GPU reset if processes persist
        log_message "Attempting GPU reset..."
        echo 1 | sudo tee /sys/class/drm/card0/device/reset 2>/dev/null || true
        echo 1 | sudo tee /sys/class/drm/card1/device/reset 2>/dev/null || true
    fi

    # Clear ROCm memory cache
    log_message "Clearing ROCm memory caches..."
    sleep 2

    # Check if memory was freed
    sleep 5
    new_usage=$(rocm-smi --showmeminfo vram --csv | grep -A1 "GPU" | tail -n +2 | awk -F',' '{print $3}' | sed 's/%//g')

    log_message "Memory usage after cleanup:"
    gpu_id=0
    for usage in $new_usage; do
        log_message "  GPU $gpu_id: ${usage}%"
        gpu_id=$((gpu_id + 1))
    done
}

# Function to monitor continuously
monitor_continuously() {
    log_message "🔄 Starting continuous GPU memory monitoring..."

    while true; do
        check_gpu_memory
        sleep 30  # Check every 30 seconds
    done
}

# Main execution
case "${1:-check}" in
    "monitor")
        monitor_continuously
        ;;
    "cleanup")
        cleanup_gpu_memory
        ;;
    "check"|*)
        check_gpu_memory
        ;;
esac
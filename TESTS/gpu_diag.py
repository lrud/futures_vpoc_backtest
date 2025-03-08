#!/opt/conda/envs/py_3.12/bin/python

import sys
import subprocess
import platform

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return "", str(e)

def comprehensive_gpu_diagnostic():
    print("=== Comprehensive GPU and System Diagnostic ===")
    
    # System Information
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Kernel Version: {platform.release()}")
    
    # ROCm and GPU Diagnostics
    diagnostic_commands = [
        ("ROCm Version", "rocm-smi --showproduct"),
        ("ROCm Agent Enumerator", "/opt/rocm/bin/rocm_agent_enumerator"),
        ("Offload Arch", "/opt/rocm/bin/offload-arch"),
        ("AMD GPU Kernel Modules", "lsmod | grep -E 'amdgpu|kfd'"),
        ("Kernel AMD GPU Drivers", "dkms status | grep amdgpu")
    ]
    
    for name, command in diagnostic_commands:
        print(f"\n=== {name} ===")
        output, error = run_command(command)
        print(output if output else "No output")
        if error:
            print(f"Error: {error}")
    
    # Detailed ROCm Info
    print("\n=== Detailed ROCm Information ===")
    try:
        import subprocess
        rocminfo_output = subprocess.check_output(["/opt/rocm/bin/rocminfo"], 
                                                  universal_newlines=True)
        # Print only the first few lines and last few lines to avoid overwhelming output
        print("First 20 lines of rocminfo:")
        print('\n'.join(rocminfo_output.split('\n')[:20]))
        print("\n... (truncated) ...\n")
        print("Last 20 lines of rocminfo:")
        print('\n'.join(rocminfo_output.split('\n')[-20:]))
    except Exception as e:
        print(f"Error running rocminfo: {e}")
    
    # Python and PyTorch Diagnostic
    print("\n=== Python and PyTorch Diagnostic ===")
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"ROCm Version: {torch.version.hip}")
        print(f"CUDA/ROCm Available: {torch.cuda.is_available()}")
        print(f"Device Count: {torch.cuda.device_count()}")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"\nDevice {i}:")
                print(f"  Name: {torch.cuda.get_device_name(i)}")
                print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    except Exception as e:
        print(f"PyTorch Diagnostic Error: {e}")

if __name__ == "__main__":
    comprehensive_gpu_diagnostic()
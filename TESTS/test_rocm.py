import sys
import platform
import subprocess
import os

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return "", str(e)

def comprehensive_rocm_diagnostic():
    print("=== Comprehensive ROCm and PyTorch Diagnostic ===")
    
    # System Information
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    # ROCm Installation Checks
    rocm_checks = [
        ("ROCm Installation Path", "ls /opt/rocm"),
        ("ROCm Binary Locations", "ls /opt/rocm/bin"),
        ("ROCm Agent Enumerator", "/opt/rocm/bin/rocm_agent_enumerator"),
        ("ROCm Info", "/opt/rocm/bin/rocminfo")
    ]
    
    print("\n=== ROCm Installation Verification ===")
    for name, command in rocm_checks:
        output, error = run_command(command)
        print(f"{name}:")
        if output:
            print(f" ✓ Output: {output}")
        elif error:
            print(f" ⚠ Error: {error}")
        else:
            print(" ❌ No output")
    
    # ROCm SMI Check
    print("\n=== ROCm System Management Interface (SMI) ===")
    smi_output, smi_error = run_command("rocm-smi")
    print(smi_output if smi_output else f"Error: {smi_error}")
    
    # Environment Variables
    print("\n=== ROCm Environment Variables ===")
    rocm_vars = [
        "ROCM_PATH",
        "HSA_OVERRIDE_GFX_VERSION",
        "PYTORCH_ROCM_ARCH",
        "HIP_VISIBLE_DEVICES",
        "PATH",
        "LD_LIBRARY_PATH"
    ]
    
    for var in rocm_vars:
        value = os.environ.get(var, "Not Set")
        print(f"{var}: {value}")
    
    # Pip Package Verification
    print("\n=== Installed Packages ===")
    packages_to_check = [
        "torch", "torchvision", "torchaudio",
        "rocm-smi", "hip-runtime-amd"
    ]
    
    for package in packages_to_check:
        output, error = run_command(f"pip3 show {package}")
        print(f"\n{package}:")
        print(output if output else " Not installed")
    
    # PyTorch Diagnostic
    print("\n=== PyTorch GPU Diagnostic ===")
    diagnostic_code = """
import sys
import torch
print(f"Python Path: {sys.executable}")
print(f"PyTorch Version: {torch.__version__}")
print(f"Torch Compiled with ROCm: {torch.version.hip}")
print(f"CUDA/ROCm Available: {torch.cuda.is_available()}")
print(f"Device Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"\\nDevice {i}:")
        print(f" Name: {torch.cuda.get_device_name(i)}")
        print(f" Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
"""
    
    # Run the diagnostic code
    try:
        exec(diagnostic_code)
    except Exception as e:
        print(f"Error during PyTorch diagnostic: {e}")

if __name__ == "__main__":
    comprehensive_rocm_diagnostic()
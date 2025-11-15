#!/usr/bin/env python3
"""
GPU Monitoring Script - Run this BEFORE starting parallel_train.py
This will verify your GPUs are detected and ready for parallel training.
"""

import torch
import multiprocessing

print("="*60)
print("GPU VERIFICATION SCRIPT")
print("="*60)

# Check CUDA availability
print(f"\n1. CUDA Available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("   ❌ ERROR: CUDA not available!")
    print("   Make sure you're running on a GPU instance with CUDA drivers.")
    exit(1)

# Count GPUs
n_gpus = torch.cuda.device_count()
print(f"2. Number of GPUs detected: {n_gpus}")

if n_gpus == 0:
    print("   ❌ ERROR: No GPUs detected!")
    exit(1)
elif n_gpus == 4:
    print("   ✅ Perfect! 4 GPUs detected (g4dn.12xlarge)")
else:
    print(f"   ⚠️  Found {n_gpus} GPU(s) - expected 4 for g4dn.12xlarge")

# Show GPU details
print("\n3. GPU Details:")
for i in range(n_gpus):
    props = torch.cuda.get_device_properties(i)
    memory_gb = props.total_memory / (1024**3)
    print(f"   GPU {i}: {props.name}")
    print(f"      Total Memory: {memory_gb:.2f} GB")
    print(f"      Compute Capability: {props.major}.{props.minor}")

# Check CPU count for parallel processes
n_cpus = multiprocessing.cpu_count()
print(f"\n4. CPUs Available: {n_cpus}")

# Test GPU memory allocation
print("\n5. Testing GPU memory allocation...")
try:
    for i in range(n_gpus):
        torch.cuda.set_device(i)
        test_tensor = torch.randn(1000, 1000).cuda()
        del test_tensor
        torch.cuda.empty_cache()
        print(f"   ✅ GPU {i}: Memory allocation test passed")
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    exit(1)

# Show memory availability
print("\n6. GPU Memory Status:")
for i in range(n_gpus):
    free, total = torch.cuda.mem_get_info(i)
    free_gb = free / (1024**3)
    total_gb = total / (1024**3)
    print(f"   GPU {i}: {free_gb:.2f} GB free / {total_gb:.2f} GB total")

print("\n" + "="*60)
print("✅ ALL CHECKS PASSED - Ready for parallel training!")
print("="*60)
print("\nNext steps:")
print("  1. Run: python parallel_train.py")
print("  2. In another terminal, monitor with: watch -n 1 nvidia-smi")
print("  3. You should see all 4 GPUs at ~100% utilization")
print("="*60)

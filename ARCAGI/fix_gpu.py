import os
import re

def fix_file(filepath, replacements):
    """Apply replacements to a file"""
    if not os.path.exists(filepath):
        print(f"⚠️  File not found: {filepath}")
        return
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    for old, new in replacements:
        content = content.replace(old, new)
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Fixed {filepath}")
    else:
        print(f"- No changes needed in {filepath}")

print("🚀 Undoing CPU fixes and enabling GPU for AWS deployment...\n")

# Fix arc_compressor.py
fix_file('arc_compressor.py', [
    ("torch.set_default_device('cpu')", "torch.set_default_device('cuda')"),
])

# Fix preprocessing.py
fix_file('preprocessing.py', [
    (".to('cpu')", ".to(torch.get_default_device())"),
])

# Fix train.py
fix_file('train.py', [
    ("torch.set_default_device('cpu')", "torch.set_default_device('cuda')"),
])

# Fix solve_task.py
fix_file('solve_task.py', [
    ("torch.set_default_device('cpu')", "torch.set_default_device('cuda')"),
    ("# torch.cuda.set_device(gpu_id)  # Disabled for CPU", "torch.cuda.set_device(gpu_id)"),
    ("# torch.cuda.reset_peak_memory_stats()  # Disabled for CPU", "torch.cuda.reset_peak_memory_stats()"),
    ("# torch.cuda.empty_cache()  # Disabled for CPU", "torch.cuda.empty_cache()"),
    ("memory_dict[task_name] = 0  # CPU mode", "memory_dict[task_name] = torch.cuda.max_memory_allocated()"),
])

print("\n✅ All files fixed for GPU!")
print("\n🎯 Next steps:")
print("  1. Verify GPUs: python -c 'import torch; print(f\"GPUs available: {torch.cuda.device_count()}\")'")
print("  2. Run training: python parallel_train.py")
print("\n💡 Note: parallel_train.py is already GPU-ready and will use all available GPUs!")
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

# Fix preprocessing.py
fix_file('preprocessing.py', [
    (".to(torch.get_default_device())", ".to('cpu')"),
])

# Fix train.py - add after torch.manual_seed(0)
with open('train.py', 'r', encoding='utf-8') as f:
    content = f.read()

if "torch.set_default_device('cpu')" not in content:
    # Find the line after torch.manual_seed(0)
    content = content.replace(
        "torch.manual_seed(0)",
        "torch.manual_seed(0)\ntorch.set_default_device('cpu')"
    )
    with open('train.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✓ Fixed train.py")
else:
    print("- No changes needed in train.py")

# Fix arc_compressor.py
fix_file('arc_compressor.py', [
    ("torch.set_default_device('cuda')", "torch.set_default_device('cpu')"),
])

# Fix analyze_example.py
fix_file('analyze_example.py', [
    ("torch.set_default_device('cuda')", "torch.set_default_device('cpu')"),
])

# Fix solve_task.py
fix_file('solve_task.py', [
    ("torch.set_default_device('cuda')", "torch.set_default_device('cpu')"),
    ("torch.cuda.set_device(gpu_id)", "# torch.cuda.set_device(gpu_id)  # Disabled for CPU"),
    ("torch.cuda.reset_peak_memory_stats()", "# torch.cuda.reset_peak_memory_stats()  # Disabled for CPU"),
    ("torch.cuda.max_memory_allocated()", "0  # CPU mode"),
    ("torch.cuda.empty_cache()", "# torch.cuda.empty_cache()  # Disabled for CPU"),
])

print("\n✅ All files fixed for CPU!")
print("\nNow you can run:")
print("  python train.py")
print("  python analyze_example.py")
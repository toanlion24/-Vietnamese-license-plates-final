# DEBUG SCRIPT - Kiểm tra Dataset Structure
# ========================================

# 1. Xem thực tế folder structure
# ================================
import os
from pathlib import Path

print("=" * 50)
print("DEBUG: Dataset Structure")
print("=" * 50)

# Check /content folder
print("\n[1] /content folder:")
for item in os.listdir('/content'):
    print(f"   - {item}")

# Check /content/dataset folder
print("\n[2] /content/dataset folder:")
if os.path.exists('/content/dataset'):
    for item in os.listdir('/content/dataset'):
        print(f"   - {item}")
else:
    print("   [X] /content/dataset NOT FOUND")

# Check all folders recursively
print("\n[3] All folders in /content:")
for root, dirs, files in os.walk('/content'):
    level = root.replace('/content', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    if level < 3:  # Only show 3 levels deep
        subindent = ' ' * 2 * (level + 1)
        for file in files[:3]:  # Show first 3 files
            print(f'{subindent}{file}')
        if len(files) > 3:
            print(f'{subindent}... ({len(files)} files total)')

# 2. Tìm images folder
# ====================
print("\n" + "=" * 50)
print("DEBUG: Finding images folder")
print("=" * 50)

def find_folder(start_path, target_name):
    """Find folder with specific name"""
    for root, dirs, files in os.walk(start_path):
        if os.path.basename(root) == target_name:
            return root
    return None

# Try to find images folder
images_path = find_folder('/content', 'images')
if images_path:
    print(f"\n[OK] Found 'images' folder at: {images_path}")
    print(f"     Contents: {os.listdir(images_path)}")
else:
    print("\n[X] 'images' folder NOT FOUND")

# Try to find labels folder
labels_path = find_folder('/content', 'labels')
if labels_path:
    print(f"\n[OK] Found 'labels' folder at: {labels_path}")
    print(f"     Contents: {os.listdir(labels_path)}")
else:
    print("\n[X] 'labels' folder NOT FOUND")

# 3. Show correct paths
# ====================
print("\n" + "=" * 50)
print("CORRECT PATHS")
print("=" * 50)

if images_path and labels_path:
    base = os.path.dirname(images_path)
    print(f"\npath: {base}")
    print(f"train: images/train")
    print(f"val: images/val")
    print(f"test: images/test")

# 4. Verify each split
# ====================
print("\n" + "=" * 50)
print("VERIFY SPLITS")
print("=" * 50)

if images_path:
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(images_path, split)
        lbl_dir = os.path.join(labels_path, split)
        
        if os.path.exists(img_dir):
            imgs = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
            print(f"\n[OK] {split.upper()}/")
            print(f"     Images: {imgs}")
        else:
            print(f"\n[X] {split.upper()}/ - NOT FOUND")
        
        if os.path.exists(lbl_dir):
            lbls = len([f for f in os.listdir(lbl_dir) if f.endswith('.txt')])
            print(f"     Labels: {lbls}")
        else:
            print(f"     Labels: NOT FOUND")

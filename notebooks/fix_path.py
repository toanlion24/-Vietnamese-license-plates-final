# FIX SCRIPT - Đổi tên folder để Python tìm được
# ============================================================

import os
import shutil

print("=" * 60)
print("FIX: Đổi tên folder có space")
print("=" * 60)

dataset_path = "/content/dataset"

# Đổi tên folder "License Plate Detection Dataset" -> "lp_dataset"
old_name = os.path.join(dataset_path, "License Plate Detection Dataset")
new_name = os.path.join(dataset_path, "lp_dataset")

if os.path.exists(old_name):
    if os.path.exists(new_name):
        shutil.rmtree(new_name)
    os.rename(old_name, new_name)
    print(f"OK: Đổi tên thành công")
    print(f"   {old_name}")
    print(f" -> {new_name}")
else:
    print(f"ERROR: Không tìm thấy folder: {old_name}")
    print("Các folder hiện có:")
    for item in os.listdir(dataset_path):
        print(f"   - {item}")

# Verify structure mới
print("\n" + "=" * 60)
print("Verify structure sau khi fix")
print("=" * 60)

for split in ['train', 'val', 'test']:
    img_dir = os.path.join(new_name, 'images', split)
    lbl_dir = os.path.join(new_name, 'labels', split)

    if os.path.exists(img_dir):
        imgs = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        lbls = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')] if os.path.exists(lbl_dir) else []
        print(f"  {split.upper():5} | images: {len(imgs):5} | labels: {len(lbls):5}")
    else:
        print(f"  {split.upper()}: NOT FOUND")

print("\nCORRECT PATH FOR data.yaml:")
print(f"path: {new_name}")
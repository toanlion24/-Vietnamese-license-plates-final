"""
Dataset Verification Script
Checks dataset structure and annotation quality
"""

import os
import sys
from pathlib import Path
from collections import Counter

def verify_dataset():
    """Verify the Vietnamese LPR dataset"""
    
    print("=" * 60)
    print("VIETNAMESE LICENSE PLATE DATASET VERIFICATION")
    print("=" * 60)
    
    base_path = Path(r"D:\ComputerVisionLasted\License Plate Detection Dataset")
    
    print("\n[*] Dataset Structure:")
    
    splits = ['train', 'val', 'test']
    stats = {}
    
    for split in splits:
        img_dir = base_path / 'images' / split
        lbl_dir = base_path / 'labels' / split
        
        if not img_dir.exists():
            print(f"  [X] {split}: images folder not found")
            continue
        
        if not lbl_dir.exists():
            print(f"  [X] {split}: labels folder not found")
            continue
        
        # Count files
        img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
        lbl_files = list(lbl_dir.glob('*.txt'))
        
        # Check for images without labels
        img_names = {f.stem for f in img_files}
        lbl_names = {f.stem for f in lbl_files}
        missing_labels = img_names - lbl_names
        orphan_labels = lbl_names - img_names
        
        stats[split] = {
            'images': len(img_files),
            'labels': len(lbl_files),
            'missing_labels': len(missing_labels),
            'orphan_labels': len(orphan_labels)
        }
        
        print(f"\n  [OK] {split.upper()}:")
        print(f"     Images: {len(img_files):,}")
        print(f"     Labels: {len(lbl_files):,}")
        
        if missing_labels:
            print(f"     [!] Images without labels: {len(missing_labels)}")
        if orphan_labels:
            print(f"     [!] Labels without images: {len(orphan_labels)}")
    
    # Total
    if stats:
        total_imgs = sum(s['images'] for s in stats.values())
        total_lbls = sum(s['labels'] for s in stats.values())
        
        print(f"\n[+] TOTAL:")
        print(f"   Images: {total_imgs:,}")
        print(f"   Labels: {total_lbls:,}")
    
    # Analyze annotations
    print("\n\n[*] Annotation Analysis:")
    print("-" * 40)
    
    class_counts = Counter()
    bbox_widths = []
    bbox_heights = []
    bbox_areas = []
    
    for split in splits:
        lbl_dir = base_path / 'labels' / split
        
        if not lbl_dir.exists():
            continue
        
        lbl_files = list(lbl_dir.glob('*.txt'))[:500]  # Sample 500 per split
        for lbl_file in lbl_files:
            try:
                with open(lbl_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls = int(parts[0])
                            w = float(parts[3])
                            h = float(parts[4])
                            
                            class_counts[cls] += 1
                            bbox_widths.append(w)
                            bbox_heights.append(h)
                            bbox_areas.append(w * h)
            except Exception as e:
                print(f"Error reading {lbl_file}: {e}")
    
    print(f"\n  Classes found:")
    for cls, count in sorted(class_counts.items()):
        class_name = 'license_plate' if cls == 0 else f'class_{cls}'
        print(f"    Class {cls} ({class_name}): {count:,} annotations")
    
    if bbox_widths:
        avg_w = sum(bbox_widths)/len(bbox_widths)*100
        avg_h = sum(bbox_heights)/len(bbox_heights)*100
        avg_area = sum(bbox_areas)/len(bbox_areas)*100
        
        print(f"\n  Bounding Box Statistics:")
        print(f"    Avg Width:  {avg_w:.2f}% of image")
        print(f"    Avg Height: {avg_h:.2f}% of image")
        print(f"    Avg Area:   {avg_area:.4f}% of image")
        
        print(f"\n  Width range:  {min(bbox_widths)*100:.2f}% - {max(bbox_widths)*100:.2f}%")
        print(f"  Height range: {min(bbox_heights)*100:.2f}% - {max(bbox_heights)*100:.2f}%")
    
    # Sample images check
    print("\n\n[*] Sample Images Check:")
    print("-" * 40)
    
    train_dir = base_path / 'images' / 'train'
    if train_dir.exists():
        samples = list(train_dir.glob('*.jpg'))[:5]
        for sample in samples:
            stem = sample.stem
            lbl_path = base_path / 'labels' / 'train' / f'{stem}.txt'
            
            if lbl_path.exists():
                with open(lbl_path, 'r') as f:
                    content = f.read().strip()
                print(f"  [OK] {sample.name}: {content[:50]}...")
            else:
                print(f"  [X] {sample.name}: No label found")
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    
    return stats

if __name__ == "__main__":
    verify_dataset()

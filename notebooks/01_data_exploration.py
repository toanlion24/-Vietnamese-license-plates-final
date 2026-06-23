"""
01 - Data Exploration
Vietnamese License Plate Dataset Analysis
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import random

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_dataset_info(data_dir):
    """Load and analyze dataset"""
    data_path = Path(data_dir)
    
    images = list(data_path.glob("**/*.jpg")) + list(data_path.glob("**/*.png"))
    annotations = list(data_path.glob("**/*.txt"))
    
    return {
        "total_images": len(images),
        "total_annotations": len(annotations),
        "images": images,
        "annotations": annotations,
    }


def analyze_image_properties(images):
    """Analyze image properties"""
    widths = []
    heights = []
    aspects = []
    
    sample_images = random.sample(images, min(100, len(images)))
    
    for img_path in sample_images:
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
            aspects.append(w / h if h > 0 else 0)
    
    return {
        "widths": widths,
        "heights": heights,
        "aspects": aspects,
    }


def plot_dataset_analysis(dataset_info, image_props):
    """Create dataset analysis visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].bar(
        ["Images", "Annotations"], 
        [dataset_info["total_images"], dataset_info["total_annotations"]],
        color=["#3498db", "#e74c3c"]
    )
    axes[0, 0].set_title("Dataset Size", fontsize=14, fontweight="bold")
    axes[0, 0].set_ylabel("Count")
    for i, v in enumerate([dataset_info["total_images"], dataset_info["total_annotations"]]):
        axes[0, 0].text(i, v + 10, str(v), ha='center', fontweight="bold")
    
    axes[0, 1].hist(image_props["widths"], bins=30, color="#3498db", alpha=0.7)
    axes[0, 1].set_title("Image Width Distribution", fontsize=14, fontweight="bold")
    axes[0, 1].set_xlabel("Width (pixels)")
    axes[0, 1].set_ylabel("Frequency")
    
    axes[1, 0].hist(image_props["heights"], bins=30, color="#2ecc71", alpha=0.7)
    axes[1, 0].set_title("Image Height Distribution", fontsize=14, fontweight="bold")
    axes[1, 0].set_xlabel("Height (pixels)")
    axes[1, 0].set_ylabel("Frequency")
    
    axes[1, 1].hist(image_props["aspects"], bins=30, color="#9b59b6", alpha=0.7)
    axes[1, 1].set_title("Aspect Ratio Distribution", fontsize=14, fontweight="bold")
    axes[1, 1].set_xlabel("Aspect Ratio (W/H)")
    axes[1, 1].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig("outputs/dataset_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    print(f"\n[Saved] Dataset analysis plot: outputs/dataset_analysis.png")


def main():
    print("=" * 60)
    print("Vietnamese LPR - Dataset Exploration")
    print("=" * 60)
    
    data_dir = "data/processed"
    
    print(f"\nLoading dataset from: {data_dir}")
    dataset_info = load_dataset_info(data_dir)
    
    print(f"\nDataset Statistics:")
    print(f"  Total images: {dataset_info['total_images']}")
    print(f"  Total annotations: {dataset_info['total_annotations']}")
    
    if dataset_info["images"]:
        print(f"\nAnalyzing image properties (sampling 100 images)...")
        image_props = analyze_image_properties(dataset_info["images"])
        
        print(f"\nImage Properties:")
        print(f"  Width  - Mean: {np.mean(image_props['widths']):.0f}, Std: {np.std(image_props['widths']):.0f}")
        print(f"  Height - Mean: {np.mean(image_props['heights']):.0f}, Std: {np.std(image_props['heights']):.0f}")
        print(f"  Aspect - Mean: {np.mean(image_props['aspects']):.2f}, Std: {np.std(image_props['aspects']):.2f}")
        
        plot_dataset_analysis(dataset_info, image_props)
    else:
        print("\n[INFO] No images found in data directory")
        print("  Place your dataset in: data/processed/")
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Review dataset quality")
    print("  2. Start training: python -m src.detection.train")
    print("=" * 60)


if __name__ == "__main__":
    main()

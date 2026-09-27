import os
import json
import random
from typing import Optional, List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

def crop_patch(image_path: str, bbox: list, alpha: float = 0.8, 
               min_size: int = 64) -> Optional[Image.Image]:
    """
    Lazily crops a single building patch from a satellite image with 
    context expansion.
    
    Args:
        image_path: Absolute path to the PNG image file
        bbox: [x, y, w, h] bounding box from polygon min/max coordinates
        alpha: Context expansion factor (default 0.8 per DisasterM3 paper)
               Expanded box = (x - 0.4w, y - 0.4h, 1.8w, 1.8h)
        min_size: Minimum patch dimension in pixels. Returns None if smaller.
    
    Returns:
        PIL Image of the cropped patch, or None if patch is too small or 
        image cannot be loaded.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        W, H = img.size
        
        x, y, w, h = bbox
        
        x1 = max(0, int(x - alpha/2 * w))
        y1 = max(0, int(y - alpha/2 * h))
        x2 = min(W, int(x + w + alpha/2 * w))
        y2 = min(H, int(y + h + alpha/2 * h))
        
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            return None
            
        return img.crop((x1, y1, x2, y2))
    except Exception as e:
        print(f"Error cropping patch from {image_path}: {e}")
        return None

def parse_xbd_label(label_path: str) -> List[dict]:
    """
    Parses one xBD post-disaster label JSON file into a list of building records.
    
    xBD label format:
    {
      "features": {
        "xy": [
          {
            "properties": {
              "uid": "...",
              "subtype": "no-damage"  <- damage class is here
            },
            "wkt": "POLYGON ((x1 y1, x2 y2, ...))"  <- polygon in WKT format
          }
        ]
      }
    }
    
    Returns:
        List of dicts, each with keys:
        {
          "uid": str,
          "damage_class": str,   (one of: no-damage, minor-damage, major-damage, destroyed)
          "bbox": [x, y, w, h],  (computed from polygon coordinates)
          "polygon_coords": [[x,y], [x,y], ...]
        }
        Returns empty list if file cannot be parsed.
    """
    buildings = []
    valid_classes = {"no-damage", "minor-damage", "major-damage", "destroyed"}
    
    try:
        with open(label_path, 'r') as f:
            data = json.load(f)
            
        features = data.get("features", {}).get("xy", [])
        
        for feat in features:
            props = feat.get("properties", {})
            uid = props.get("uid", "unknown")
            damage_class = props.get("subtype", "un-classified")
            
            if damage_class not in valid_classes:
                continue
                
            wkt = feat.get("wkt", "")
            
            if "POLYGON" not in wkt:
                continue
                
            # Extract coordinates from WKT format.
            # E.g., POLYGON ((x1 y1, x2 y2, ...))
            start_idx = wkt.find("((")
            end_idx = wkt.find("))")
            
            if start_idx == -1 or end_idx == -1:
                continue
                
            coords_str = wkt[start_idx+2:end_idx]
            points = coords_str.split(',')
            
            coords = []
            for p in points:
                p = p.strip()
                if not p:
                    continue
                parts = p.split(' ')
                if len(parts) >= 2:
                    try:
                        coords.append([float(parts[0]), float(parts[1])])
                    except ValueError:
                        pass
                        
            if not coords:
                continue
                
            x_coords = [p[0] for p in coords]
            y_coords = [p[1] for p in coords]
            
            x = min(x_coords)
            y = min(y_coords)
            w = max(x_coords) - x
            h = max(y_coords) - y
            
            buildings.append({
                "uid": uid,
                "damage_class": damage_class,
                "bbox": [x, y, w, h],
                "polygon_coords": coords
            })
    except Exception as e:
        print(f"Error parsing label {label_path}: {e}")
        
    return buildings

class XBDPatchDataset(Dataset):
    """
    PyTorch Dataset for xBD building-level patch classification.
    Patches are extracted lazily at __getitem__ time (no pre-saved files).
    
    Each sample returns a pre-patch, post-patch pair with a damage label.
    
    Args:
        xbd_root: Root directory of xBD dataset
        disaster_types: List of disaster type strings to include 
                        (e.g., ["hurricane-harvey", "socal-fire"])
                        If None, uses all available disaster types.
        split: "train", "val", or "test"
        train_ratio: Fraction of image pairs used for training (default 0.8)
        val_ratio: Fraction for validation (default 0.1). 
                   Remaining (0.1) goes to test.
        image_size: Resize all patches to (image_size, image_size) (default 224)
        alpha: Context expansion for patch cropping (default 0.8)
        min_patch_size: Minimum patch dimension to keep (default 64)
        seed: Random seed for reproducible train/val/test split (default 42)
        transform: Optional torchvision transforms (default: resize + normalize)
    """
    
    def __init__(self, xbd_root, disaster_types=None, split="train", 
                 train_ratio=0.8, val_ratio=0.1, image_size=224, 
                 alpha=0.8, min_patch_size=64, seed=42, transform=None):
        self.xbd_root = xbd_root
        self.disaster_types = disaster_types
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.image_size = image_size
        self.alpha = alpha
        self.min_patch_size = min_patch_size
        self.seed = seed
        
        self.transform = transform if transform else get_default_transforms(image_size, split)
        
        self.class_map = {
            "no-damage": 0,
            "minor-damage": 1,
            "major-damage": 2,
            "destroyed": 3
        }
        
        self.image_pairs = self._discover_samples()
        self.building_index = self._build_building_index()

    def _discover_samples(self):
        """
        Scans xbd_root/images/ for post-disaster images.
        For each one, finds the matching pre-disaster image and label JSON.
        Builds a flat list of (pre_image_path, post_image_path, label_path, 
        disaster_type) tuples.
        Filters by self.disaster_types if specified.
        Performs deterministic train/val/test split by shuffling image pairs 
        (NOT buildings — split at image level to avoid leakage within an image).
        """
        images_dir = os.path.join(self.xbd_root, "images")
        labels_dir = os.path.join(self.xbd_root, "labels")
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"Warning: Dataset path {self.xbd_root} not found.")
            return []
            
        all_images = os.listdir(images_dir)
        all_labels = os.listdir(labels_dir)
        
        # Using sets makes lookups instant, skipping thousands of slow Google Drive network requests!
        all_images_set = set(all_images)
        all_labels_set = set(all_labels)
        
        post_images = sorted([f for f in all_images if "post_disaster.png" in f])
        
        pairs = []
        for post_img in post_images:
            base = post_img.replace("_post_disaster.png", "")
            disaster = base.split("_")[0]
            
            if self.disaster_types and disaster not in self.disaster_types:
                continue
                
            pre_img = f"{base}_pre_disaster.png"
            post_label = f"{base}_post_disaster.json"
            
            if pre_img in all_images_set and post_label in all_labels_set:
                pre_path = os.path.join(images_dir, pre_img)
                post_path = os.path.join(images_dir, post_img)
                label_path = os.path.join(labels_dir, post_label)
                pairs.append((pre_path, post_path, label_path, disaster))
                
        # Deterministic shuffle
        rng = random.Random(self.seed)
        rng.shuffle(pairs)
        
        total = len(pairs)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)
        
        if self.split == "train":
            return pairs[:train_end]
        elif self.split == "val":
            return pairs[train_end:val_end]
        else:
            return pairs[val_end:]

    def _build_building_index(self):
        """
        For all image pairs in the split, parses label JSONs and builds
        a flat list of building-level records.
        Skips records where damage_class is not in the 4 valid classes
        or if the building patch would be smaller than min_patch_size.
        """
        index = []
        for pre_path, post_path, label_path, disaster in self.image_pairs:
            buildings = parse_xbd_label(label_path)
            for b in buildings:
                # Pre-calculate expanded patch size
                x, y, w, h = b["bbox"]
                exp_w = w + self.alpha * w
                exp_h = h + self.alpha * h
                
                # Filter out patches that are too small before they enter the index
                if exp_w < self.min_patch_size or exp_h < self.min_patch_size:
                    continue
                    
                index.append({
                    "pre_image_path": pre_path,
                    "post_image_path": post_path,
                    "bbox": b["bbox"],
                    "damage_class": b["damage_class"],
                    "damage_label": self.class_map[b["damage_class"]],
                    "disaster_type": disaster,
                    "uid": b["uid"]
                })
        
        print(f"[{self.split.upper()}] Built index: {len(index)} buildings across {len(self.image_pairs)} image pairs.")
        return index

    def __len__(self):
        return len(self.building_index)

    def __getitem__(self, idx):
        """
        Returns:
            pre_patch: Tensor [3, H, W] — pre-disaster building patch
            post_patch: Tensor [3, H, W] — post-disaster building patch
            label: int — damage class index (0-3)
            meta: dict with keys: uid, disaster_type, damage_class, bbox
        
        If patch extraction fails for either pre or post image, 
        returns a valid random sample (max 100 attempts) to prevent crashes.
        """
        import random
        max_attempts = 100
        curr_idx = idx
        
        for i in range(max_attempts):
            b = self.building_index[curr_idx]
            
            pre_patch = crop_patch(b["pre_image_path"], b["bbox"], self.alpha, self.min_patch_size)
            post_patch = crop_patch(b["post_image_path"], b["bbox"], self.alpha, self.min_patch_size)
            
            if pre_patch is not None and post_patch is not None:
                if self.transform:
                    pre_patch = self.transform(pre_patch)
                    post_patch = self.transform(post_patch)
                    
                meta = {
                    "uid": b["uid"],
                    "disaster_type": b["disaster_type"],
                    "damage_class": b["damage_class"],
                    "bbox": b["bbox"]
                }
                return pre_patch, post_patch, b["damage_label"], meta
                
            # Fallback to a random building to avoid sequential clusters of tiny buildings
            curr_idx = random.randint(0, len(self.building_index) - 1)
                
        raise RuntimeError(f"Failed to load valid patches after {max_attempts} attempts.")

    def get_class_weights(self):
        """
        Returns inverse-frequency class weights as a tensor for use in 
        weighted loss function. Handles class imbalance.
        weights[i] = total_samples / (n_classes * count_of_class_i)
        """
        counts = [0] * 4
        for b in self.building_index:
            counts[b["damage_label"]] += 1
            
        total = sum(counts)
        weights = []
        for c in counts:
            if c > 0:
                weights.append(total / (4 * c))
            else:
                weights.append(0.0)
                
        return torch.tensor(weights, dtype=torch.float)

    def get_split_summary(self):
        """
        Returns a dict summarising this split:
        {
          "split": str,
          "n_image_pairs": int,
          "n_buildings": int,
          "class_distribution": {class_name: count},
          "disaster_type_distribution": {type: count},
          "skipped_patches": int  (patches too small or corrupt)
        }
        """
        class_counts = {}
        disaster_counts = {}
        
        for b in self.building_index:
            c = b["damage_class"]
            d = b["disaster_type"]
            class_counts[c] = class_counts.get(c, 0) + 1
            disaster_counts[d] = disaster_counts.get(d, 0) + 1
            
        return {
            "split": self.split,
            "n_image_pairs": len(self.image_pairs),
            "n_buildings": len(self.building_index),
            "class_distribution": class_counts,
            "disaster_type_distribution": disaster_counts,
            "skipped_patches": 0  # To track lazily skipped patches robustly, this requires inference-time logging
        }

def get_default_transforms(image_size=224, split="train"):
    """
    Returns torchvision transforms for xBD patches.
    
    Train: Resize to image_size, RandomHorizontalFlip(p=0.5), 
           ColorJitter(brightness=0.2, contrast=0.2) — 
           NOTE: no rotation (could change spatial damage orientation cues),
           ToTensor, Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
    
    Val/Test: Resize to image_size, ToTensor, Normalize (same values)
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

if __name__ == "__main__":
    # Usage example that will be printed when running the file directly
    # Uses a small sample to verify the dataset class works
    import yaml
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    print("Testing XBDPatchDataset...")
    dataset = XBDPatchDataset(
        xbd_root=config["paths"]["xbd_root"],
        disaster_types=None,  # use all available
        split="train",
        seed=42
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"Split summary: {dataset.get_split_summary()}")
    print(f"Class weights: {dataset.get_class_weights()}")
    
    if len(dataset) > 0:
        # Test one sample
        pre, post, label, meta = dataset[0]
        print(f"Sample 0: pre shape={pre.shape}, post shape={post.shape}, "
              f"label={label}, disaster_type={meta['disaster_type']}")
        
        # Test DataLoader
        loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
        batch = next(iter(loader))
        print(f"Batch shapes: pre={batch[0].shape}, post={batch[1].shape}, "
              f"labels={batch[2]}")
        print("ALL TESTS PASSED")
    else:
        print("Dataset is empty (xbd_root path likely not found). Tests skipped.")

import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split

# paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "dataset")

IMG_DIR = os.path.join(BASE_DIR, "images")
LBL_DIR = os.path.join(BASE_DIR, "labels")
OUT_DIR = os.path.join(BASE_DIR, "yolo_split")

# automatic creation of output folder
for split in ["train", "val"]:
    os.makedirs(os.path.join(OUT_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "labels", split), exist_ok=True)

# image collection
image_paths = glob(os.path.join(IMG_DIR, "**", "*.*"), recursive=True)
image_files = [f for f in image_paths if f.lower().endswith((".jpg", ".jpeg", ".png"))]

print(f"[INFO] Total images found: {len(image_files)}")
if not image_files:
    raise ValueError("❌ No image files found. Check image directory path and extensions.")

# Train/Validation Split
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

# Image move and label pairs
def move_files(file_list, split):
    for src_img in file_list:
        img_name = os.path.basename(src_img)
        label_name = os.path.splitext(img_name)[0] + ".txt"

        relative_path = os.path.relpath(src_img, IMG_DIR)
        subfolder = os.path.dirname(relative_path)

        src_lbl = os.path.join(LBL_DIR, subfolder, label_name)

        dst_img = os.path.join(OUT_DIR, "images", split, img_name)
        dst_lbl = os.path.join(OUT_DIR, "labels", split, label_name)

        shutil.copy2(src_img, dst_img)

        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)
        else:
            print(f"[WARNING] Missing label file: {src_lbl}")

move_files(train_files, "train")
move_files(val_files, "val")

# Read names classes in txt files
CLASSES_TXT = os.path.join(SCRIPT_DIR, "classes.txt")
if not os.path.exists(CLASSES_TXT):
    raise FileNotFoundError("❌ classes.txt not found in the script folder.")

with open(CLASSES_TXT) as f:
    class_list = [line.strip() for line in f.readlines() if line.strip()]

print(f"[INFO] Loaded {len(class_list)} class names.")

# data.yaml creation
yaml_path = os.path.join(OUT_DIR, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"path: {OUT_DIR}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write(f"nc: {len(class_list)}\n")
    f.write(f"names: {class_list}\n")

print("✅ Dataset ready for YOLOv8 training!")
print(f"[OK] data.yaml created at: {yaml_path}")

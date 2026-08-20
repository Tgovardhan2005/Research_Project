# AgriVision Edge Pipeline - Stage 1: Leaf Locator (YOLOv8)
# ==========================================================
# Dynamic Environmental Filtering: Merges PlantDoc and FieldPlant datasets,
# standardizes annotations to single-class 'Target_Leaf', and trains/resumes YOLOv8-Nano.

import os
import shutil
import glob
import yaml

# 1. Dataset Downloads via Roboflow API
def download_datasets():
    from roboflow import Roboflow
    
    print("Downloading PlantDoc Dataset from Roboflow...")
    rf1 = Roboflow(api_key="DMlwzmxlx7Ua2eBpOTbH")
    project1 = rf1.workspace("qiuan-wu").project("plantdoc-dataset")
    version1 = project1.version(1)
    dataset1 = version1.download("yolov8")

    print("Downloading FieldPlant Dataset from Roboflow...")
    rf2 = Roboflow(api_key="DMlwzmxlx7Ua2eBpOTbH")
    project2 = rf2.workspace("plant-disease-detection").project("fieldplant")
    version2 = project2.version(11)
    dataset2 = version2.download("yolov8")
    
    return dataset1, dataset2

# 2. Merge, Rename, and Standardize Classes to 'Target_Leaf'
def process_and_merge_datasets():
    datasets = ['/content/PlantDoc-Dataset-1', '/content/FieldPlant-11']
    splits = ['train', 'valid']

    base_dir = '/content/Stage1_Dataset'
    for split in splits:
        os.makedirs(os.path.join(base_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, split, 'labels'), exist_ok=True)

    image_count = 0

    print("Merging datasets, renaming files, and standardizing classes to 'Target_Leaf'...")
    for dataset in datasets:
        for split in splits:
            img_dir = os.path.join(dataset, split, 'images')
            lbl_dir = os.path.join(dataset, split, 'labels')

            if not os.path.exists(img_dir):
                continue

            for img_path in glob.glob(os.path.join(img_dir, '*.*')):
                ext = os.path.splitext(img_path)[1]

                # Shorten file names to avoid 255 character path limit issues
                new_img_name = f"leaf_{image_count}{ext}"
                new_lbl_name = f"leaf_{image_count}.txt"

                original_lbl_name = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
                lbl_path = os.path.join(lbl_dir, original_lbl_name)

                # Copy image with unified short name
                shutil.copy(img_path, os.path.join(base_dir, split, 'images', new_img_name))

                # Copy and update label file to force class ID to 0 ('Target_Leaf')
                if os.path.exists(lbl_path):
                    with open(lbl_path, 'r') as f_in:
                        lines = f_in.readlines()

                    with open(os.path.join(base_dir, split, 'labels', new_lbl_name), 'w') as f_out:
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) > 0:
                                parts[0] = '0'  # Forcing single class ID 0
                                f_out.write(' '.join(parts) + '\n')
                image_count += 1

    print(f"✅ Successfully unified and renamed {image_count} images into a single-class dataset!")

    # Generate custom YOLO data.yaml configuration
    yaml_content = {
        'path': base_dir,
        'train': 'train/images',
        'val': 'valid/images',
        'nc': 1,
        'names': ['Target_Leaf']
    }

    with open(os.path.join(base_dir, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_content, f)
        
    return base_dir

# 3. Model Training & Resuming
def train_stage1_model():
    from ultralytics import YOLO

    # Check for existing checkpoint to resume, otherwise initialize new YOLOv8 model
    checkpoint_path = '/content/last (2).pt'
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        model = YOLO(checkpoint_path)
        results = model.train(resume=True)
    else:
        print("Initializing new YOLOv8n model for Stage 1 Leaf Locator...")
        model = YOLO('yolov8n.pt')
        results = model.train(
            data='/content/Stage1_Dataset/data.yaml',
            epochs=50,
            imgsz=640,
            batch=16,
            project='/content/runs/detect/AgriVision_Edge_Pipeline',
            name='Stage1_Leaf_Locator'
        )
    return results

# 4. Compression & Result Packaging
def package_training_results():
    try:
        from google.colab import files
        is_colab = True
    except ImportError:
        is_colab = False

    source_folder = '/content/runs/detect/AgriVision_Edge_Pipeline/Stage1_Leaf_Locator'
    output_zip = '/content/Stage1_Results'

    if os.path.exists(source_folder):
        print("Zipping training output directory...")
        shutil.make_archive(output_zip, 'zip', source_folder)
        print(f"Archive created at {output_zip}.zip")
        if is_colab:
            print("Triggering download...")
            files.download(output_zip + '.zip')
    else:
        print(f"Source folder {source_folder} not found.")

if __name__ == '__main__':
    # Execute Stage 1 Pipeline
    download_datasets()
    process_and_merge_datasets()
    train_stage1_model()
    package_training_results()

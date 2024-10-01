import os
import shutil
import random
from math import ceil

# Set random seed for reproducibility
random.seed(41)

# Paths
base_dir = 'datasets/BraTs2020_t1_t2_tiny'  # Replace with your base folder path
oriT1_dir = os.path.join(base_dir, 'oriT1')
oriT2_dir = os.path.join(base_dir, 'oriT2')
lrT1x2_dir = os.path.join(base_dir, 'orLRbicT1/x2')
lrT1x4_dir = os.path.join(base_dir, 'orLRbicT1/x4')
lrT2x2_dir = os.path.join(base_dir, 'orLRbicT2/x2')
lrT2x4_dir = os.path.join(base_dir, 'orLRbicT2/x4')

# Output directories
train_oriT1_dir = os.path.join(base_dir, 'train/oriT1')
train_oriT2_dir = os.path.join(base_dir, 'train/oriT2')
train_lrT1x2_dir = os.path.join(base_dir, 'train/orLRbicT1/x2')
train_lrT1x4_dir = os.path.join(base_dir, 'train/orLRbicT1/x4')
train_lrT2x2_dir = os.path.join(base_dir, 'train/orLRbicT2/x2')
train_lrT2x4_dir = os.path.join(base_dir, 'train/orLRbicT2/x4')

val_oriT1_dir = os.path.join(base_dir, 'val/oriT1')
val_oriT2_dir = os.path.join(base_dir, 'val/oriT2')
val_lrT1x2_dir = os.path.join(base_dir, 'val/orLRbicT1/x2')
val_lrT1x4_dir = os.path.join(base_dir, 'val/orLRbicT1/x4')
val_lrT2x2_dir = os.path.join(base_dir, 'val/orLRbicT2/x2')
val_lrT2x4_dir = os.path.join(base_dir, 'val/orLRbicT2/x4')

# Create directories if they don't exist
os.makedirs(train_oriT1_dir, exist_ok=True)
os.makedirs(train_oriT2_dir, exist_ok=True)
os.makedirs(train_lrT1x2_dir, exist_ok=True)
os.makedirs(train_lrT1x4_dir, exist_ok=True)
os.makedirs(train_lrT2x2_dir, exist_ok=True)
os.makedirs(train_lrT2x4_dir, exist_ok=True)

os.makedirs(val_oriT1_dir, exist_ok=True)
os.makedirs(val_oriT2_dir, exist_ok=True)
os.makedirs(val_lrT1x2_dir, exist_ok=True)
os.makedirs(val_lrT1x4_dir, exist_ok=True)
os.makedirs(val_lrT2x2_dir, exist_ok=True)
os.makedirs(val_lrT2x4_dir, exist_ok=True)

# Get list of all image files
T1_files = sorted(os.listdir(oriT1_dir))

# Function to perform manual KFold split (n_splits=5)
def manual_kfold_split(file_list, n_splits=5, seed=41):
    # Set the seed for reproducibility
    random.seed(seed)

    # Shuffle the file list
    shuffled_files = file_list[:]
    random.shuffle(shuffled_files)

    # Determine the size of each fold
    fold_size = len(shuffled_files) // n_splits
    remainder = len(shuffled_files) % n_splits
    
    # Creating fold boundaries
    folds = []
    start = 0
    for i in range(n_splits):
        end = start + fold_size + (1 if i < remainder else 0)
        folds.append(shuffled_files[start:end])
        start = end
    
    # First fold as validation set, the rest as train set
    val_files = folds[0]
    train_files = [file for i in range(1, n_splits) for file in folds[i]]
    
    return train_files, val_files

# Perform KFold split to get train and validation files for first fold
n_splits = 5
train_files, val_files = manual_kfold_split(T1_files, n_splits=n_splits, seed=41)

# Function to move files
def move_files(file_list, src_dir, dest_dir):
    for file_name in file_list:
        src_path = os.path.join(src_dir, file_name)
        dest_path = os.path.join(dest_dir, file_name)
        shutil.copy2(src_path, dest_path)

# Move training files
move_files(train_files, oriT1_dir, train_oriT1_dir)
move_files(train_files, oriT2_dir, train_oriT2_dir)
move_files(train_files, lrT1x2_dir, train_lrT1x2_dir)
move_files(train_files, lrT1x4_dir, train_lrT1x4_dir)
move_files(train_files, lrT2x2_dir, train_lrT2x2_dir)
move_files(train_files, lrT2x4_dir, train_lrT2x4_dir)

# Move validation files
move_files(val_files, oriT1_dir, val_oriT1_dir)
move_files(val_files, oriT2_dir, val_oriT2_dir)
move_files(val_files, lrT1x2_dir, val_lrT1x2_dir)
move_files(val_files, lrT1x4_dir, val_lrT1x4_dir)
move_files(val_files, lrT2x2_dir, val_lrT2x2_dir)
move_files(val_files, lrT2x4_dir, val_lrT2x4_dir)

print('Files have been split and moved successfully according to KFold first fold!')

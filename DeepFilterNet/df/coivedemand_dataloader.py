# split_train_to_val.py

import os

import shutil

import random

def create_val_from_train(train_clean_dir, train_noisy_dir, val_ratio=0.1):

    

    # Get all training files

    train_files = [f for f in os.listdir(train_clean_dir) if f.endswith('.wav')]

    

    # Simple random split (no speaker grouping)

    random.seed(42)  # For reproducibility

    random.shuffle(train_files)

    

    # Calculate how many files for validation

    n_val_files = int(len(train_files) * val_ratio)

    val_files = train_files[:n_val_files]

    

    # Create validation directories

    base_dir = os.path.dirname(os.path.dirname(train_clean_dir))  # Go up to voicebank_dns_format

    val_clean_dir = os.path.join(base_dir, "validation_set", "clean")

    val_noisy_dir = os.path.join(base_dir, "validation_set", "noisy")

    

    os.makedirs(val_clean_dir, exist_ok=True)

    os.makedirs(val_noisy_dir, exist_ok=True)

    

    # Move validation files

    moved_files = 0

    for file in val_files:

        try:

            # Move to validation

            shutil.move(f"{train_clean_dir}/{file}", f"{val_clean_dir}/{file}")

            shutil.move(f"{train_noisy_dir}/{file}", f"{val_noisy_dir}/{file}")

            moved_files += 1

        except FileNotFoundError as e:

            print(f"Warning: Could not find {file} in noisy directory")

    

    print(f"Total training files: {len(train_files)}")

    print(f"Moved {moved_files} file pairs to validation ({val_ratio*100:.1f}%)")

    print(f"Remaining in training: {len(train_files) - moved_files}")

    print(f"Validation dirs: {val_clean_dir}, {val_noisy_dir}")

# Your usage with full path:

import os

home_dir = os.path.expanduser("~")

create_val_from_train(

    "/home/emilybederov/Unet/audio_data/voicebank_dns_format/training_set/clean",    
    "/home/emilybederov/Unet/audio_data/voicebank_dns_format/training_set/noisy"

)

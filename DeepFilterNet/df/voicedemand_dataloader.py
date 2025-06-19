# split_train_to_val.py
import os
import shutil
import random

def create_val_from_train(train_clean_dir, train_noisy_dir, val_ratio=0.1):
    
    # Get all training files
    train_files = [f for f in os.listdir(train_clean_dir) if f.endswith('.wav')]
    
    # Split by speaker
    speakers = list(set([f.split('_')[0] for f in train_files]))
    random.shuffle(speakers)
    
    val_speakers = speakers[:int(len(speakers) * val_ratio)]
    
    # Create validation directories
    val_clean_dir = train_clean_dir.replace('trainset', 'valset')
    val_noisy_dir = train_noisy_dir.replace('trainset', 'valset')
    os.makedirs(val_clean_dir, exist_ok=True)
    os.makedirs(val_noisy_dir, exist_ok=True)
    
    # Move validation files
    for file in train_files:
        speaker = file.split('_')[0]
        if speaker in val_speakers:
            # Move to validation
            shutil.move(f"{train_clean_dir}/{file}", f"{val_clean_dir}/{file}")
            shutil.move(f"{train_noisy_dir}/{file}", f"{val_noisy_dir}/{file}")
    
    print(f"Created validation set with {len(val_speakers)} speakers")

# Usage  
create_val_from_train(
     "/home/emilybederov/Unet/audio_data/voicebank_dns_format/training_set/clean",
    "/home/emilybederov/Unet/audio_data/voicebank_dns_format/training_set/noisy"
)
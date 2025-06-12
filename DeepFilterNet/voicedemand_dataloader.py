# voicedemand_dataloader.py
import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from typing import NamedTuple, Optional

class VoiceBankBatch(NamedTuple):
    speech: torch.Tensor      # Clean speech 
    noisy: torch.Tensor       # Noisy speech
    feat_erb: torch.Tensor    # ERB features 
    feat_spec: torch.Tensor   # Spectral features
    snr: torch.Tensor         # SNR values
    ids: torch.Tensor         # Sample IDs
    timings: torch.Tensor     # Dummy timings for compatibility

class VoiceBankDataset(Dataset):
    def __init__(self, clean_dir: str, noisy_dir: str):
        self.clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.wav')])
        self.noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.wav')])
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        
    def __len__(self):
        return len(self.clean_files)
    
    def __getitem__(self, idx):
        # Load audio
        clean_path = os.path.join(self.clean_dir, self.clean_files[idx])
        noisy_path = os.path.join(self.noisy_dir, self.noisy_files[idx])
        
        clean, sr = torchaudio.load(clean_path)
        noisy, _ = torchaudio.load(noisy_path)
        
        # Simple preprocessing - convert to mono if needed
        if clean.shape[0] > 1:
            clean = clean.mean(0, keepdim=True)
        if noisy.shape[0] > 1:
            noisy = noisy.mean(0, keepdim=True)
        
        # Calculate simple SNR
        clean_power = (clean ** 2).mean()
        noise_power = ((noisy - clean) ** 2).mean()
        snr = 10 * torch.log10(clean_power / (noise_power + 1e-8))
        
        return {
            'clean': clean.squeeze(0),
            'noisy': noisy.squeeze(0),
            'snr': snr,
            'id': idx
        }

class VoiceBankDataLoader:
    def __init__(self, train_clean: str, train_noisy: str, 
                 val_clean: str, val_noisy: str,
                 test_clean: str, test_noisy: str,
                 batch_size: int = 8, batch_size_eval: int = 4, **kwargs):
        
        self.datasets = {
            'train': VoiceBankDataset(train_clean, train_noisy),
            'valid': VoiceBankDataset(val_clean, val_noisy),
            'test': VoiceBankDataset(test_clean, test_noisy)
        }
        
        self.batch_sizes = {
            'train': batch_size,
            'valid': batch_size_eval, 
            'test': batch_size_eval
        }
    
    def iter_epoch(self, split: str, seed: int):
        dataset = self.datasets[split]
        batch_size = self.batch_sizes[split]
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=2,
            collate_fn=self._collate_fn
        )
        
        for batch in loader:
            yield batch
    
    def _collate_fn(self, batch_list):
        # Stack all tensors properly
        speeches = torch.stack([b['clean'] for b in batch_list])
        noisys = torch.stack([b['noisy'] for b in batch_list])
        snrs = torch.stack([b['snr'] for b in batch_list])
        ids = torch.stack([torch.tensor(b['id']) for b in batch_list])
        
        # Create dummy features for now (will be computed in training loop)
        B, T = speeches.shape
        feat_erb = torch.zeros(B, 1, T//480, 32)  # Dummy ERB features
        feat_spec = torch.zeros(B, 1, T//480, 96, 2)  # Dummy spec features
        timings = torch.zeros(5)  # Dummy timings
        
        return VoiceBankBatch(
            speech=speeches.unsqueeze(1),     # Add channel dim
            noisy=noisys.unsqueeze(1),        # Add channel dim
            feat_erb=feat_erb,
            feat_spec=feat_spec,
            snr=snrs,
            ids=ids,
            timings=timings
        )
    
    def len(self, split: str):
        dataset = self.datasets[split]
        batch_size = self.batch_sizes[split]
        return len(dataset) // batch_size
    
    def get_batch_size(self, split: str):
        return self.batch_sizes[split]
    
    def set_batch_size(self, batch_size: int, split: str):
        self.batch_sizes[split] = batch_size

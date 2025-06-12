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
        
        # Convert to mono if needed
        if clean.shape[0] > 1:
            clean = clean.mean(0, keepdim=True)
        if noisy.shape[0] > 1:
            noisy = noisy.mean(0, keepdim=True)
        
        # Resample to model SR if needed (usually 48kHz)
        target_sr = 48000  # DeepFilterNet expects 48kHz
        if sr != target_sr:
            clean = torchaudio.functional.resample(clean, sr, target_sr)
            noisy = torchaudio.functional.resample(noisy, sr, target_sr)
        
        # STFT processing
        n_fft = 960
        hop_length = 480
        
        # Compute STFT
        clean_stft = torch.stft(clean.squeeze(0), n_fft, hop_length, 
                               window=torch.hann_window(n_fft), 
                               return_complex=True)
        noisy_stft = torch.stft(noisy.squeeze(0), n_fft, hop_length,
                               window=torch.hann_window(n_fft),
                               return_complex=True)
        
        # Convert to real representation [T, F, 2]
        clean_real = torch.view_as_real(clean_stft).permute(1, 0, 2)  # [T, F, 2]
        noisy_real = torch.view_as_real(noisy_stft).permute(1, 0, 2)  # [T, F, 2]
        
        # Create simple ERB and spec features (you can improve these later)
        T, F, _ = noisy_real.shape
        erb_feat = torch.randn(T, 32)  # Dummy ERB features for now
        spec_feat = noisy_real[:, :96, :]  # First 96 frequency bins
        
        # Calculate SNR
        clean_power = (clean ** 2).mean()
        noise_power = ((noisy - clean) ** 2).mean()
        snr = 10 * torch.log10(clean_power / (noise_power + 1e-8))
        
        return {
            'clean': clean_real,      # [T, F, 2]
            'noisy': noisy_real,      # [T, F, 2] 
            'erb_feat': erb_feat,     # [T, E]
            'spec_feat': spec_feat,   # [T, F', 2]
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

    def __len__(self):
        """Return length of training dataset for LR scheduler setup"""
        return self.len('train')
    
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
        speeches = torch.stack([b['clean'] for b in batch_list])      # [B, T, F, 2]
        noisys = torch.stack([b['noisy'] for b in batch_list])        # [B, T, F, 2]
        erb_feats = torch.stack([b['erb_feat'] for b in batch_list])  # [B, T, E]
        spec_feats = torch.stack([b['spec_feat'] for b in batch_list]) # [B, T, F', 2]
        snrs = torch.stack([b['snr'] for b in batch_list])
        ids = torch.stack([torch.tensor(b['id']) for b in batch_list])
        
        # Add channel dimension: [B, T, F, 2] -> [B, 1, T, F, 2]
        speeches = speeches.unsqueeze(1)
        noisys = noisys.unsqueeze(1) 
        erb_feats = erb_feats.unsqueeze(1)    # [B, 1, T, E]
        spec_feats = spec_feats.unsqueeze(1)  # [B, 1, T, F', 2]
        
        timings = torch.zeros(5)  # Dummy timings
        
        return VoiceBankBatch(
            speech=speeches,
            noisy=noisys,
            feat_erb=erb_feats,
            feat_spec=spec_feats,
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

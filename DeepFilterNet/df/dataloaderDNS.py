import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from typing import NamedTuple
from pathlib import Path

class DeepFilterBatch(NamedTuple):
    speech: torch.Tensor
    noisy: torch.Tensor
    feat_erb: torch.Tensor
    feat_spec: torch.Tensor
    snr: torch.Tensor
    ids: torch.Tensor
    timings: torch.Tensor

class DeepFilterDataset(Dataset):
    def __init__(self, clean_dir: str, noisy_dir: str):
        self.clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith(".wav")])
        self.noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith(".wav")])
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir)
        assert len(self.clean_files) == len(self.noisy_files), "Mismatched clean/noisy file counts"

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_path = self.clean_dir / self.clean_files[idx]
        noisy_path = self.noisy_dir / self.noisy_files[idx]

        clean, sr = torchaudio.load(clean_path)
        noisy, _ = torchaudio.load(noisy_path)

        if clean.shape[0] > 1:
            clean = clean.mean(0, keepdim=True)
        if noisy.shape[0] > 1:
            noisy = noisy.mean(0, keepdim=True)

        clean_power = (clean ** 2).mean()
        noise_power = ((noisy - clean) ** 2).mean()
        snr = 10 * torch.log10(clean_power / (noise_power + 1e-8))

        return {
            'clean': clean.squeeze(0),
            'noisy': noisy.squeeze(0),
            'snr': snr,
            'id': idx
        }

class DeepFilterDataLoader:
    def __init__(self, data_root="data", batch_size=8, batch_size_eval=4):
        self.datasets = {
            'train': DeepFilterDataset(f"{data_root}/train/clean", f"{data_root}/train/noisy"),
            'valid': DeepFilterDataset(f"{data_root}/val/clean", f"{data_root}/val/noisy"),
            'test': DeepFilterDataset(f"{data_root}/test/clean", f"{data_root}/test/noisy"),
        }
        self.batch_sizes = {
            'train': batch_size,
            'valid': batch_size_eval,
            'test': batch_size_eval,
        }

    def iter_epoch(self, split: str, seed: int = 0):
        torch.manual_seed(seed)
        dataset = self.datasets[split]
        loader = DataLoader(
            dataset,
            batch_size=self.batch_sizes[split],
            shuffle=(split == 'train'),
            num_workers=2,
            collate_fn=self._collate_fn
        )
        for batch in loader:
            yield batch

    def _collate_fn(self, batch_list):
        speeches = torch.stack([b['clean'] for b in batch_list])
        noisys = torch.stack([b['noisy'] for b in batch_list])
        snrs = torch.stack([b['snr'] for b in batch_list])
        ids = torch.tensor([b['id'] for b in batch_list])

        B, T = speeches.shape
        feat_erb = torch.zeros(B, 1, T // 480, 32)
        feat_spec = torch.zeros(B, 1, T // 480, 96, 2)
        timings = torch.zeros(5)

        return DeepFilterBatch(
            speech=speeches.unsqueeze(1),
            noisy=noisys.unsqueeze(1),
            feat_erb=feat_erb,
            feat_spec=feat_spec,
            snr=snrs,
            ids=ids,
            timings=timings
        )

    def len(self, split: str):
        return len(self.datasets[split]) // self.batch_sizes[split]

    def get_batch_size(self, split: str):
        return self.batch_sizes[split]

    def set_batch_size(self, batch_size: int, split: str):
        self.batch_sizes[split] = batch_size
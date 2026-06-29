
import argparse
import torch
from torch.utils.data import Dataset, DataLoader, Subset, Sampler
import numpy as np
import random
from collections import defaultdict
import sys
import torch.nn.utils as nn_utils

from utils.resample import time_resample
from utils.timefeatures import time_features
from utils.meter import AverageMeter
from layers.utils import WeightedFocalLoss
from models.DualFormer import Model

class Config:
    def __init__(self, args):
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.pred_len = max(self.pred_len) if isinstance(self.pred_len, (list, tuple)) else self.pred_lens
        self.device = torch.device(self.device if torch.cuda.is_available() else 'cpu')
        self.loss_fn = WeightedFocalLoss().to(self.device)
        print("=== Config Summary ===")
        for k, v in sorted(vars(self).items()):
            if k != 'loss_fn':
                print(f"  {k}: {v}")
        print("======================\n")
class PopulationDataset(Dataset):
    def __init__(self, all_glucose, all_timestamps, all_masks, all_types, seq_len, pred_len, time_freq='t'):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.time_freq = time_freq
        self.samples = [(g, t, m, typ) for g, t, m, typ in zip(all_glucose, all_timestamps, all_masks, all_types)]
        self.valid_indices = []
        for pid, (g, _, m, _) in enumerate(self.samples):
            total_len = len(g)
            for start in range(total_len - seq_len - pred_len + 1):
                end = start + seq_len + pred_len
                if m[start + seq_len:end].sum() > 0:
                    self.valid_indices.append((pid, start))
        self.total_valid = len(self.valid_indices)
        print(f"[Dataset] Total valid windows: {self.total_valid}")

    def __len__(self):
        return self.total_valid

    def __getitem__(self, idx):
        pid, s = self.valid_indices[idx]
        g, ts, mask, typ = self.samples[pid]
        e = min(s + self.seq_len + self.pred_len, len(g))
        s = max(0, e - self.seq_len - self.pred_len)

        x_enc = g[s:s + self.seq_len].reshape(-1, 1)
        y = g[s + self.seq_len:e].reshape(-1, 1)
        mask_pred = mask[s + self.seq_len:e].reshape(-1, 1)
        time_feat = time_features(ts, self.time_freq)
        x_mark_enc = time_feat[:, s:s + self.seq_len].T
        y_mark_dec = time_feat[:, s + self.seq_len:e].T

        # pad
        if y.shape[0] < self.pred_len:
            pad_len = self.pred_len - y.shape[0]
            y = np.pad(y, ((0, pad_len), (0, 0)), mode='constant')
            mask_pred = np.pad(mask_pred, ((0, pad_len), (0, 0)), mode='constant')
            y_mark_dec = np.pad(y_mark_dec, ((0, pad_len), (0, 0)), mode='constant')

        x_dec = np.zeros((self.pred_len, 1), dtype=np.float32)
        if len(x_enc) > 0:
            x_dec[0, 0] = x_enc[-1, 0]

        return (
            torch.from_numpy(x_enc).float(),
            torch.from_numpy(x_mark_enc).float(),
            torch.from_numpy(x_dec).float(),
            torch.from_numpy(y_mark_dec).float(),
            torch.from_numpy(y).float(),
            torch.from_numpy(mask_pred).float(),
            torch.tensor(typ, dtype=torch.long)
        )
class SubsetWithSamples(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.samples = dataset.samples
        self.valid_indices = [dataset.valid_indices[i] for i in indices]
        self.seq_len = dataset.seq_len
        self.pred_len = dataset.pred_len
def fishr_regularization(gradients_per_patient):
    if len(gradients_per_patient) < 2:
        return torch.tensor(0.0, device=gradients_per_patient[0].device if gradients_per_patient else torch.device('cpu'))
    grads = torch.stack(gradients_per_patient).float()
    mean_grad = grads.mean(dim=0, keepdim=True)
    centered = grads - mean_grad
    return (centered ** 2).sum(dim=1).mean()
class PatientBalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, samples_per_patient=16,
                 min_patients_per_batch=3, seed=42):
        if not hasattr(dataset, 'valid_indices'):
            raise TypeError("dataset must have .valid_indices")

        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_patient = samples_per_patient
        self.min_patients = max(2, min_patients_per_batch)
        self.rng = random.Random(seed)
        self.patient_to_indices = defaultdict(list)
        for local_idx, (pid, _) in enumerate(dataset.valid_indices):
            self.patient_to_indices[pid].append(local_idx)
        self.patients = list(self.patient_to_indices.keys())
        if len(self.patients) < 2:
            raise ValueError(f"Need >=2 patients, got {len(self.patients)}")
        total_windows = sum(len(idxs) for idxs in self.patient_to_indices.values())
        self._num_batches = total_windows // batch_size
        if self._num_batches == 0:
            raise ValueError("Not enough samples")

        print(f"[Sampler] Patients: {len(self.patients)}, Total windows: {total_windows}, Batches/epoch: {self._num_batches}")

    def __len__(self):
        return self._num_batches

    def __iter__(self):
        available_indices = set(range(len(self.dataset.valid_indices)))
        batches_yielded = 0

        while available_indices and batches_yielded < self._num_batches:
            n_patients = min(len(self.patients), self.min_patients + self.rng.randint(0, 3))
            selected_patients = self.rng.sample(self.patients, n_patients)

            batch_indices = []
            used_patients = set()

            for pid in selected_patients:
                if pid in used_patients:
                    continue
                patient_avail = [idx for idx in self.patient_to_indices[pid] if idx in available_indices]
                if not patient_avail:
                    continue

                n_sample = min(len(patient_avail), self.samples_per_patient)
                sampled = self.rng.sample(patient_avail, n_sample)
                batch_indices.extend(sampled)
                used_patients.add(pid)

                if len(batch_indices) >= self.batch_size:
                    break

            if len(batch_indices) >= self.batch_size // 2:
                for idx in batch_indices:
                    available_indices.discard(idx)
                yield batch_indices[:self.batch_size]
                batches_yielded += 1
def identity_collate(batch):

    return batch
def evaluate(model, loader, loss_fn, device, return_preds=False):
    model.eval()
    total_loss = total_count = 0
    preds_list, trues_list = [], []

    with torch.no_grad():
        for batch in loader:
            x_enc = torch.stack([b[0] for b in batch]).to(device)
            x_mark_enc = torch.stack([b[1] for b in batch]).to(device)
            y = torch.stack([b[4] for b in batch]).to(device)
            mask = torch.stack([b[5] for b in batch]).to(device)
            type_id = torch.stack([b[6] for b in batch]).to(device)
            out = model(x_enc, x_mark_enc, type_id)

            pred = loss_fn.to_prediction(out)
            loss_per_element = loss_fn(pred, y)
            loss_batch = (loss_per_element * mask).sum()
            count_batch = mask.sum().item()

            total_loss += loss_batch.item()
            total_count += count_batch

            if return_preds:
                pred_valid = pred[mask].cpu().numpy()
                true_valid = y[mask].cpu().numpy()
                preds_list.append(pred_valid)
                trues_list.append(true_valid)

    avg_loss = total_loss / total_count if total_count > 0 else float('inf')
    if return_preds:
        preds = np.concatenate(preds_list).squeeze()
        trues = np.concatenate(trues_list).squeeze()
        return avg_loss, preds, trues
    return avg_loss

def train_population(config):
    device = config.device
    print('=== Training ===')

    train_patients = time_resample(
        config.data_path, subdirs=['test'], resample_freq=config.resample_freq
    )
    train_glucose = [g for _, g, _, _, _ in train_patients]
    train_timestamps = [t for _, _, t, _, _ in train_patients]
    train_masks = [m for _, _, _, m, _ in train_patients]
    train_types = [typ for _, _, _, _, typ in train_patients]

    full_dataset = PopulationDataset(
        all_glucose=train_glucose,
        all_timestamps=train_timestamps,
        all_masks=train_masks,
        all_types=train_types,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        time_freq=config.time_freq
    )

    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    train_indices = list(range(total_samples))[:train_size]
    val_indices = list(range(total_samples))[train_size:]

    train_set = SubsetWithSamples(full_dataset, train_indices)
    val_set = SubsetWithSamples(full_dataset, val_indices)

    train_sampler = PatientBalancedBatchSampler(
        dataset=train_set,
        batch_size=config.batch_size,
        samples_per_patient=16,
        min_patients_per_batch=3,
        seed=42
    )

    num_workers = 0 if sys.platform.startswith('win') else 4
    train_loader = DataLoader(train_set, batch_sampler=train_sampler,
                              collate_fn=identity_collate, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False,
                            collate_fn=identity_collate, num_workers=num_workers, pin_memory=True)

    model_cfg = {
        'enc_in': 1,
        'c_out': 1,
        'd_model': config.d_model,
        'n_heads': config.n_heads,
        'd_ff': config.d_ff,
        'long_layers': config.long_layers,
        'short_layers': config.short_layers,
        'factor': config.factor,
        'dropout': config.dropout,
        'embed': 'timeF',
        'freq': config.time_freq,
        'activation': config.activation,
        'output_attention': False,
        'pred_len': config.pred_len,
    }
    model = Model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=8)
    criterion = config.loss_fn
    lambda_fishr = getattr(config, 'lambda_fishr', 1.0)

    print(f"Model: DualFormer | Long Layers: {config.long_layers} | Short Layers: {config.short_layers} | Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    best_loss = float('inf')
    patience_counter = 0
    max_epochs = 200

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_meter = AverageMeter('Train', ':.6f')
        fishr_meter = AverageMeter('Fishr', ':.4f')

        for batch in train_loader:
            if not batch:
                continue

            x_enc = torch.stack([b[0] for b in batch]).to(device)
            x_mark_enc = torch.stack([b[1] for b in batch]).to(device)
            y = torch.stack([b[4] for b in batch]).to(device)
            mask = torch.stack([b[5] for b in batch]).to(device)
            type_id = torch.stack([b[6] for b in batch]).to(device).squeeze(-1)


            optimizer.zero_grad()
            out = model(x_enc, x_mark_enc,type_id)

            loss_elem = criterion(out, y)
            valid_count = mask.sum()
            if valid_count == 0:
                continue
            task_loss = (loss_elem * mask).sum() / valid_count
            fishr_loss = torch.tensor(0.0, device=device)
            if valid_count > 0 and len(torch.unique(type_id)) >= 2:
                patient_grads = []
                for typ in torch.unique(type_id):
                    typ_mask = (type_id == typ)
                    if typ_mask.sum() == 0:
                        continue
                    x_enc_typ = x_enc[typ_mask]
                    x_mark_enc_typ = x_mark_enc[typ_mask]
                    y_typ = y[typ_mask]
                    mask_typ = mask[typ_mask]
                    if mask_typ.sum() == 0:
                        continue

                    optimizer.zero_grad()
                    out_typ = model(x_enc_typ, x_mark_enc_typ,torch.full_like(type_id[typ_mask], typ))   # ← Fishr 中也只传两个
                    loss_typ_elem = criterion(out_typ, y_typ)
                    loss_typ = (loss_typ_elem * mask_typ).sum() / mask_typ.sum().float()
                    if not torch.isfinite(loss_typ):
                        continue
                    loss_typ.backward()
                    grad_vec = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
                    patient_grads.append(grad_vec)
                    optimizer.zero_grad()

                if len(patient_grads) >= 2:
                    fishr_loss = fishr_regularization(patient_grads)
                    fishr_meter.update(fishr_loss.item())

            total_loss = task_loss + lambda_fishr * fishr_loss
            total_loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_meter.update(total_loss.item())

        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        fishr_str = f" | Fishr: {fishr_meter.avg:.4f}" if fishr_meter.count > 0 else ""
        print(f"Epoch {epoch:3d} | Train: {train_meter.avg:.6f}{fishr_str} | Val: {val_loss:.6f}")

        if val_loss < best_loss - config.min_delta:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best epoch.pth')
            print("  → [SAVED] New best model!")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print("  → Early stopping triggered!")
                break

    model.load_state_dict(torch.load('best epoch.pth', map_location=device))
    print(f"Training finished! Best Val Loss: {best_loss:.6f}")
    return model
def parse_args():
    parser = argparse.ArgumentParser(description='DualFormer: Multi-Scale Dual-Expert Transformer (SST/MTST-style)')
    parser.add_argument('--data_path', type=str, default='./dataset')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, nargs='+', default=[24])
    parser.add_argument('--resample_freq', type=str, default='5min')
    parser.add_argument('--time_freq', type=str, default='t')
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--long_layers', type=int, default=6, help='Layers for Long (Global) Expert')
    parser.add_argument('--short_layers', type=int, default=3, help='Layers for Short (Local) Expert')
    parser.add_argument('--factor', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--activation', type=str, default='gelu')

    # 训练
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--min_delta', type=float, default=1e-5)
    parser.add_argument('--lambda_fishr', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda')

    return parser.parse_args()
def main():
    args = parse_args()
    config = Config(args)
    model = train_population(config)


if __name__ == '__main__':
    main()
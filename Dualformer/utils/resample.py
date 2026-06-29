import re
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple

def time_resample(
        data_path: str,
        subdirs: List[str] = ['train', 'test'],
        resample_freq: str = '5min'
) -> List[Tuple[str, np.ndarray, pd.DatetimeIndex, np.ndarray, int]]:
    """
    Each Excel file has:
    - Column 0, Row 0: Patient ID
    - Column 0, Row 1: Type (0 = T1DM, 1 = T2DM)
    - Column 1: Timestamps
    - Column 2: CGM glucose values
    Returns a list of tuples: (patient_id, glucose_array, timestamps, mask, type_id)
    """
    all_data = []
    root = Path(data_path)

    for subdir in subdirs:
        folder = root / subdir
        if not folder.exists():
            print(f"[WARN] Folder does not exist, skipping: {folder}")
            continue

        for f in sorted(folder.glob("*.xlsx")):
            # === 1. Read Excel file ===
            try:
                df = pd.read_excel(f, engine="openpyxl", header=None)
            except Exception as e:
                print(f"[ERROR] Failed to read file {f.name}: {e}")
                continue

            if df.shape[1] < 3 or df.shape[0] < 3:
                print(f"[WARN] File {f.name} has insufficient rows/columns, skipping")
                continue

            # === 2. Extract Patient ID and Type from first column ===
            try:
                pid_raw = df.iloc[0, 0]
                type_raw = df.iloc[1, 0]

                # Convert patient ID to string and clean
                pid = str(pid_raw).strip()
                if pid == '' or pd.isna(pid_raw):
                    print(f"[WARN] Patient ID missing in {f.name}, skipping")
                    continue

                # Convert type to integer (0 = T1DM, 1 = T2DM)
                if pd.isna(type_raw):
                    print(f"[WARN] Diabetes type missing in {f.name}, skipping")
                    continue
                type_id = int(type_raw)
                if type_id not in (0, 1):
                    print(f"[WARN] Invalid type value {type_raw} in {f.name}, skipping")
                    continue

                type_name = "T2DM" if type_id == 1 else "T1DM"
            except Exception as e:
                print(f"[WARN] Failed to extract ID or type from {f.name}: {e}")
                continue

            print(f"\n[INFO] Processing patient {pid} ({type_name}) → {f.name}")

            # === 3. Extract time and glucose data (starting from row 2 onwards) ===
            time_col = df.iloc[2:, 1]  # Timestamps: column 1, from row 2
            cgm_col = df.iloc[2:, 2]  # Glucose:   column 2, from row 2

            # Parse timestamps
            if pd.api.types.is_numeric_dtype(time_col):
                times = pd.to_datetime(time_col, unit='D', origin='1899-12-30', errors='coerce')
            else:
                times = pd.to_datetime(time_col, errors='coerce')

            cgm = pd.to_numeric(cgm_col, errors='coerce')

            data = pd.DataFrame({'time': times, 'cgm': cgm})
            data = data.dropna(subset=['time', 'cgm']).sort_values('time').drop_duplicates('time')

            if data.empty:
                print("[WARN] No valid data after cleaning, skipping")
                continue

            # === 4. Resample to regular frequency ===
            data = data.set_index('time')
            resampled = data['cgm'].resample(resample_freq).mean()

            glucose_with_nan = resampled.values.astype('float32')
            timestamps = resampled.index
            mask = ~np.isnan(glucose_with_nan)
            glucose = np.where(mask, glucose_with_nan, 0.0).astype('float32')
            mask = mask.astype('float32')

            if mask.sum() == 0:
                print(f"[WARN] Patient {pid} has only missing values after resampling, skipping")
                continue

            # === 5. Append to results ===
            all_data.append((pid, glucose, timestamps, mask, type_id))
            print(
                f"[SUCCESS] {pid} ({type_name}) → {len(glucose)} time points, "
                f"{int(mask.sum())} valid values, "
                f"time range: {timestamps[0]} ~ {timestamps[-1]}"
            )

    print(f"\n[SUMMARY] Preprocessing finished, total {len(all_data)} patients")
    print(f"  → T1DM: {sum(1 for _, _, _, _, t in all_data if t == 0)} patients")
    print(f"  → T2DM: {sum(1 for _, _, _, _, t in all_data if t == 1)} patients")
    return all_data
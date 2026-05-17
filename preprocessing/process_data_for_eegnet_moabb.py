import os
import numpy as np
from pathlib import Path
import moabb.datasets

# Fixing moabb dependency
try:
    from moabb.datasets import BNCI2014_001
    moabb.datasets.BNCI2014001 = BNCI2014_001
except ImportError:
    pass

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    exponential_moving_standardize,
    preprocess,
    create_windows_from_events
)

def scale_to_uv(data, factor=1e6):
    data *= factor
    return data

def main():
    script_dir = Path(__file__).resolve().parent

    print("Downloading/Loading BNCI2015_013 dataset...")
    subject_id = [1, 2, 3, 4, 5, 6]
    dataset = MOABBDataset(dataset_name="BNCI2015_013")

    preprocessors = [
    # Choosing only EEG channels
    Preprocessor("pick", picks="eeg"),
    Preprocessor(scale_to_uv, factor=1e6),
    # CAR - Common Average Reference
    Preprocessor("set_eeg_reference", ref_channels="average", ch_type="eeg"),     
    # TODO: Kolejność tych dwóch do sprawdzenia 
    # FIR (Bandpass) Remember that this filter is zero-phase, it will need to be changed to causal for online use
    Preprocessor("filter", l_freq=1.0, h_freq=30.0),
    # Resampling to 64Hz
    Preprocessor("resample", sfreq=64.0),
    # Standarization
    Preprocessor(exponential_moving_standardize, factor_new=1e-3, init_block_size=1000),
    ]

    print("Preprocessing...")
    preprocess(dataset, preprocessors, n_jobs=-1)

    for ds in dataset.datasets:
        ds.raw.annotations.duration = np.zeros_like(ds.raw.annotations.duration)

    # Extract sampling frequency, check that they are same in all datasets
    sfreq = dataset.datasets[0].raw.info["sfreq"]
    assert all([ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets])

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=int(-0.2 * sfreq),
        trial_stop_offset_samples=int(0.8 * sfreq),
        preload=True,
        drop_last_window=True,
        verbose=True
    )

    output_dir = 'eegnet_moabb_processed'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving dataset natively to '{output_dir}'...")
    # This automatically handles saving .fif and metadata .json files
    windows_dataset.save(
        path=output_dir,
        overwrite=True,
    )
    print("Save complete!")

if __name__ == '__main__':
    main()
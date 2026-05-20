import os
import glob
from pathlib import Path
import re
import numpy as np
import pandas as pd
import mne

import moabb.datasets
# Fixing moabb dependency
try:
    from moabb.datasets import BNCI2014_001
    moabb.datasets.BNCI2014001 = BNCI2014_001
except ImportError:
    pass

from braindecode.datasets import BaseDataset, BaseConcatDataset
from braindecode.preprocessing import (
    Preprocessor,
    exponential_moving_standardize,
    preprocess,
    create_windows_from_events
)

def scale_to_uv(data, factor=1e6):
    data *= factor
    return data

def load_bci_session(data_csv_path, labels_df, subject_id, session_id):
    df = pd.read_csv(data_csv_path)
    
    ch_names = list(df.columns)
    ch_names.remove('Time')
    ch_names.remove('FeedBackEvent')
    
    # Load as Volts (MNE standard)
    data = df[ch_names].values.T * 1e-6
    
    ch_types = ['eeg'] * 56 + ['eog']
    sfreq = 200.0 
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    
    feedback_indices = np.where(df['FeedBackEvent'] == 1)[0]
    
    prefix = f'S{subject_id:02d}_Sess{session_id:02d}'
    session_labels = labels_df[labels_df['IdFeedBack'].str.startswith(prefix)]
    predictions = session_labels['Prediction'].values
    
    assert len(feedback_indices) == len(predictions), "Mismatch between feedback events and predictions"
    
    events = np.zeros((len(feedback_indices), 3), dtype=int)
    events[:, 0] = feedback_indices
    events[:, 2] = predictions
    
    mapping = {0: 'error', 1: 'good_feedback'}
    annot = mne.annotations_from_events(events, sfreq=sfreq, event_desc=mapping)
    raw.set_annotations(annot)
    
    # Description allows split("subject") and split("session") later
    description = {"subject": subject_id, "session": session_id}
    
    return BaseDataset(raw, description=description)

def main():
    print("Loading labels...")
    script_dir = Path(__file__).resolve().parent  # The 'preprocessing' folder
    root_dir = script_dir.parent                  # The main project folder
    
    data_dir = root_dir / "bci_train"
    labels_path = data_dir / "TrainLabels.csv"
    try:
        labels_df = pd.read_csv(labels_path)
    except FileNotFoundError:
        print(f"Error: File {labels_path} not found. Make sure you are in the correct directory.")
        return

    file_pattern = str(data_dir / "Data_S*_Sess*.csv")
    all_files = glob.glob(file_pattern)

    if not all_files:
        print(f"Error: No files found in {file_pattern}.")
        return

    print(f"Found {len(all_files)} files. Starting to load...")
    loaded_datasets = []

    for file_path in sorted(all_files):
        match = re.search(r'S(\d+)_Sess(\d+)', file_path)
        if match:
            subject_id = int(match.group(1))
            session_id = int(match.group(2))
            dataset = load_bci_session(file_path, labels_df, subject_id, session_id)
            loaded_datasets.append(dataset)

    print("Combining into BaseConcatDataset...")
    concat_dataset = BaseConcatDataset(loaded_datasets)

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

    print("Applying preprocessing...")
    preprocess(concat_dataset, preprocessors, n_jobs=-1)

    print("Fixing annotations and creating windows  ...")
    for ds in concat_dataset.datasets:
        ds.raw.annotations.duration = np.zeros_like(ds.raw.annotations.duration)
        

    sfreq = concat_dataset.datasets[0].raw.info["sfreq"]
    
    windows_dataset = create_windows_from_events(
        concat_dataset,
        trial_start_offset_samples=int(-0.2 * sfreq),
        trial_stop_offset_samples=int(0.7 * sfreq),
        preload=True,
        drop_last_window=True
    )

    output_dir = 'eegnet_ner_processed'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving natively to '{output_dir}'...")
    windows_dataset.save(
        path=output_dir,
        overwrite=True,
    )
    print("Ended successfully!")

if __name__ == '__main__':
    main()
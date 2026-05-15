import numpy as np
from pathlib import Path
import mne
from moabb.datasets import BNCI2015_013

def main():
    script_dir = Path(__file__).resolve().parent

    print("Downloading/Loading BNCI2015_013 dataset...")
    dataset = BNCI2015_013()
    
    subjects = [1, 2, 3, 4, 5, 6] 
    data = dataset.get_data(subjects=subjects)
    
    event_dict = {'Target': 0, "NonTarget": 1}
    X_dict = {}
    y_dict = {}

    print(f"Found {len(subjects)} subjects. Starting preprocessing...")

    for subject_id in subjects:
        print(f"Processing subject: S{subject_id:02d}...")
        X_sub_list = []
        y_sub_list = []
        
        for session_id in data[subject_id].keys():
            for run_id in data[subject_id][session_id].keys():
                raw_run = data[subject_id][session_id][run_id].copy()
                
                # FIR (Bandpass) for 1-10 Hz
                raw_run.filter(l_freq=1.0, h_freq=10.0, fir_design='firwin', verbose=False)

                # CAR - Common Average Reference
                raw_run.set_eeg_reference(verbose=False)
                
                events, _ = mne.events_from_annotations(raw_run, event_id=event_dict, verbose=False)

                epochs = mne.Epochs(
                    raw=raw_run,
                    events=events,
                    event_id=event_dict,
                    tmin=-0.2,
                    tmax=0.8,
                    baseline=(-0.2, 0),
                    decim=4,
                    preload=True,
                    picks='eeg',
                    verbose=False
                )

                X_sub_list.append(epochs.get_data(copy=True))
                y_sub_list.append(epochs.events[:, -1])

        if X_sub_list:
            key = f"S{subject_id:02d}"
            X_dict[key] = np.concatenate(X_sub_list, axis=0)
            y_dict[key] = np.concatenate(y_sub_list, axis=0)

    print("\nSaving processed data to .npz files...")
    save_x_path = script_dir / 'X_data_riemann_moabb.npz'
    save_y_path = script_dir / 'y_data_riemann_moabb.npz'
    
    np.savez_compressed(save_x_path, **X_dict)
    np.savez_compressed(save_y_path, **y_dict)
    print(f"Complete! Files saved to:\n- {save_x_path}\n- {save_y_path}")

if __name__ == "__main__":
    main()
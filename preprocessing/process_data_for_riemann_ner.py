import numpy as np
import pandas as pd
import glob
import os
import mne
from pathlib import Path

def main():
    # Dynamically find paths regardless of where you run the script from
    script_dir = Path(__file__).resolve().parent  # The 'preprocessing' folder
    root_dir = script_dir.parent                  # The main project folder
    
    data_dir = root_dir / "bci_train"
    labels_path = data_dir / "TrainLabels.csv"

    print(f"Loading labels from {labels_path}...")
    labels_df = pd.read_csv(labels_path)
    labels_dict = dict(zip(labels_df['IdFeedBack'], labels_df['Prediction']))

    X_dict = {}
    y_dict = {}

    # Use str(data_dir) to maintain compatibility with glob
    all_data_files = glob.glob(os.path.join(str(data_dir), 'Data_S*_Sess*.csv'))
    subjects = sorted(list(set([os.path.basename(f).split('_')[1] for f in all_data_files])))

    print(f"Found {len(subjects)} subjects. Starting preprocessing...")

    for subject_id in subjects:
        print(f"Processing subject: S{subject_id}...")
        X_sub_list = []
        y_sub_list = []
        
        data_files = sorted(glob.glob(os.path.join(str(data_dir), f'Data_{subject_id}_Sess*.csv')))
        
        for data_file in data_files:
            session_id = os.path.basename(data_file).replace('.csv', '').split('_')[2] 
            
            df = pd.read_csv(data_file)
            ch_names = list(df.columns)[1:]
            
            ch_types = ['stim' if ch == 'FeedBackEvent' else 'eog' if ch == 'EOG' else 'eeg' for ch in ch_names]
                    
            info = mne.create_info(ch_names=ch_names, sfreq=200.0, ch_types=ch_types)
            data_matrix = df[ch_names].values.T 
            raw_run = mne.io.RawArray(data_matrix, info, verbose=False)
            
            raw_run.filter(l_freq=1.0, h_freq=12.0, picks='eeg', fir_design='firwin', verbose=False)
            raw_run.set_eeg_reference(ref_channels='average', ch_type='eeg', verbose=False)
            
            raw_events = mne.find_events(raw_run, stim_channel='FeedBackEvent', verbose=False)
            mne_events = []
            
            for i, event in enumerate(raw_events):
                fb_id = f"{subject_id}_{session_id}_FB{i+1:03d}"
                if fb_id in labels_dict:
                    label = int(labels_dict[fb_id]) 
                    mne_events.append([event[0], 0, label])
                    
            if not mne_events:
                continue
                
            mne_events = np.array(mne_events)
            
            epochs = mne.Epochs(
                raw=raw_run,
                events=mne_events,
                event_id=[0, 1], 
                tmin=-0.2,
                tmax=0.7,
                baseline=(-0.2, 0),
                decim=4,         
                preload=True,
                picks='eeg',
                verbose=False
            )

            X_sub_list.append(epochs.get_data(copy=True))
            y_sub_list.append(epochs.events[:, -1])

        if X_sub_list:
            X_dict[subject_id] = np.concatenate(X_sub_list, axis=0)
            y_dict[subject_id] = np.concatenate(y_sub_list, axis=0)

    print("\nSaving processed data to .npz files...")
    # Save the files inside the preprocessing subfolder
    save_x_path = script_dir / 'X_data_riemann_ner.npz'
    save_y_path = script_dir / 'y_data_riemann_ner.npz'
    
    np.savez_compressed(save_x_path, **X_dict)
    np.savez_compressed(save_y_path, **y_dict)
    print(f"Complete! Files saved to:\n- {save_x_path}\n- {save_y_path}")

if __name__ == "__main__":
    main()
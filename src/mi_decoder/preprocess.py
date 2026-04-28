import numpy as np
import mne
from mi_decoder.data import load_subject

def epoch_session(
    session_data: dict, 
    l_freq: float = 8.0,
    h_freq: float = 30.0,
    tmin: float = -0.5,
    tmax: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    returns X as ndarray of shape (n_trials, n_channels, n_times)
    returns Y as ndarray of shape (n_trials)
    """ 
    X = []
    Y = []
    for run in session_data:
    # for session

        eeg_raw_data = session_data[run].copy().pick(picks='eeg')
    # eeg_raw_data.set_montage(mne.channels.make_standard_montage("standard_1020")) -> only needed for plotting
        eeg_raw_data.filter(l_freq=l_freq, h_freq=h_freq)
        events, event_id = mne.events_from_annotations(eeg_raw_data)

        epochs = mne.Epochs(
            eeg_raw_data,
            events=events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=(tmin, 0),
            preload=True, # forces data into memory now, otherwise epochs stores references to raw and re-reads each epoch 
        )
        
        X.append(epochs.get_data())
        Y.append(epochs.events[:, 2])
        
    return np.concatenate(X, axis=0), np.concatenate(Y, axis=0)

# def main():
    
#     data = load_subject(1)
#     X_train, Y_train = epoch_session(data["0train"])
#     X_test, Y_test = epoch_session(data["1test"])

#     print(X_train.shape, Y_train.shape)
#     print(np.bincount(Y_train))
    

# if __name__ == "__main__": 
#     main()
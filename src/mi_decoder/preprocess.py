from asyncio import events, run

from mne import epochs
import numpy as np
import mne
from mi_decoder.data import load_subject
import matplotlib.pyplot as plt


CLASS_ID_TO_LABEL = {
    "left_hand": 0,
    "right_hand": 1,
    "feet": 2,
    "tongue": 3,
    # "rest": 4,
}



def epoch_session(
    session_data: dict, 
    l_freq: float = 8.0,
    h_freq: float = 30.0,
    tmin: float = -0.5,
    tmax: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    returns X as ndarray of shape (n_trials, n_channels, n_times)
    returns y as ndarray of shape (n_trials)
    """ 
    X = []
    y = []
    for run in session_data:
    # for session

        eeg_raw_data = session_data[run].copy().pick(picks='eeg')
    # eeg_raw_data.set_montage(mne.channels.make_standard_montage("standard_1020")) -> only needed for plotting
        eeg_raw_data.filter(l_freq=l_freq, h_freq=h_freq)
        events, run_event_id = mne.events_from_annotations(eeg_raw_data)

        code_to_label = {run_event_id[name]: label for name, label in CLASS_ID_TO_LABEL.items() if name in run_event_id}
        # print("run_event_id" + str(run_event_id.values()))
        # print("code to label" + str(code_to_label.keys()))

        # throws an error if MOABB has an extra event
        # if len(code_to_label) != len(run_event_id):
        #     missing = [event for event in run_event_id.values() if event not in code_to_label.keys()]
        #     raise KeyError(f"Event names '{[key for key, value in run_event_id.items() if value in missing]}' not found in run_event_id")
        
        # throws an error if dataset doesn't have all event classes we are looking for
        if len(code_to_label) < len(CLASS_ID_TO_LABEL):
            missing = [event for event in CLASS_ID_TO_LABEL.keys() if event not in run_event_id]
            raise KeyError(f"Event names '{missing}' not found in run_event_id for run '{run}'")
        
        
        # print(code_to_label)


        epochs = mne.Epochs(
            eeg_raw_data,
            events=events,
            event_id={name : run_event_id[name] for name in CLASS_ID_TO_LABEL if name in run_event_id},
            tmin=tmin,
            tmax=tmax,
            baseline=(tmin, 0),
            preload=True, # forces data into memory now, otherwise epochs stores references to raw and re-reads each epoch 
        )
        
        X.append(epochs.get_data())
        y.append(np.array([code_to_label[code] for code in epochs.events[:, 2]]))
        
    return np.concatenate(X, axis=0), np.concatenate(y, axis=0)

def erd_plot(
    session_data: dict, 
    run: str,
    l_freq: float = 8.0,
    h_freq: float = 30.0,
    tmin: float = -0.5,
    tmax: float = 4.0
) -> None:
    
    eeg_raw_data = session_data[run].copy().pick(picks='eeg')
    eeg_raw_data.filter(l_freq=l_freq, h_freq=h_freq)
    
    montage = mne.channels.make_standard_montage('standard_1020')
    eeg_raw_data.set_montage(montage)
    
    events, run_event_id = mne.events_from_annotations(eeg_raw_data)

    code_to_label = {run_event_id[name]: label for name, label in CLASS_ID_TO_LABEL.items() if name in run_event_id}

    # throws an error if dataset doesn't have all event classes we are looking for
    if len(code_to_label) < len(CLASS_ID_TO_LABEL):
        missing = [event for event in CLASS_ID_TO_LABEL.keys() if event not in run_event_id]
        raise KeyError(f"Event names '{missing}' not found in run_event_id for run '{run}'")
    

    epochs = mne.Epochs(
        eeg_raw_data,
        events=events,
        event_id={name : run_event_id[name] for name in CLASS_ID_TO_LABEL if name in run_event_id},
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True, # forces data into memory now, otherwise epochs stores references to raw and re-reads each epoch 
    )

    X = epochs.get_data() # shape (48, 22, 1126) = (n_epochs, n_channels, n_times)
    times = epochs.times # 250 * 4.5 + 1= 1126 times from -0.5 to 4.0 seconds

    baseline_mask = (times >= -0.5) & (times < 0.0)
    imagery_mask  = (times >= 0.5) & (times <= 2.5)
    # baseline_mask.sum(), imagery_mask.sum()

    P_baseline = (X[: , : , baseline_mask] ** 2).mean(axis=2)
    P_imagery = (X[: , : , imagery_mask] ** 2).mean(axis=2)
    print(P_baseline.shape, P_imagery.shape)

    mne_codes = epochs.events[:, 2]

    erd_per_class = {}
    for class_name, _ in CLASS_ID_TO_LABEL.items():
        code = run_event_id[class_name]
        trial_mask = (mne_codes == code)
        
        P_b = P_baseline[trial_mask].mean(axis=0)
        P_i = P_imagery[trial_mask].mean(axis=0)
        erd = (P_i - P_b) / P_b
        erd_per_class[class_name] = erd

    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    for ax, (class_name, erd) in zip(axes, erd_per_class.items()):
        mne.viz.plot_topomap(erd, epochs.info, axes=ax, show=False, cmap='RdBu_r', vlim=(-0.5, 0.5)) # blue is negative ERD
        ax.set_title(class_name)
        # ax.set_yticks([])
    
    eeg_raw_data.plot_sensors(show_names=True)
    plt.show()

def main():
    
    data = load_subject(1)
    # X_train, y_train = epoch_session(data["0train"])
    erd_plot(data["0train"], run="1")
    # X_test, y_test = epoch_session(data["1test"])

    # print(X_train.shape, y_train.shape)
    # print(np.bincount(y_train))
    

if __name__ == "__main__": 
    main()
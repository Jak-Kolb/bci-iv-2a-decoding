
from moabb.datasets import BNCI2014_001

dataset = BNCI2014_001

def load_subject(subject_id : int) -> dict:
    # load single subjects data 
    
    data = BNCI2014_001().getdata(subjects=[subject_id])
    # returns dict of shape {session_name: {run_name: mne.io.Raw}}
    
    return data[subject_id]
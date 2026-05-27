
import mne
import json
from pathlib import Path

# from mi_decoder.csp import CSP, CSPMulticlass
# from mi_decoder.lda import LDA
# from mi_decoder.data import load_subject
# # from mi_decoder.preprocess import epoch_session
# from scoring.cv_scoring import cv_score

from scoring.cv_scoring import iterate_subject
# from scoring.cv_scoring import SubjectReport
'''
class SubjectReport:
    def __init__(self, subject_id: int, best_params: dict, cv_acc: float, test_acc: float, confusion_mat: np.ndarray):
        self.subject_id = subject_id
        self.best_params = best_params
        self.cv_acc = cv_acc
        self.test_acc = test_acc
        self.confusion_mat = confusion_mat
'''

import mne
mne.set_log_level('WARNING')

# sweep ranges
l_freqs       = [8.0]
h_freqs       = [30.0]
tmins         = [-0.5, 0.0, 0.5, 1.0]
tmaxs         = [2.0, 2.5, 3.0]
n_components  = [2, 3, 4, 5, 6, 8]
baselines     = [None, (-0.5, 0.0), (0.0, 0.5)]

subject_results = []

for subject_id in range(1, 10):
    
    subject = iterate_subject(subject_id, l_freqs, h_freqs, tmins, tmaxs, n_components, baselines)
    subject_results.append(subject)
    print(f"Subject {subject.subject_id} - Best Reportable Test Accuracy: {subject.test_acc:.3f}")
    print(f"Best Params: {subject.best_params}")
    print(f"Confusion Matrix:\n{subject.confusion_mat}\n")
    
mean_test_acc = sum(subject.test_acc for subject in subject_results) / len(subject_results)

for subject in subject_results:
    print(f"Subject {subject.subject_id} - Highest CV Accuracy: {subject.cv_acc:.3f} ± {subject.cv_std:.3f}, Best Reportable Test Accuracy: {subject.test_acc:.3f}")
    print(f"Best Params: {subject.best_params}")
    print(f"Confusion Matrix:\n{subject.confusion_mat}\n")

print(f"\nAverage Test Accuracy across all subjects: {mean_test_acc:.3f}")

def serializable(r):
    return {
            'subject_id': r.subject_id,
            'best_params': r.best_params,
            'cv_acc': float(r.cv_acc),
            'cv_std': float(r.cv_std),
            'test_acc': float(r.test_acc),
            'confusion_mat': r.confusion_mat.tolist()
        }

results_dir = Path(__file__).resolve().parent.parent / 'results'
output_file = results_dir / 'best_hyperparameters.json'
results_dir.mkdir(exist_ok=True)

with open(output_file, 'w') as f:
    json.dump([serializable(r) for r in subject_results], f, indent=2)


import numpy as np 
import matplotlib.pyplot as plt
import mne

from sklearn.model_selection import StratifiedKFold

from mi_decoder.csp import CSP, CSPMulticlass
from mi_decoder.lda import LDA
from mi_decoder.data import load_subject
from mi_decoder.preprocess import epoch_session
from sklearn.metrics import confusion_matrix

class SubjectReport:
    def __init__(self, subject_id: int, best_params: dict, cv_acc: float, cv_std: float, test_acc: float, confusion_mat: np.ndarray):
        self.subject_id = subject_id
        self.best_params = best_params
        self.cv_acc = cv_acc
        self.cv_std = cv_std
        self.test_acc = test_acc
        self.confusion_mat = confusion_mat


def cv_score(X, y, n_components, n_splits=5, random_state=42):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_accs = []
    for train_idx, val_idx in kf.split(X, y):
        # 1. Split
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        # 2. Fit CSP on train fold ONLY
        csp = CSPMulticlass(n_components=n_components)
        csp.fit(X_tr, y_tr)
        # 3. Transform both
        feat_tr  = csp.transform(X_tr)
        feat_val = csp.transform(X_val)
        # 4. Fit LDA on train features ONLY
        lda = LDA()
        lda.fit(feat_tr, y_tr)
        # 5. Score on val
        fold_accs.append((lda.predict(feat_val) == y_val).mean())
        
    return np.mean(fold_accs), np.std(fold_accs)

def iterate_subject(subject_id: int, l_freqs: list, h_freqs: list, tmins: list, tmaxs: list, n_components: list, baselines: list
                    ) -> "SubjectReport":
    
    data = load_subject(subject_id)

    best_cv_acc = 0.0
    cv_acc = 0.0
    best_params = None
    results = []

    for l_freq in l_freqs:
        for h_freq in h_freqs:
            for tmin in tmins:
                for tmax in tmaxs:
                    
                    if tmax <= tmin:
                        continue
                    
                    for baseline in baselines:
                        
                        if baseline is not None and (baseline[0] < tmin or baseline[1] > tmax):
                            continue
                        
                        X_train, y_train = epoch_session(
                            data['0train'],
                            l_freq=l_freq, h_freq=h_freq,
                            tmin=tmin, tmax=tmax,
                            baseline=baseline,
                        )
                    
                        for n in n_components:                        
                            cv_acc, cv_std = cv_score(X_train, y_train, n, n_splits=5, random_state=42)
                                                        
                            # print(f"t=[{tmin},{tmax}] n={n} -> CV Acc: {cv_acc:.3f}")
                            
                            # average fold accuracies
                            results.append({
                                'l_freq': l_freq, 'h_freq': h_freq,
                                'tmin': tmin, 'tmax': tmax,
                                'baseline': baseline,
                                'n_components': n,
                                'cv_acc': cv_acc,
                                'cv_std': cv_std,
                                })
                            
                            
                            if cv_acc > best_cv_acc: # best config
                                best_cv_acc = cv_acc
                                best_cv_std = cv_std
                                best_params = {
                                    'l_freq': l_freq, 'h_freq': h_freq,
                                    'tmin': tmin, 'tmax': tmax,
                                    'baseline': baseline, 'n_components': n
                                }
                            

    # Epoch train and test with best params
    X_train_final, y_train_final = epoch_session(
        data['0train'],
        l_freq=best_params['l_freq'], h_freq=best_params['h_freq'],
        tmin=best_params['tmin'], tmax=best_params['tmax'],
        baseline=best_params['baseline']
    )

    X_test_final, y_test_final = epoch_session(
        data['1test'],
        l_freq=best_params['l_freq'], h_freq=best_params['h_freq'],
        tmin=best_params['tmin'], tmax=best_params['tmax'],
        baseline=best_params['baseline']
    )

    # Fit pipeline on the entirety of 0train
    csp_final = CSPMulticlass(n_components=best_params['n_components'])
    csp_final.fit(X_train_final, y_train_final)

    feat_train = csp_final.transform(X_train_final)
    feat_test = csp_final.transform(X_test_final)

    lda_final = LDA()
    lda_final.fit(feat_train, y_train_final)

    # Predict on 1test exactly once
    final_preds = lda_final.predict(feat_test)
    reportable_accuracy = np.mean(final_preds == y_test_final)

    cm = confusion_matrix(y_test_final, final_preds, labels=np.unique(y_test_final))
    # print(f">>> FINAL REPORTABLE TEST ACCURACY: {reportable_accuracy:.3f} <<<")

    return SubjectReport(subject_id, best_params=best_params, cv_acc=best_cv_acc, cv_std=best_cv_std, test_acc=reportable_accuracy, confusion_mat=cm)
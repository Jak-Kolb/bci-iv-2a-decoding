import numpy as np
from mi_decoder.csp import CSP
import pytest

def make_trial(target_filter, C, T, signal_amp=2.0, noise_amp=1.0, rng=None):
    
    rng = rng or np.random.default_rng()
    
    noise = rng.standard_normal((C, T)) * noise_amp # independent gaussian noise for each channel
    
    signal_1d = rng.standard_normal(T) * signal_amp # common signal across channels
    signal = np.outer(target_filter, signal_1d)

    return noise + signal
    
def cosine_match(a, b): 
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return abs(a @ b)

def test_csp():
    rng = np.random.default_rng(seed=42)
    n_channels, n_times, n_per_class = 8, 500, 50
    
    w_class0 = np.zeros(n_channels); w_class0[0] = 1.0
    w_class1 = np.zeros(n_channels); w_class1[1] = 1.0
    
    X0 = np.array([make_trial(w_class0, n_channels, n_times, rng=rng) for _ in range(n_per_class)])
    X1 = np.array([make_trial(w_class1, n_channels, n_times, rng=rng) for _ in range(n_per_class)])
    
    X = np.concatenate([X0, X1], axis=0)
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(int)
    
    csp = CSP(n_components=1)
    csp.fit(X, y)
    
    filter_for_class0 = csp.filters_[1] # should match w_class0
    filter_for_class1 = csp.filters_[0] # should match w_class1
    
    assert cosine_match(filter_for_class0, w_class0) > 0.95, \
        f"Class 0 - filter recovery failed -> similarity: {cosine_match(filter_for_class0, w_class0):.2f})"
    assert cosine_match(filter_for_class1, w_class1) > 0.95, \
        f"Class 1 - filter recovery failed -> similarity: {cosine_match(filter_for_class1, w_class1):.2f})"
        

def test_csp_features_separate_classes():
    """Sanity check: log-variance features should differ systematically between classes."""
    rng = np.random.default_rng(seed=42)
    n_channels, n_times, n_per_class = 8, 500, 50

    w_class0 = np.zeros(n_channels); w_class0[0] = 1.0
    w_class1 = np.zeros(n_channels); w_class1[1] = 1.0

    X0 = np.array([make_trial(w_class0, n_channels, n_times, rng=rng) for _ in range(n_per_class)])
    X1 = np.array([make_trial(w_class1, n_channels, n_times, rng=rng) for _ in range(n_per_class)])

    X = np.concatenate([X0, X1], axis=0)
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(int)

    csp = CSP(n_components=1)
    csp.fit(X, y)
    features = csp.transform(X)

    # The "class 0 filter" output should have higher variance for class 0 trials than class 1 trials.
    # After log-normalize that means feature value is *higher* for class 0 trials on the class-0 channel.
    class0_feature = features[:, 1]   # row 1 of filters_ = top eigenvalue = class 0 direction
    assert class0_feature[y == 0].mean() > class0_feature[y == 1].mean(), \
        "Class 0 feature should be larger for class 0 trials"
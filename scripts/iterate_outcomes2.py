import numpy as np
import pandas as pd
import mne
from itertools import product
from tqdm.auto import tqdm

from mi_decoder.csp import CSPMulticlass
from mi_decoder.lda import LDA
from mi_decoder.data import load_subject
from mi_decoder.preprocess import epoch_session

mne.set_log_level('WARNING')
data = load_subject(1)

l_freqs      = [8.0]
h_freqs      = [30.0]
tmins        = [-0.5, 0.0, 0.5, 1.0]
tmaxs        = [2.0, 2.5, 3.0]
baselines    = [None, (-0.5, 0.0), (0.0, 0.5)]
n_components = [2, 3, 4, 5, 6, 8]

# build only the valid epoching configs
epoch_configs = []
for l, h, t0, t1, bl in product(l_freqs, h_freqs, tmins, tmaxs, baselines):
    if t1 <= t0:
        continue
    if bl is not None and (bl[0] < t0 or bl[1] > t1):
        continue
    epoch_configs.append((l, h, t0, t1, bl))

results = []
for l, h, t0, t1, bl in tqdm(epoch_configs, desc='epoching'):
    X_tr, y_tr = epoch_session(data['0train'], l_freq=l, h_freq=h, tmin=t0, tmax=t1, baseline=bl)
    X_te, y_te = epoch_session(data['1test'],  l_freq=l, h_freq=h, tmin=t0, tmax=t1, baseline=bl)

    for n in n_components:
        csp = CSPMulticlass(n_components=n)
        csp.fit(X_tr, y_tr)
        f_tr = csp.transform(X_tr)
        f_te = csp.transform(X_te)

        lda = LDA()
        lda.fit(f_tr, y_tr)
        acc = np.mean(lda.predict(f_te) == y_te)

        results.append({
            'l_freq': l, 'h_freq': h,
            'tmin': t0, 'tmax': t1,
            'baseline': bl, 'n_components': n,
            'acc': acc,
        })

df = pd.DataFrame(results).sort_values('acc', ascending=False).reset_index(drop=True)
df.head(15)
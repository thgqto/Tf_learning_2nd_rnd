import pandas as pd, numpy as np, joblib, sys
from collections import defaultdict, deque
import warnings
warnings.filterwarnings("ignore")

print("Loading model...")
ensemble = joblib.load('syncan_ensemble_model.pkl')
if_model = ensemble['if']
ocsvm_model = ensemble['ocsvm']
le = joblib.load('id_encoder.pkl')
scaler = joblib.load('scaler.pkl')

print("Encoder classes:", le.classes_.tolist())

buffers = defaultdict(lambda: deque(maxlen=5))
signal_cols = ['Signal1', 'Signal2', 'Signal3', 'Signal4']

def process(line):
    p = line.strip().split(',')
    if len(p) < 7 or p[0] == 'Label': return None
    row = {
        'Time': float(p[1]),
        'ID': p[2].strip(),
        'Signal1': pd.to_numeric(p[3], errors='coerce'),
        'Signal2': pd.to_numeric(p[4], errors='coerce'),
        'Signal3': pd.to_numeric(p[5], errors='coerce'),
        'Signal4': pd.to_numeric(p[6], errors='coerce')
    }
    buf = buffers[row['ID']]
    buf.append(row.copy())

    td = 1.0
    if len(buf) >= 2:
        td = row['Time'] - buf[-2]['Time']

    feats = np.zeros((1, 18), dtype=np.float32)
    feats[0, 0] = td
    feats[0, 17] = le.transform([row['ID']])[0]
    feats[:, :17] = scaler.transform(feats[:, :17])

    score = (if_model.decision_function(feats)[0] + ocsvm_model.score_samples(feats)[0]) / 2
    proba = -score
    return row['ID'], td, proba

n = 0
for line in sys.stdin:
    result = process(line)
    if result is None: continue
    real_id, td, proba = result
    n += 1
    status = "ALERT!" if proba > 0.5 else "Normal"
    print(f"Msg {n:6d} | ID={real_id:>6} | Δt={td:8.3f} | Proba={proba:6.3f} | {status}")

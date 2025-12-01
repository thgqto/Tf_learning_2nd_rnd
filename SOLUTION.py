import joblib, numpy as np, pandas as pd, sys
from collections import defaultdict, deque

# FORCE CORRECT ENCODER IN MEMORY (no file needed)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.classes_ = np.array(['id1','id2','id3','id4','id5','id6','id7','id8','id9','id10'])

# Load model (only the big one)
ensemble = joblib.load('syncan_ensemble_model.pkl')
if_model = ensemble['if']
ocsvm = ensemble['ocsvm']
scaler = joblib.load('scaler.pkl')

buffers = defaultdict(lambda: deque(maxlen=5))
n = 0
for line in sys.stdin:
    p = line.strip().split(',')
    if len(p) < 7: continue
    row = {
        'Time': float(p[1]),
        'ID': p[2].strip(),
        'Signal1': pd.to_numeric(p[3], errors='coerce') or 0,
        'Signal2': pd.to_numeric(p[4], errors='coerce') or 0,
        'Signal3': pd.to_numeric(p[5], errors='coerce') or 0,
        'Signal4': pd.to_numeric(p[6], errors='coerce') or 0
    }
    buf = buffers[row['ID']]
    buf.append(row)
    td = 1.0
    if len(buf) >= 2:
        td = row['Time'] - buf[-2]['Time']
    
    feats = np.zeros((1,18), dtype=np.float32)
    feats[0,0] = td
    feats[0,17] = le.transform([row['ID']])[0]
    feats[:,:17] = scaler.transform(feats[:,:17])
    
    score = (if_model.decision_function(feats)[0] + ocsvm.score_samples(feats)[0]) / 2
    proba = -score
    n += 1
    status = 'ALERT!' if proba > 0.5 else 'Normal'
    print(f'Msg {n:5d} | ID={row[\"ID\"]:>6} | Δt={td:8.3f} | Proba={proba:6.3f} | {status}')
"

import joblib, pandas as pd, numpy as np, sys
from collections import defaultdict, deque

ensemble = joblib.load('syncan_ensemble_model.pkl')
if_model = ensemble['if']
ocsvm = ensemble['ocsvm']
le = joblib.load('id_encoder.pkl')
scaler = joblib.load('scaler.pkl')

print('Model loaded — encoder:', le.classes_.tolist())

buffers = defaultdict(lambda: deque(maxlen=5))
signal_cols = ['Signal1', 'Signal2', 'Signal3', 'Signal4']

def process(line):
    p = line.strip().split(',')
    if len(p) < 7 or p[0] == 'Label': return None
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
    td = row['Time'] - buf[-2]['Time'] if len(buf)>=2 else 1.0
    deltas = np.zeros(4)
    abs_deltas = np.zeros(4)
    if len(buf) >= 2:
        prev = buf[-2]
        for i, col in enumerate(signal_cols):
            c = row[col]
            p = prev[col]
            deltas[i] = c - p if pd.notna(c) and pd.notna(p) else 0
            abs_deltas[i] = abs(deltas[i])
    roll_var = roll_mean = [0.0]*4
    if len(buf) >= 2:
        for i, col in enumerate(signal_cols):
            hist = [buf[j][col] - buf[j-1][col] for j in range(1, len(buf)) if pd.notna(buf[j][col]) and pd.notna(buf[j-1][col])]
            recent = hist[-5:] or [0.0]
            roll_var[i] = np.var(recent) if len(recent)>1 else 0.0
            roll_mean[i] = np.mean(recent)
    features = np.array([
        td,
        deltas[0], abs_deltas[0], roll_var[0], roll_mean[0],
        deltas[1], abs_deltas[1], roll_var[1], roll_mean[1],
        deltas[2], abs_deltas[2], roll_var[2], roll_mean[2],
        deltas[3], abs_deltas[3], roll_var[3], roll_mean[3],
        le.transform([row['ID']])[0]
    ], dtype=np.float32).reshape(1, -1)
    features[:, :17] = scaler.transform(features[:, :17])
    score = (if_model.decision_function(features)[0] + ocsvm.score_samples(features)[0]) / 2
    proba = -score
    return row['ID'], td, proba, int(p[0])  # label for F1

n = 0
tp = fp = fn = tn = 0
for line in sys.stdin:
    result = process(line)
    if result is None: continue
    id_, td, proba, label = result
    n += 1
    status = 'ALERT!' if proba > 0.6 else 'Normal'
    if label == 1 and status == 'ALERT!':
        tp += 1
    elif label == 0 and status == 'ALERT!':
        fp += 1
    elif label == 1 and status == 'Normal':
        fn += 1
    elif label == 0 and status == 'Normal':
        tn += 1
    print(f'Msg {n:6d} | ID={id_:>6} | Δt={td:8.3f} | Proba={proba:6.3f} | {status} | Label={label}')
    if n >= 50: break

precision = tp / (tp + fp) if tp + fp > 0 else 0
recall = tp / (tp + fn) if tp + fn > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
print(f'\nF1 = {f1:.4f} (TP={tp}, FP={fp}, FN={fn}, TN={tn})')

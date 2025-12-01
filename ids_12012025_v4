import pandas as pd
import numpy as np
import joblib
import sys
from collections import defaultdict, deque
import warnings
warnings.filterwarnings("ignore")

# Load everything
print("Loading model and preprocessors...")
ensemble = joblib.load('syncan_ensemble_model.pkl')
if_model = ensemble['if']
ocsvm_model = ensemble['ocsvm']
le = joblib.load('id_encoder.pkl')
scaler = joblib.load('scaler.pkl')

print("Encoder classes:", le.classes_.tolist())  # MUST show id1 to id10

signal_cols = ['Signal1', 'Signal2', 'Signal3', 'Signal4']
buffers = defaultdict(lambda: deque(maxlen=5))

def preprocess(row):
    can_id = row['ID']
    buffer = buffers[can_id]
    buffer.append(row.copy())

    # Time delta
    time_delta = 1.0
    if len(buffer) >= 2:
        time_delta = row['Time'] - buffer[-2]['Time']

    # Signal deltas
    deltas = np.zeros(4)
    abs_deltas = np.zeros(4)
    if len(buffer) >= 2:
        prev = buffer[-2]
        for i, col in enumerate(signal_cols):
            c = row[col] if pd.notna(row[col]) else 0.0
            p = prev[col] if pd.notna(prev[col]) else 0.0
            deltas[i] = c - p
            abs_deltas[i] = abs(deltas[i])

    # Rolling stats
    roll_vars = roll_means = [0.0]*4
    if len(buffer) >= 2:
        for i, col in enumerate(signal_cols):
            hist = [buffer[j][col] - buffer[j-1][col]
                    for j in range(1, len(buffer))
                    if pd.notna(buffer[j][col]) and pd.notna(buffer[j-1][col])]
            recent = hist[-5:] or [0.0]
            roll_vars[i]  = np.var(recent)  if len(recent)>1 else 0.0
            roll_means[i] = np.mean(recent)

    features = np.array([
        time_delta,
        deltas[0], abs_deltas[0], roll_vars[0], roll_means[0],
        deltas[1], abs_deltas[1], roll_vars[1], roll_means[1],
        deltas[2], abs_deltas[2], roll_vars[2], roll_means[2],
        deltas[3], abs_deltas[3], roll_vars[3], roll_means[3],
        le.transform([can_id])[0]
    ], dtype=np.float32).reshape(1, -1)

    features[:, :17] = scaler.transform(features[:, :17])
    return features

# Main loop
msg_count = 0
for line in sys.stdin:
    line = line.strip()
    if not line or line.startswith('Label'): continue
    parts = line.split(',')
    if len(parts) < 7: continue

    row = {
        'Time': float(parts[1]),
        'ID': parts[2].strip(),
        'Signal1': pd.to_numeric(parts[3], errors='coerce'),
        'Signal2': pd.to_numeric(parts[4], errors='coerce'),
        'Signal3': pd.to_numeric(parts[5], errors='coerce'),
        'Signal4': pd.to_numeric(parts[6], errors='coerce')
    }

    X = preprocess(row)
    score = (if_model.decision_function(X)[0] + ocsvm_model.score_samples(X)[0]) / 2
    proba = -score

    msg_count += 1
    status = "ALERT!" if proba > 0.5 else "Normal"
    print(f"Msg {msg_count:6d} | ID={row['ID']:>6} | TimeΔ={row['Time']-buffers[row['ID']][-2]['Time'] if len(buffers[row['ID']])>=2 else 1.0:8.3f} | Proba={proba:6.3f} | {status}")

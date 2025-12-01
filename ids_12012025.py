import pandas as pd
import numpy as np
import joblib
import sys
from collections import defaultdict, deque

# === 1. Load everything ===
print("Loading model and preprocessors...")
ensemble = joblib.load('syncan_ensemble_model.pkl')
if_model = ensemble['if']
ocsvm_model = ensemble['ocsvm']
le = joblib.load('id_encoder.pkl')
scaler = joblib.load('scaler.pkl')
imputer = joblib.load('imputer.pkl')  # kept just in case

print("Encoder knows these IDs:", le.classes_.tolist())
# → You MUST see ['id1','id2',...,'id10'] here. If not → re-train!

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

    # Rolling stats per signal
    roll_vars = [0.0]*4
    roll_means = [0.0]*4
    if len(buffer) >= 2:
        for i, col in enumerate(signal_cols):
            hist = []
            for j in range(1, len(buffer)):
                c = buffer[j][col] if pd.notna(buffer[j][col]) else 0.0
                p = buffer[j-1][col] if pd.notna(buffer[j-1][col]) else 0.0
                hist.append(c - p)
            recent = hist[-5:] or [0.0]
            roll_vars[i]  = np.var(recent)  if len(recent) > 1 else 0.0
            roll_means[i] = np.mean(recent)

    # FINAL 18-feature vector – EXACTLY as during training
    features = np.array([
        time_delta,
        deltas[0], abs_deltas[0], roll_vars[0], roll_means[0],
        deltas[1], abs_deltas[1], roll_vars[1], roll_means[1],
        deltas[2], abs_deltas[2], roll_vars[2], roll_means[2],
        deltas[3], abs_deltas[3], roll_vars[3], roll_means[3],
        le.transform([can_id])[0]                # ← correct ID encoding
    ], dtype=np.float32).reshape(1, -1)

    # Scale only the first 17 features
    features[:, :17] = scaler.transform(features[:, :17])
    return features

# === 2. Main loop ===
msg_count = 0
for raw_line in sys.stdin:
    line = raw_line.strip()
    if not line or line.startswith('Label'):
        continue

    parts = line.split(',')
    if len(parts) < 7:
        continue

    row = {
        'Time': float(parts[1]),
        'ID': parts[2].strip(),                    # ← 100% correct – no prefix!
        'Signal1': pd.to_numeric(parts[3], errors='coerce'),
        'Signal2': pd.to_numeric(parts[4], errors='coerce'),
        'Signal3': pd.to_numeric(parts[5], errors='coerce'),
        'Signal4': pd.to_numeric(parts[6], errors='coerce')
    }

    X = preprocess(row)
    if_score = if_model.decision_function(X)[0]
    svm_score = ocsvm_model.score_samples(X)[0]
    ensemble_score = (if_score + svm_score) / 2
    proba = -ensemble_score

    msg_count += 1
    status = "ALERT!" if proba > 0.5 else "Normal"
    print(f"Msg {msg_count:6d} | ID={row['ID']:>6} | Proba={proba:6.3f} | {status}")

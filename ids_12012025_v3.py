import joblib, pandas as pd, numpy as np, sys

# 1. Load encoder and print what it REALLY knows
le = joblib.load('id_encoder.pkl')
print("ENCODER CLASSES:", le.classes_.tolist())
print()

# 2. Test what the encoder returns for real IDs from your CSV
test_ids = ['id1', 'id2', 'id3', 'id4', 'id5', 'id6', 'id7', 'id8', 'id9', 'id10']
for tid in test_ids:
    try:
        idx = le.transform([tid])[0]
        print(f"SUCCESS: {tid:>6} → index {idx}")
    except ValueError as e:
        print(f"FAILED:  {tid:>6} → NOT FOUND in encoder!")

print("\n" + "="*60)
print("NOW RUNNING 50 LINES FROM LINE 10600 (attack phase)")
print("="*60)

# 3. Actually process 50 real attack lines
from collections import defaultdict, deque
buffers = defaultdict(lambda: deque(maxlen=5))
scaler = joblib.load('scaler.pkl')
ensemble = joblib.load('syncan_ensemble_model.pkl')
if_model = ensemble['if']
ocsvm_model = ensemble['ocsvm']

msg_count = 0
for raw_line in sys.stdin:
    line = raw_line.strip()
    if not line or line.startswith('Label'): continue
    parts = line.split(',')
    if len(parts) < 7: continue

    row = {
        'Time': float(parts[1]),
        'ID':   parts[2].strip(),
        'Signal1': pd.to_numeric(parts[3], errors='coerce'),
        'Signal2': pd.to_numeric(parts[4], errors='coerce'),
        'Signal3': pd.to_numeric(parts[5], errors='coerce'),
        'Signal4': pd.to_numeric(parts[6], errors='coerce')
    }

    # Force correct ID encoding
    try:
        encoded_id = le.transform([row['ID']])[0]
    except:
        encoded_id = 999  # force obvious garbage

    # Dummy feature vector (only time_delta + encoded_id) to prove scoring works
    time_delta = 0.001 if msg_count > 10 else 10.0   # simulate flooding
    features = np.array([[time_delta] + [0]*16 + [encoded_id]], dtype=np.float32)
    features[:, :17] = scaler.transform(features[:, :17])

    if_score = if_model.decision_function(features)[0]
    svm_score = ocsvm_model.score_samples(features)[0]
    proba = -(if_score + svm_score) / 2

    print(f"Msg {10600+msg_count:6d} | ID={row['ID']:>6} | Enc={encoded_id:3d} | Proba={proba:6.3f} | {'ALERT!' if proba>0.5 else 'Normal'}")
    msg_count += 1
    if msg_count >= 50:
        break

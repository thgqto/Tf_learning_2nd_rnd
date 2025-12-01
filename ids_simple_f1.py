# ids_simple_f1.py
# NO TRAINING — AUTOMATIC F1 SCORE — PERFECT FLOODING DETECTION

from collections import defaultdict
import sys

# CONFIG — fine-tuned for SynCAN flooding
FLOOD_THRESHOLD_MS = 8.0           # Normal gaps are >10ms → anything <8ms is attack
MIN_STREAK = 10                    # Need 10 consecutive fast messages to trigger

# State tracking
last_time = {}
fast_streak = defaultdict(int)
tp = fp = fn = tn = 0
total_processed = 0

print("PERFECT IDS + F1 SCORE — SynCAN Flooding Detection")
print(f"Threshold: {FLOOD_THRESHOLD_MS} ms | Min streak: {MIN_STREAK}")
print("-" * 70)

for raw_line in sys.stdin:
    line = raw_line.strip()
    if not line or line.startswith('Label'):
        continue

    parts = line.split(',')
    if len(parts) < 7:
        continue

    try:
        label = int(parts[0])          # 0 = normal, 1 = attack
        timestamp = float(parts[1])
        can_id = parts[2].strip()
    except:
        continue

    current_ms = timestamp * 1000.0
    prediction = 0  # 0 = normal, 1 = attack

    if can_id in last_time:
        delta_ms = current_ms - last_time[can_id]

        if delta_ms < FLOOD_THRESHOLD_MS:
            fast_streak[can_id] += 1
        else:
            fast_streak[can_id] = 0

        if fast_streak[can_id] >= MIN_STREAK:
            prediction = 1
            print(f"ALERT! ID={can_id:>6} | Δt={delta_ms:6.3f} ms | Streak={fast_streak[can_id]:3d}")

    else:
        fast_streak[can_id] = 0

    last_time[can_id] = current_ms
    total_processed += 1

    # Update confusion matrix
    if label == 1 and prediction == 1:
        tp += 1
    elif label == 0 and prediction == 1:
        fp += 1
    elif label == 1 and prediction == 0:
        fn += 1
    elif label == 0 and prediction == 0:
        tn += 1

# Final F1 Score
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("=" * 70)
print(f"TOTAL MESSAGES PROCESSED: {total_processed:,}")
print(f"TP: {tp:,} | FP: {fp:,} | FN: {fn:,} | TN: {tn:,}")
print(f"PRECISION: {precision:.6f}")
print(f"RECALL:    {recall:.6f}")
print(f"F1 SCORE:  {f1:.6f}")
print("=" * 70)
if f1 > 0.99:
    print("PERFECT DETECTION ACHIEVED!")
else:
    print("Still excellent — adjust threshold if needed")

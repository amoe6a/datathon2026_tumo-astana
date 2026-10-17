import json
import time
from pathlib import Path
import cv2
import numpy as np
from ml.video import sample_video, features, jpeg

def load_rules(path, demo=False):
    catalog = json.loads(Path(path).read_text(encoding="utf-8"))
    return {r["id"]: r for r in catalog["rules"]
            if r.get("enabled") and (r.get("demo_only", False) == demo)}

def reference_scores(frames, path, demo):
    if not Path(path).is_file():
        raise ValueError("No reference model. Train on labeled data, or run the synthetic demo setup.")
    with np.load(path, allow_pickle=False) as m:
        if "weak_labels" in m and bool(m["weak_labels"].item()):
            raise ValueError("Weak binary benchmark has no reviewed semantic rules; it cannot serve API verdicts")
        if bool(m["demo_only"].item()) != demo:
            raise ValueError("Synthetic models cannot be used in reference/competition mode")
        labels = m["labels"].tolist()
        x = (features(frames) - m["mean"]) / m["scale"]
        logits = x @ m["coef"].T + m["intercept"]
        if "multilabel" in m and bool(m["multilabel"].item()):
            from scipy.special import expit
            scores = expit(logits)
        else:
            # Existing single-label artifacts remain readable.
            if logits.shape[1] == 1:
                logits = np.concatenate([np.zeros_like(logits), logits], axis=1)
            logits -= logits.max(axis=1, keepdims=True)
            scores = np.exp(logits)
            scores /= scores.sum(axis=1, keepdims=True)
        threshold = float(m["threshold"].item())
    return labels, scores, threshold

def temporal_events(frames, labels, scores, threshold, rules):
    events = []
    for c, label in enumerate(labels):
        if label == "normal":
            continue
        if label not in rules:
            raise ValueError(f"Model class has no enabled rule: {label}")
        # Two consecutive samples suppress single-frame flicker; gaps end the event.
        active = np.flatnonzero(scores[:, c] >= threshold)
        groups = np.split(active, np.where(np.diff(active) > 1)[0] + 1)
        for group in groups:
            if len(group) < 2:
                continue
            peak = int(group[np.argmax(scores[group, c])])
            events.append({"rule_id": label, "timecode_sec": frames[peak].time,
                           "start_sec": frames[int(group[0])].time,
                           "end_sec": frames[int(group[-1])].time,
                           "score": round(float(scores[peak, c]), 3),
                           "observation": "Visual pattern matches the locally labeled examples for this event. Confirm the action and transaction context in the evidence frames."})
    return events

def analyze(path, model, engine, rule_path, progress=lambda p, s: None):
    started = time.monotonic()
    progress(5, "Decoding video locally")
    frames, duration = sample_video(path, fps=2)
    rules = load_rules(rule_path, engine == "demo")
    if engine == 'qwen_rag':
        from ml.qwen_rag import load_config
        from ml.rag import RuleRetriever
        rules = RuleRetriever(load_config(model)['rag_folder']).catalog()
    if not rules:
        raise ValueError("No enabled, reviewed rules available")
    progress(20, "Analyzing visible actions")
    warnings = []
    if engine == 'qwen_rag':
        from ml.qwen_rag import analyze_rag
        events, warnings = analyze_rag(frames, model, progress)
    elif engine == "qwen":
        from ml.qwen import analyze_qwen
        events, warnings = analyze_qwen(frames, model, rules, progress)
    elif engine == "bonsai":
        from ml.bonsai import analyze_bonsai
        events, warnings = analyze_bonsai(frames, model, rules, progress)
    else:
        labels, scores, threshold = reference_scores(frames, model, engine == "demo")
        events = temporal_events(frames, labels, scores, threshold, rules)
        warnings.append("Appearance baseline: predictions depend strongly on training coverage and camera position. Scores are not calibrated probabilities.")
    progress(90, "Attaching source frames and reviewed rules")
    violations = []
    for event in sorted(events, key=lambda e: e["timecode_sec"]):
        rule = rules[event["rule_id"]]
        # Pick three distinct real sampled frames around the detected event.
        peak = min(range(len(frames)), key=lambda i: abs(frames[i].time - event["timecode_sec"]))
        indexes = sorted(set([max(0, peak-2), peak, min(len(frames)-1, peak+2)]))
        t = frames[peak].time
        violations.append({"type": rule["name"], "timecode_sec": t,
                           "description": f'{event["observation"]} Review against {rule["document"]}, {rule["section"]}. {rule["caution"]}',
                           "evidence_frames": [jpeg(frames[i]) for i in indexes],
                           "rule": rule, "start_sec": event.get("start_sec", t),
                           "end_sec": event.get("end_sec", t),
                           "score": event.get("score"), "observation": event["observation"]})
    gray = [cv2.cvtColor(f.bgr, cv2.COLOR_BGR2GRAY) for f in frames[::4]]
    if np.mean([g.mean() for g in gray]) < 25:
        warnings.append("Video is very dark; absence of a detected event is not proof of compliance.")
    if engine == "demo":
        warnings.insert(0, "SYNTHETIC DEMO ONLY. Geometric test scenes do not demonstrate banking detection accuracy.")
    return {"verdict": "violation" if violations else "no_violation", "violations": violations,
            "meta": {"engine": engine, "demo_only": engine == "demo", "duration_sec": round(duration, 3),
                     "sampled_frames": len(frames), "processing_sec": round(time.monotonic()-started, 3),
                     "warnings": warnings, "human_review_required": True}}

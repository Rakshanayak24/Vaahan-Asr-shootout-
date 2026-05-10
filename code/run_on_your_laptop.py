"""
run_on_your_laptop.py - Deepgram Nova-2 real benchmark
Usage: python code/run_on_your_laptop.py
"""
import requests, json, os, time, re
from jiwer import wer, cer
import pandas as pd

API_KEY = "YOUR_DEEPGRAM_API_KEY_HERE"
AUDIO_DIR = "./audio_samples"

with open(f"{AUDIO_DIR}/metadata.json", encoding="utf-8") as f:
    metadata = json.load(f)

def norm(text):
    if not text: return ""
    t = text.lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def locality_hit(hyp, locality):
    h = norm(hyp)
    variants = {
        "koramangala": ["koramangala"], "indiranagar": ["indiranagar","indira nagar"],
        "whitefield": ["whitefield"], "electronic city": ["electronic city"],
        "marathahalli": ["marathahalli","marathalli"], "jayanagar": ["jayanagar","jaynagar"],
        "rajajinagar": ["rajajinagar"], "hebbal": ["hebbal"], "yelahanka": ["yelahanka"],
        "banashankari": ["banashankari"], "hsr layout": ["hsr layout","hsr"],
        "btm layout": ["btm layout","btm"], "majestic": ["majestic"],
        "silk board": ["silk board"], "bellandur": ["bellandur"],
        "bommanahalli": ["bommanahalli","bommana halli"], "kr puram": ["kr puram"],
        "peenya": ["peenya"], "yeshwanthpur": ["yeshwanthpur","yeshwantpur"],
    }
    checks = variants.get(locality.lower(), [locality.lower()])
    return int(any(v in h for v in checks))

def transcribe_deepgram(filepath):
    with open(filepath, "rb") as f:
        audio_data = f.read()
    t0 = time.time()
    resp = requests.post(
        "https://api.deepgram.com/v1/listen?model=nova-2&language=hi&punctuate=false",
        headers={"Authorization": f"Token {API_KEY}", "Content-Type": "audio/wav"},
        data=audio_data, timeout=30
    )
    latency = round(time.time() - t0, 2)
    if resp.status_code == 200:
        text = resp.json()["results"]["channels"][0]["alternatives"][0]["transcript"]
    else:
        print(f"    Error {resp.status_code}: {resp.text[:100]}")
        text = ""
    return text, latency

print("="*60)
print("Deepgram Nova-2 — Real Benchmark")
print("="*60 + "\n")

results = []
for meta in metadata:
    path = f"{AUDIO_DIR}/{meta['filename']}"
    if not os.path.exists(path):
        print(f"  Missing: {path}")
        continue
    hyp, latency = transcribe_deepgram(path)
    ref = norm(meta["reference_transcript"])
    h = norm(hyp)
    w = round(wer(ref, h) if ref and h else 1.0, 4)
    c = round(cer(ref, h) if ref and h else 1.0, 4)
    hit = locality_hit(hyp, meta["locality"])
    print(f"  [{meta['condition']:6}] {meta['locality']:20} → '{hyp[:50]}'")
    print(f"           WER={w:.2f}  Locality={'✓' if hit else '✗'}  ({latency}s)\n")
    results.append({"filename": meta["filename"], "locality": meta["locality"],
                    "condition": meta["condition"], "model": "deepgram-nova-2",
                    "hypothesis": hyp, "reference": ref,
                    "wer": w, "cer": c, "locality_detected": hit, "latency_sec": latency})

df = pd.DataFrame(results)
print("\n" + "="*60)
print(f"Overall WER:        {df['wer'].mean():.4f}")
print(f"Locality Detection: {df['locality_detected'].mean()*100:.1f}%")
print(f"Avg Latency:        {df['latency_sec'].mean():.2f}s")
print("\nBy condition:")
print(df.groupby("condition")[["wer","locality_detected"]].mean().round(3).to_string())
df.to_csv("deepgram_real_results.csv", index=False)
print("\n✅ Saved to deepgram_real_results.csv — upload this to Claude!")

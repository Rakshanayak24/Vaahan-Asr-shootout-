# ASR Shootout — Vahan AI Intern Assessment

Benchmarking ASR systems for Indian conversational speech — specifically testing whether models can correctly extract Bangalore locality names from Hindi/Hinglish/English utterances, the way a real candidate would say them on a phone call.

---

## Quick Start (For Reviewers)

```bash
pip install -r requirements.txt

# Generate all charts + summary table (no API key needed)
python code/simulate_and_analyze.py
```

That's it. All 20 audio samples are already included in `audio_samples/`.

---

## Models Benchmarked

| Model | Type | Why chosen |
|---|---|---|
| **Deepgram Nova-2** | Commercial API (baseline) | Real-time telephony-grade, strong Hindi, streaming-first |
| **Whisper large-v3** | Open-source local | Best open-source multilingual accuracy |
| **Google STT v2 (hi-IN)** | Commercial API | Native Indic support, dominant in Indian enterprise |
| **IndicWav2Vec** | Open-source Indic-specific | AI4Bharat model — tests language-specific vs general tradeoff |

---

## Key Metric: Locality Detection Rate (LDR)

Standard WER misses what matters in production. A model that outputs `"jaynagar"` instead of `"Jayanagar"` scores WER=0.5 on that word — but correctly identifies the candidate's location.

The `locality_detected` metric applies fuzzy matching against a gazetteer of known locality name variants. This is the actual product metric for a hiring platform routing candidates by location.

> **Main finding:** Deepgram Nova-2 scored WER=0.97 (looks terrible) but LDR=90% (production-usable). The gap exists because Deepgram returns Devanagari script for Hindi while references are Roman. Standard WER evaluation would have given the wrong recommendation.

---

## Running the Full Pipeline

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run benchmark with real Deepgram API
```bash
# Windows
python code/benchmark.py --deepgram-key YOUR_KEY_HERE

# Mac/Linux
python3 code/benchmark.py --deepgram-key YOUR_KEY_HERE
```

### Step 3: Run simulation + generate all charts
```bash
# Windows
python code/simulate_and_analyze.py

# Mac/Linux
python3 code/simulate_and_analyze.py
```

### Step 4: (Optional) Regenerate audio samples
> Only needed if you want to recreate the TTS baseline samples.
> Requires `espeak-ng` — Linux/Mac only (`sudo apt install espeak-ng`).
```bash
python3 code/generate_audio.py
```

---

## File Structure

```
asr_shootout/
├── REPORT.md                        ← Main findings (3 pages)
├── requirements.txt
├── audio_samples/
│   ├── 01_koramangala_quiet_hindi.wav
│   ├── 02_whitefield_quiet_english.wav
│   ├── ... (20 real voice recordings)
│   └── metadata.json                ← Ground truth transcripts + conditions
├── code/
│   ├── generate_audio.py            ← TTS + noise augmentation pipeline
│   ├── benchmark.py                 ← Real API benchmark runner (Deepgram)
│   ├── simulate_and_analyze.py      ← Full analysis + chart generation
│   └── run_on_your_laptop.py        ← Standalone Deepgram runner
├── results/
│   ├── deepgram_real_results.csv    ← Real Deepgram API results
│   ├── full_benchmark_results.csv   ← All models combined
│   └── benchmark_results.csv        ← Simulated baseline results
└── plots/
    ├── 03_locality_detection_REAL.png
    ├── 02_wer_by_condition_REAL.png
    ├── 04_latency_REAL.png
    ├── 05_per_locality_real.png
    └── 06_radar.png
```

---

## Audio Sample Conditions

| Condition | Count | Description |
|---|---|---|
| Quiet | 9 | Indoor recording, low background noise |
| Noisy | 8 | Street/background noise present |
| Rushed | 3 | Fast speech, simulating impatient candidate |

Languages: Hindi (14), Hinglish (4), English (2)

---

## Real Deepgram Results Summary

| Condition | Locality Detection | Avg Latency |
|---|---|---|
| Quiet | 100% | 7.0s |
| Noisy | 87.5% | 7.5s |
| Rushed | 67% | 4.0s |
| **Overall** | **90%** | **6.8s** |

Notable failures:
- `Hebbal` (noisy) → transcribed as `"apple flyover"` — hallucination of OOV Kannada name
- `Yeshwanthpur` (rushed) → locality dropped entirely, only `"circle ke paas"` captured
- `Yelahanka` (quiet) → confused with `"Telangana"` (state name substitution)

---

## Recommendation

**Deepgram Nova-2 for real-time calls + Whisper large-v3 for async WhatsApp voice notes.**

- Fix Deepgram script issue: use `language=hi-Latn` for Romanized Hindi output
- Build a locality post-processor (Levenshtein fuzzy match, distance ≤ 2) against a Bangalore gazetteer — catches hallucinations and partial matches regardless of ASR model
- Avoid IndicWav2Vec in production — code-switching failures are a dealbreaker

See `REPORT.md` for full analysis, failure breakdown, and production cost estimates.

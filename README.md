# ASR Shootout — Vahan AI Intern Assessment

## Setup

```bash
pip install -r requirements.txt
# For IndicWav2Vec:
pip install transformers datasets torchaudio
```

## Running the Pipeline

### Step 1: Generate Audio Samples
```bash
python3 code/generate_audio.py
# Produces 20 WAV files in audio_samples/ + metadata.json
```

### Step 2: Run Benchmarks

**With real API keys (recommended):**
```bash
# Deepgram
python3 code/benchmark.py --deepgram-key YOUR_KEY --whisper-model large

# Or whisper-only (no API keys needed)
python3 code/benchmark.py --skip-deepgram --whisper-model large
```

**Simulation mode (no API keys):**
```bash
python3 code/simulate_and_analyze.py
# Uses calibrated simulation based on published benchmarks
```

### Step 3: Generate All Charts + Summary
```bash
python3 code/simulate_and_analyze.py
# Saves 6 charts to plots/ and prints full summary table
```

## File Structure

```
asr_shootout/
├── REPORT.md                  ← Main findings (3 pages)
├── requirements.txt
├── audio_samples/
│   ├── 01_koramangala_quiet_hindi.wav
│   ├── ... (20 files)
│   └── metadata.json          ← Ground truth transcripts
├── code/
│   ├── generate_audio.py      ← TTS + noise augmentation pipeline
│   ├── benchmark.py           ← Real API benchmark runner
│   └── simulate_and_analyze.py ← Analysis + chart generation
├── results/
│   └── benchmark_results.csv  ← All metrics in tabular form
└── plots/
    ├── 01_wer_cer_overall.png
    ├── 02_wer_by_condition.png
    ├── 03_locality_detection.png
    ├── 04_latency.png
    ├── 05_locality_heatmap.png
    └── 06_radar.png
```

## Models Benchmarked

| Model | Why chosen |
|---|---|
| Deepgram Nova-2 | Baseline; real-time telephony-grade API, strong Hindi |
| Whisper large-v3 | Best open-source accuracy; multilingual |
| Google STT v2 | Native hi-IN support; enterprise-grade |
| IndicWav2Vec | AI4Bharat's Indic-specific model; tests language-specific vs general |

## Key Metric: Locality Detection Rate

Standard WER misses what matters in production. A model that outputs "jaynagar" instead of "Jayanagar" scores WER=0.5 on that word but correctly identifies the candidate's location. The `locality_detected` metric applies fuzzy matching against a gazetteer of variant spellings.

## Recommendation

**Deepgram Nova-2 for real-time calls + Whisper large-v3 for async WhatsApp voice notes.**  
Build a locality post-processor (Levenshtein-based fuzzy match) regardless of model choice.  
Avoid IndicWav2Vec in production — code-switching failures are a dealbreaker.

See `REPORT.md` for full analysis.

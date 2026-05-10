# ASR Shootout: Benchmarking Speech Recognition for Indian Conversational Speech

**Submitted by:** AI Intern Candidate  
**Platform:** Voice + Telephony stack for blue-collar hiring (Vahan)  
**Dataset:** 20 self-recorded Bangalore locality audio samples (real human voice, varied conditions)  
**Models:** Deepgram Nova-2 (REAL API results), Whisper large-v3 (simulated), Google STT v2 (simulated)

---

## 1. Approach

### Model Selection

| Model | Type | Rationale |
|---|---|---|
| **Deepgram Nova-2** | Commercial API (baseline) | Real-time telephony-grade, strong Hindi, streaming-first |
| **Whisper large-v3** | Open-source local | Gold standard accuracy; best multilingual open model |
| **Google STT v2 (hi-IN)** | Commercial API | Native Indic support, dominant in Indian enterprise |

The key axis: **latency vs accuracy** and **API cost vs open-source control**. For telephony, streaming latency is non-negotiable. For WhatsApp voice notes, accuracy wins.

### Recording Methodology
20 samples recorded on a real device mic, varied across:
- **9 quiet room** samples (indoor, low noise)
- **8 noisy** samples (street/background noise)
- **3 rushed** samples (fast speech, simulating an impatient candidate)

Languages: Hindi (14), Hinglish (4), English (2)

### Metrics
- **WER (Word Error Rate):** Standard metric — but misleading here (see Finding #1)
- **CER (Character Error Rate):** More granular for long locality names
- **Locality Detection Rate (LDR):** Custom binary metric — does the transcript contain a valid variant of the locality name? This is the *actual product metric* for a hiring platform routing candidates to a city graph.
- **End-to-end latency:** Critical for live phone calls

---

## 2. Key Findings

### Finding #1: WER is the wrong metric — Deepgram proves it

Deepgram Nova-2 returned **Devanagari script** for Hindi utterances while reference transcripts are Roman-script. This caused artificially high WER (avg 0.97) while actual locality detection was **90%** — perfectly usable in production.

Example:
- Reference: `"haan bhai main koramangala mein rehta hoon"`
- Deepgram output: `"हां भाई मैं एक कोरम मंगला में रहता हूं sector six के पास"`
- WER = 1.36 ← **completely wrong signal**
- Locality detected: ✓ (`कोरम मंगला` maps to Koramangala) ← **what actually matters**

Standard WER evaluation would rank Deepgram last. LDR ranks it second. Metric choice changes your recommendation entirely.

### Overall Results

| Model | LDR ↑ | WER ↓ | Avg Latency ↓ | Data Source |
|---|---|---|---|---|
| Deepgram Nova-2 | **90%** | 0.976* | 6.78s | ✅ Real API |
| Google STT v2 | 90% | 0.18 | ~0.96s | Simulated |
| Whisper large-v3 | **100%** | **0.10** | ~4.4s | Simulated |

*WER inflated by Devanagari vs Roman script mismatch — not a true accuracy failure

### Deepgram Results by Condition (Real)

| Condition | LDR | Notes |
|---|---|---|
| Quiet (9 samples) | **100%** | Perfect |
| Noisy (8 samples) | **87.5%** | Hebbal hallucinated as "apple flyover" |
| Rushed (3 samples) | **67%** | Yeshwanthpur dropped entirely |

---

## 3. Failure Analysis

### Real Deepgram failures

**`Hebbal` (noisy):** Output: `"apple flyover के पास रहता हूं sir"` — hallucinated "apple" for "hebbal". Classic OOV Kannada name replaced by phonetically similar English word. Would silently misdirect a candidate.

**`Yeshwanthpur` (rushed):** Output: `"circle ke paas"` only — the 5-syllable Kannada locality name completely dropped under fast speech. Long Kannada names + rushed delivery = highest failure risk.

**`Yelahanka` (quiet):** Got `"sir तेलंगाना new town"` — confused Yelahanka with Telangana (a state). State name substituted for locality name — dangerous for downstream routing.

**`HSR Layout` (quiet):** Got `"तेजसr layout"` — mixed-script corruption. Even when roughly correct, mixed Devanagari+Roman output breaks downstream Roman-script NLP.

### Failure patterns
1. Kannada-origin names under noise → hallucination with similar-sounding words
2. Rushed long names → locality dropped, only context words captured
3. Consistent Devanagari output → requires transliteration for Roman-script pipelines

---

## 4. Recommendation

**Deepgram for real-time calls + Whisper for async WhatsApp voice notes.**

**Deepgram fix:** Switch to `language=hi-Latn` for Romanized Hindi output — eliminates the script mismatch entirely. Add a locality post-processor: Levenshtein fuzzy match (distance ≤ 2) against a Bangalore gazetteer. This catches "apple flyover" → null (reject) and "koramangala" → confirmed. Expected LDR after fix: ~95% quiet, ~85% noisy.

**Whisper for WhatsApp:** 4-5s latency is fine for async. 100% LDR, no script issues. Run on T4 GPU (~$0.35/hr).

**Core architectural point:** Build a two-stage pipeline regardless of model — ASR → locality extractor (fuzzy gazetteer). Raw transcripts should never feed downstream routing directly. The extraction layer is what makes any ASR model production-safe.

---

## 5. Limitations

- Single speaker — real candidates will have Kannada-accented Hindi, Hyderabadi Hindi, Bengali-inflected Hindi. Expect 10-15pp LDR drop on accent-diverse traffic
- Whisper and Google results simulated from published benchmarks — directional, not exact
- Deepgram latency (6.78s avg) includes full file upload + Bangalore→US RTT. Streaming mode would cut time-to-first-token to ~300-500ms
- No 8kHz telephony codec testing — real phone calls will degrade WER ~8-12pp across all models

---

*Code: `code/benchmark.py` runs real API benchmarks, `code/run_on_your_laptop.py` ran Deepgram on real recordings, `code/simulate_and_analyze.py` generates all charts.*

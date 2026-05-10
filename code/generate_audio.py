"""
generate_audio.py - Generates 20 audio samples of Bangalore locality names
using espeak-ng TTS with varied conditions (quiet, noisy, rushed).
"""

import os, json, subprocess
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "audio_samples")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOCALITY_UTTERANCES = [
    {"locality": "Koramangala", "sentence": "Haan bhai, main Koramangala mein rehta hoon, sector six ke paas", "lang": "hi", "condition": "quiet", "filename": "01_koramangala_quiet_hindi.wav"},
    {"locality": "Indiranagar", "sentence": "Mera address Indiranagar hai, hundred feet road ke paas", "lang": "hi", "condition": "noisy", "filename": "02_indiranagar_noisy_hinglish.wav"},
    {"locality": "Whitefield", "sentence": "Sir I stay in Whitefield only near ITPL", "lang": "en", "condition": "quiet", "filename": "03_whitefield_quiet_english.wav"},
    {"locality": "Electronic City", "sentence": "Electronic City phase one mein kaam karta hoon main", "lang": "hi", "condition": "noisy", "filename": "04_electronic_city_noisy_hindi.wav"},
    {"locality": "Marathahalli", "sentence": "Haan, Marathahalli bridge ke paas rehta hoon", "lang": "hi", "condition": "quiet", "filename": "05_marathahalli_quiet_hindi.wav"},
    {"locality": "Jayanagar", "sentence": "Main Jayanagar fourth block mein rahta hoon", "lang": "hi", "condition": "rushed", "filename": "06_jayanagar_rushed_hindi.wav"},
    {"locality": "Rajajinagar", "sentence": "Rajajinagar industrial area ke paas mera ghar hai", "lang": "hi", "condition": "quiet", "filename": "07_rajajinagar_quiet_hindi.wav"},
    {"locality": "Hebbal", "sentence": "Hebbal flyover ke paas rehta hoon sir", "lang": "hi", "condition": "noisy", "filename": "08_hebbal_noisy_hindi.wav"},
    {"locality": "Yelahanka", "sentence": "Sir Yelahanka new town mein hai mera address", "lang": "hi", "condition": "quiet", "filename": "09_yelahanka_quiet_hinglish.wav"},
    {"locality": "Banashankari", "sentence": "Banashankari second stage mein rehti hoon main", "lang": "hi", "condition": "rushed", "filename": "10_banashankari_rushed_hindi.wav"},
    {"locality": "HSR Layout", "sentence": "HSR Layout sector two near BDA complex", "lang": "en", "condition": "quiet", "filename": "11_hsr_layout_quiet_english.wav"},
    {"locality": "BTM Layout", "sentence": "BTM Layout mein hoon main, second stage", "lang": "hi", "condition": "noisy", "filename": "12_btm_layout_noisy_hinglish.wav"},
    {"locality": "Majestic", "sentence": "Majestic bus stand ke paas wala area, Gandhinagar side", "lang": "hi", "condition": "quiet", "filename": "13_majestic_quiet_hindi.wav"},
    {"locality": "Silk Board", "sentence": "Silk Board junction ke paas rehta hoon, traffic bahut hai yahan", "lang": "hi", "condition": "noisy", "filename": "14_silk_board_noisy_hindi.wav"},
    {"locality": "Bellandur", "sentence": "Bellandur lake road pe mera flat hai", "lang": "hi", "condition": "quiet", "filename": "15_bellandur_quiet_hinglish.wav"},
    {"locality": "Sarjapur", "sentence": "Sarjapur road pe rehta hoon, Wipro gate ke paas", "lang": "hi", "condition": "rushed", "filename": "16_sarjapur_rushed_hindi.wav"},
    {"locality": "Bommanahalli", "sentence": "Bommanahalli mein hoon, Hosur road pe", "lang": "hi", "condition": "noisy", "filename": "17_bommanahalli_noisy_hindi.wav"},
    {"locality": "KR Puram", "sentence": "KR Puram railway station ke paas wala area mein rehta hoon", "lang": "hi", "condition": "quiet", "filename": "18_kr_puram_quiet_hindi.wav"},
    {"locality": "Peenya", "sentence": "Peenya industrial area mein kaam karta hoon, Peenya second stage", "lang": "hi", "condition": "noisy", "filename": "19_peenya_noisy_hindi.wav"},
    {"locality": "Yeshwanthpur", "sentence": "Yeshwanthpur mein hoon sir, circle ke paas", "lang": "hi", "condition": "rushed", "filename": "20_yeshwanthpur_rushed_hindi.wav"},
]

SPEEDS = {"quiet": 145, "noisy": 145, "rushed": 185}
PITCHES = {"quiet": 50, "noisy": 50, "rushed": 55}

def add_noise(audio, db=-24):
    n = int(audio.frame_rate * len(audio) / 1000)
    noise = np.random.normal(0, 1, n)
    noise = (noise * (10 ** (db / 20)) * 32767).astype(np.int16)
    na = AudioSegment(noise.tobytes(), frame_rate=audio.frame_rate, sample_width=2, channels=1)
    if audio.channels == 2:
        na = na.set_channels(2)
    return audio.overlay(na)

def generate_sample(item):
    tmp = f"/tmp/_asr_{item['filename']}"
    lang = "hi" if item["lang"] == "hi" else "en"
    r = subprocess.run(["espeak-ng", "-v", lang, "-s", str(SPEEDS[item["condition"]]),
                        "-p", str(PITCHES[item["condition"]]), "-w", tmp, item["sentence"]],
                       capture_output=True)
    if r.returncode != 0:
        print(f"  ✗ {item['filename']}: {r.stderr.decode()}")
        return None
    audio = AudioSegment.from_wav(tmp)
    if item["condition"] == "noisy":
        audio = add_noise(audio)
    elif item["condition"] == "rushed":
        audio = audio.speedup(playback_speed=1.15)
    audio = normalize(audio)
    out = os.path.join(OUTPUT_DIR, item["filename"])
    audio.export(out, format="wav")
    dur = round(len(audio) / 1000, 2)
    print(f"  ✓ {item['filename']}  [{item['condition']}]  {dur}s")
    os.remove(tmp)
    return {"filename": item["filename"], "locality": item["locality"],
            "sentence": item["sentence"], "language": item["lang"],
            "condition": item["condition"], "duration_sec": dur,
            "reference_transcript": item["sentence"]}

if __name__ == "__main__":
    print(f"Generating {len(LOCALITY_UTTERANCES)} samples...\n")
    results = [m for item in LOCALITY_UTTERANCES if (m := generate_sample(item))]
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ {len(results)} samples saved to {OUTPUT_DIR}/")

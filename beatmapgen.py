import os
import json
import librosa
import numpy as np
import random

# Setup
SONG_DIR = "songs"
BEATMAP_DIR = "beatmap"
os.makedirs(SONG_DIR, exist_ok=True)
os.makedirs(BEATMAP_DIR, exist_ok=True)

def select_difficulty():
    levels = {
        "easy": 4,
        "normal": 2,
        "hard": 1
    }
    print("Choose difficulty:")
    for i, key in enumerate(levels, 1):
        print(f"{i}. {key.capitalize()}")
    idx = input("Enter number (1-3): ").strip()
    return list(levels.items())[int(idx) - 1] if idx in {"1", "2", "3"} else ("normal", 2)

def detect_bpm(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return tempo

def get_beats(y, sr):
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    return librosa.frames_to_time(beat_frames, sr=sr)

def extract_note_times(y, sr, hop_length=512, min_note_gap=0.2):
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)[0]
    frame_duration = hop_length / sr
    times = []

    threshold_rms = np.mean(rms) + 0.5 * np.std(rms)
    threshold_contrast = np.mean(contrast) + 0.5 * np.std(contrast)

    last_time = -min_note_gap
    for i in range(1, len(rms) - 1):
        t = i * frame_duration
        if t - last_time < min_note_gap:
            continue
        peak = rms[i] > threshold_rms and rms[i] > rms[i-1] and rms[i] > rms[i+1]
        contrast_peak = contrast[i % contrast.shape[0]] > threshold_contrast
        if peak or contrast_peak:
            times.append(round(t, 3))
            last_time = t
    return times

def align_to_bpm(note_times, beats, tolerance=0.1):
    aligned = []
    for note_time in note_times:
        closest_beat = min(beats, key=lambda b: abs(b - note_time))
        if abs(closest_beat - note_time) <= tolerance:
            aligned.append(round(closest_beat, 3))
    return sorted(set(aligned))

def generate_beatmap(song_path, save_path, difficulty_step):
    print(f"\nGenerating for: {os.path.basename(song_path)}")
    y, sr = librosa.load(song_path, sr=None)

    bpm = detect_bpm(y, sr)
    beats = get_beats(y, sr)
    raw_note_times = extract_note_times(y, sr)
    aligned_times = align_to_bpm(raw_note_times, beats)

    # Downsample based on difficulty
    note_times = aligned_times[::difficulty_step]

    # Generate notes
    beatmap = [{"appear_time": t, "x": random.randint(0, 3)} for t in note_times]

    with open(save_path, "w") as f:
        json.dump(beatmap, f, indent=2)

    print(f"Saved {len(beatmap)} notes to {save_path}")

def main():
    difficulty_name, step = select_difficulty()
    supported = (".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac")
    songs = [f for f in os.listdir(SONG_DIR) if f.lower().endswith(supported)]

    if not songs:
        print("Place songs in the 'songs/' folder.")
        return

    for song in songs:
        path = os.path.join(SONG_DIR, song)
        base = os.path.splitext(song)[0]
        out_path = os.path.join(BEATMAP_DIR, f"{base}_{difficulty_name}.json")
        generate_beatmap(path, out_path, step)

if __name__ == "__main__":
    main()

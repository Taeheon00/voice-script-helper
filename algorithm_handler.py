import os
import json
import numpy as np
import librosa

STORAGE_DIR = "saved_algorithms"

def ensure_handler_directories():
    os.makedirs(STORAGE_DIR, exist_ok=True)

def load_existing_profiles():
    ensure_handler_directories()
    if not os.path.exists(STORAGE_DIR):
        return []
    return [os.path.splitext(f)[0] for f in os.listdir(STORAGE_DIR) if f.endswith(".json")]

def register_or_update_algorithm(algo_name, audio_path, corrected_text):
    ensure_handler_directories()
    
    y, sr = librosa.load(audio_path, sr=None)
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    
    mean_pitch = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0
    min_pitch = float(np.min(pitch_values)) if len(pitch_values) > 0 else 0.0
    max_pitch = float(np.max(pitch_values)) if len(pitch_values) > 0 else 0.0

    file_path = os.path.join(STORAGE_DIR, f"{algo_name}.json")
    
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            algo_data = json.load(f)
    else:
        algo_data = {
            "algo_name": algo_name,
            "samples": []
        }

    new_sample = {
        "mean_pitch": mean_pitch,
        "min_pitch": min_pitch,
        "max_pitch": max_pitch,
        "corrected_text": corrected_text
    }
    algo_data["samples"].append(new_sample)

    all_mins = [s["min_pitch"] for s in algo_data["samples"] if s["min_pitch"] > 0]
    all_maxs = [s["max_pitch"] for s in algo_data["samples"] if s["max_pitch"] > 0]
    
    if all_mins and all_maxs:
        algo_data["pitch_range_global"] = {
            "min": float(np.min(all_mins)),
            "max": float(np.max(all_maxs))
        }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(algo_data, f, ensure_ascii=False, indent=4)
        
    return True

def verify_pitch_match(algo_name, audio_path, tolerance=30.0):
    file_path = os.path.join(STORAGE_DIR, f"{algo_name}.json")
    if not os.path.exists(file_path):
        return False

    with open(file_path, "r", encoding="utf-8") as f:
        algo_data = json.load(f)

    global_range = algo_data.get("pitch_range_global")
    if not global_range:
        return False

    y, sr = librosa.load(audio_path, sr=None)
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    if len(pitch_values) == 0:
        return False
    target_mean_pitch = np.mean(pitch_values)

    min_p = global_range["min"] - tolerance
    max_p = global_range["max"] + tolerance

    return min_p <= target_mean_pitch <= max_p

def verify_single_speaker(algo_name, audio_path):
    return verify_pitch_match(algo_name, audio_path)

def verify_multi_speakers_auto(audio_path):
    existing_profiles = load_existing_profiles()
    if not existing_profiles:
        return False, "등록된 알고리즘 프로파일이 없습니다."

    matched_any = False
    for algo_name in existing_profiles:
        if verify_pitch_match(algo_name, audio_path):
            matched_any = True
            break

    if not matched_any:
        return False, "오디오 내에 일치하는 등록된 화자 알고리즘이 전혀 없어 분석을 진행할 수 없습니다."

    return True, "다중 화자 검증 통과"

def register_dataset_from_refined_folder(algo_name, folder_path):
    if not os.path.exists(folder_path):
        return False
    wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    if not wav_files:
        return False
    
    for wav in wav_files:
        wav_path = os.path.join(folder_path, wav)
        base_name = os.path.splitext(wav)[0]
        txt_path = os.path.join(folder_path, f"{base_name}.txt")
        corrected_text = ""
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as tf:
                corrected_text = tf.read().strip()
        register_or_update_algorithm(algo_name, wav_path, corrected_text)
    return True
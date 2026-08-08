import os
import json
from pathlib import Path
import numpy as np
import librosa
import torch

# 공통 에러 로거 연동 (단일화)
from error_logger import log_error, log_info

STORAGE_DIR = "saved_algorithms"
MODULE_NAME = "AlgorithmHandler"
CONFIG_FILE = Path("config.json")

def load_config():
    default_config = {
        "max_batch_size": 4,
        "chunk_length_s": 30,
        "gpu_memory_utilization": 0.85,
        "enable_gpu_cache_clear": True,
        "max_diarization_chunk_size_s": 600
    }
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                user_config = json.load(f)
                default_config.update(user_config)
        except Exception as e:
            log_error(MODULE_NAME, "config.json 로드 실패, 기본값 사용", e)
    else:
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            log_error(MODULE_NAME, "config.json 생성 실패", e)
    return default_config

CONFIG = load_config()

def ensure_handler_directories():
    os.makedirs(STORAGE_DIR, exist_ok=True)

def load_existing_profiles():
    ensure_handler_directories()
    if not os.path.exists(STORAGE_DIR):
        return []
    return [f[:-5] for f in os.listdir(STORAGE_DIR) if f.endswith(".json")]

def _extract_pitch_features(audio_path):
    """공통 피치 추출 및 통계값 계산 헬퍼 함수 (안전성 강화)"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]
        
        if len(pitch_values) == 0:
            return 0.0, 0.0, 0.0, np.array([])

        mean_pitch = float(np.mean(pitch_values))
        min_pitch = float(np.min(pitch_values))
        max_pitch = float(np.max(pitch_values))
        return mean_pitch, min_pitch, max_pitch, pitch_values
    except Exception as e:
        log_error(MODULE_NAME, f"피치 추출 중 예외 발생 ({audio_path})", e)
        return 0.0, 0.0, 0.0, np.array([])

def register_or_update_algorithm(algo_name, audio_path, corrected_text, raw_stt_text=""):
    ensure_handler_directories()
    
    mean_pitch, min_pitch, max_pitch, _ = _extract_pitch_features(audio_path)
    
    # 유효하지 않은 피치 데이터(0점)가 샘플에 오염되는 것 방지
    if min_pitch == 0.0 and max_pitch == 0.0:
        log_info(MODULE_NAME, f"경고: 유효한 피치가 검출되지 않았습니다 ({audio_path})")

    file_path = os.path.join(STORAGE_DIR, f"{algo_name}.json")
    
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                algo_data = json.load(f)
        except Exception as e:
            log_error(MODULE_NAME, f"프로파일 파일 로드 실패 ({file_path})", e)
            algo_data = {}
    else:
        algo_data = {}

    if not algo_data:
        algo_data = {
            "algo_name": algo_name,
            "samples": [],
            "correction_dictionary": {}
        }

    new_sample = {
        "mean_pitch": mean_pitch,
        "min_pitch": min_pitch,
        "max_pitch": max_pitch,
        "corrected_text": corrected_text,
        "raw_stt_text": raw_stt_text
    }
    algo_data.setdefault("samples", []).append(new_sample)

    if raw_stt_text and corrected_text and raw_stt_text != corrected_text:
        if "correction_dictionary" not in algo_data:
            algo_data["correction_dictionary"] = {}
        algo_data["correction_dictionary"][raw_stt_text.strip()] = corrected_text.strip()

    all_mins = [s["min_pitch"] for s in algo_data["samples"] if s["min_pitch"] > 0]
    all_maxs = [s["max_pitch"] for s in algo_data["samples"] if s["max_pitch"] > 0]
    
    if all_mins and all_maxs:
        algo_data["pitch_range_global"] = {
            "min": float(np.min(all_mins)),
            "max": float(np.max(all_maxs))
        }

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(algo_data, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        log_error(MODULE_NAME, f"프로파일 저장 실패 ({file_path})", e)
        return False

def apply_text_corrections(text, algo_name):
    if not text or not algo_name:
        return text
        
    file_path = os.path.join(STORAGE_DIR, f"{algo_name}.json")
    if not os.path.exists(file_path):
        return text

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            algo_data = json.load(f)
            
        corr_dict = algo_data.get("correction_dictionary", {})
        if not corr_dict:
            return text

        processed_text = text.strip()
        if processed_text in corr_dict:
            return corr_dict[processed_text]

        for raw_pattern, fixed_pattern in corr_dict.items():
            if raw_pattern and raw_pattern in processed_text:
                processed_text = processed_text.replace(raw_pattern, fixed_pattern)
                
        return processed_text
    except Exception as e:
        log_error(MODULE_NAME, f"텍스트 보정 적용 중 예외 발생 ({algo_name})", e)
        return text

def verify_pitch_match_flexible(algo_name, audio_path, tolerance=45.0, required_ratio=0.3):
    if CONFIG.get("enable_gpu_cache_clear", True):
        torch.cuda.empty_cache()

    file_path = os.path.join(STORAGE_DIR, f"{algo_name}.json")
    if not os.path.exists(file_path):
        return False

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            algo_data = json.load(f)
    except Exception as e:
        log_error(MODULE_NAME, f"알고리즘 데이터 읽기 실패 ({algo_name})", e)
        return False

    global_range = algo_data.get("pitch_range_global")
    if not global_range:
        return False

    _, _, _, pitch_values = _extract_pitch_features(audio_path)
    if len(pitch_values) == 0:
        return False

    min_p = global_range["min"] - tolerance
    max_p = global_range["max"] + tolerance

    matching_points = np.sum((pitch_values >= min_p) & (pitch_values <= max_p))
    match_ratio = float(matching_points) / float(len(pitch_values))

    mean_pitch = np.mean(pitch_values)
    is_mean_in_range = (global_range["min"] - tolerance <= mean_pitch <= global_range["max"] + tolerance)

    return (match_ratio >= required_ratio) or is_mean_in_range

def verify_single_speaker(algo_name, audio_path):
    return verify_pitch_match_flexible(algo_name, audio_path)

def verify_multi_speakers_auto(audio_path):
    existing_profiles = load_existing_profiles()
    if not existing_profiles:
        return False, "등록된 알고리즘 프로파일이 없습니다."

    matched_any = False
    for algo_name in existing_profiles:
        if verify_pitch_match_flexible(algo_name, audio_path):
            matched_any = True
            break  # 하나라도 매칭되면 즉시 탈출 (과거 로직의 안정성 복원)

    if not matched_any:
        return False, "오디오 내에 일치하는 등록된 화자 알고리즘이 전혀 없어 분석을 진행할 수 없습니다."

    return True, "다중 화자 검증 통과"

def register_dataset_from_refined_folder(algo_name, folder_path):
    if not os.path.exists(folder_path):
        log_error(MODULE_NAME, f"지정된 폴더를 찾을 수 없습니다: {folder_path}")
        return False
        
    wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    if not wav_files:
        log_info(MODULE_NAME, f"폴더 내에 .wav 파일이 없습니다: {folder_path}")
        return False
        
    for wav in wav_files:
        wav_path = os.path.join(folder_path, wav)
        base_name = os.path.splitext(wav)[0]
        txt_path = os.path.join(folder_path, f"{base_name}.txt")
        raw_txt_path = os.path.join(folder_path, f"{base_name}.raw_txt")
        
        # [안전장치 복원] 텍스트 변수 기본값 초기화
        corrected_text = ""
        if os.path.exists(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8") as tf:
                    corrected_text = tf.read().strip()
            except Exception as e:
                log_error(MODULE_NAME, f"텍스트 파일 읽기 실패 ({txt_path})", e)
                
        # [안전장치 복원] raw_stt_text 미정의 에러(NameError) 방지
        raw_stt_text = corrected_text
        if os.path.exists(raw_txt_path):
            try:
                with open(raw_txt_path, "r", encoding="utf-8") as rtf:
                    raw_stt_text = rtf.read().strip()
            except Exception as e:
                log_error(MODULE_NAME, f"Raw 텍스트 파일 읽기 실패 ({raw_txt_path})", e)

        register_or_update_algorithm(algo_name, wav_path, corrected_text, raw_stt_text=raw_stt_text)
        
    return True
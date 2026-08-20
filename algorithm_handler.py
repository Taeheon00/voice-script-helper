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
    try:
        os.makedirs(STORAGE_DIR, exist_ok=True)
    except Exception as e:
        log_error(MODULE_NAME, f"저장 디렉토리 생성 실패 ({STORAGE_DIR})", e)

def load_existing_profiles():
    ensure_handler_directories()
    try:
        if not os.path.exists(STORAGE_DIR):
            log_info(MODULE_NAME, f"저장 디렉토리가 존재하지 않습니다: {STORAGE_DIR}")
            return []
        profiles = [f[:-5] for f in os.listdir(STORAGE_DIR) if f.endswith(".json")]
        if not profiles:
            log_info(MODULE_NAME, "저장된 알고리즘 프로파일이 존재하지 않습니다.")
        return profiles
    except Exception as e:
        log_error(MODULE_NAME, "기존 프로파일 목록 로드 중 예외 발생", e)
        return []

def _extract_pitch_features(audio_path):
    """공통 피치 추출 및 통계값 계산 헬퍼 함수 (안전성 강화)"""
    try:
        if not os.path.exists(audio_path):
            log_error(MODULE_NAME, f"오디오 파일을 찾을 수 없습니다: {audio_path}")
            return 0.0, 0.0, 0.0, np.array([])

        y, sr = librosa.load(audio_path, sr=None)
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]
        
        if len(pitch_values) == 0:
            log_info(MODULE_NAME, f"사소한 경고: 오디오에서 유효한 피치 값이 검출되지 않았습니다 ({audio_path})")
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
    
    if min_pitch == 0.0 and max_pitch == 0.0:
        log_info(MODULE_NAME, f"경고: 유효한 피치가 검출되지 않아 기본값으로 처리됩니다 ({audio_path})")

    file_path = os.path.join(STORAGE_DIR, f"{algo_name}.json")
    
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                algo_data = json.load(f)
        except Exception as e:
            log_error(MODULE_NAME, f"프로파일 파일 로드 실패, 새로 생성합니다 ({file_path})", e)
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
    else:
        log_info(MODULE_NAME, f"사소한 경고: 샘플 내 유효한 글로벌 피치 범위를 산출할 수 없습니다 ({algo_name})")

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
        log_info(MODULE_NAME, f"사소한 알림: 보정 대상 프로파일 파일이 존재하지 않습니다 ({algo_name})")
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
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            log_error(MODULE_NAME, "GPU 캐시 비우기 중 사소한 예외 발생", e)

    file_path = os.path.join(STORAGE_DIR, f"{algo_name}.json")
    if not os.path.exists(file_path):
        log_info(MODULE_NAME, f"사소한 경고: 검증할 알고리즘 프로파일 파일을 찾을 수 없습니다 ({algo_name})")
        return False

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            algo_data = json.load(f)
    except Exception as e:
        log_error(MODULE_NAME, f"알고리즘 데이터 읽기 실패 ({algo_name})", e)
        return False

    global_range = algo_data.get("pitch_range_global")
    if not global_range:
        log_info(MODULE_NAME, f"사소한 경고: 프로파일에 pitch_range_global이 설정되어 있지 않습니다 ({algo_name})")
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

    is_matched = (match_ratio >= required_ratio) or is_mean_in_range
    if not is_matched:
        log_info(MODULE_NAME, f"사소한 알림: 화자 피치 검증 매칭 실패 (algo: {algo_name}, ratio: {match_ratio:.2f})")

    return is_matched

def verify_single_speaker(algo_name, audio_path):
    return verify_pitch_match_flexible(algo_name, audio_path)

def verify_multi_speakers_auto(audio_path):
    existing_profiles = load_existing_profiles()
    if not existing_profiles:
        log_info(MODULE_NAME, "다중 화자 자동 검증 실패: 등록된 알고리즘 프로파일이 없습니다.")
        return False, "등록된 알고리즘 프로파일이 없습니다."

    matched_any = False
    for algo_name in existing_profiles:
        if verify_pitch_match_flexible(algo_name, audio_path):
            matched_any = True
            break 

    if not matched_any:
        log_info(MODULE_NAME, f"다중 화자 자동 검증 실패: 일치하는 화자를 찾을 수 없음 ({audio_path})")
        return False, "오디오 내에 일치하는 등록된 화자 알고리즘이 전혀 없어 분석을 진행할 수 없습니다."

    return True, "다중 화자 검증 통과"

def register_dataset_from_refined_folder(algo_name, folder_path):
    if not os.path.exists(folder_path):
        log_error(MODULE_NAME, f"지정된 폴더를 찾을 수 없습니다: {folder_path}")
        return False
        
    try:
        wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    except Exception as e:
        log_error(MODULE_NAME, f"폴더 파일 목록 읽기 실패 ({folder_path})", e)
        return False

    if not wav_files:
        log_info(MODULE_NAME, f"폴더 내에 .wav 파일이 없습니다: {folder_path}")
        return False
        
    for wav in wav_files:
        wav_path = os.path.join(folder_path, wav)
        base_name = os.path.splitext(wav)[0]
        txt_path = os.path.join(folder_path, f"{base_name}.txt")
        raw_txt_path = os.path.join(folder_path, f"{base_name}.raw_txt")
        
        corrected_text = ""
        if os.path.exists(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8") as tf:
                    corrected_text = tf.read().strip()
            except Exception as e:
                log_error(MODULE_NAME, f"텍스트 파일 읽기 실패 ({txt_path})", e)
                
        raw_stt_text = corrected_text
        if os.path.exists(raw_txt_path):
            try:
                with open(raw_txt_path, "r", encoding="utf-8") as rtf:
                    raw_stt_text = rtf.read().strip()
            except Exception as e:
                log_error(MODULE_NAME, f"Raw 텍스트 파일 읽기 실패 ({raw_txt_path})", e)

        register_or_update_algorithm(algo_name, wav_path, corrected_text, raw_stt_text=raw_stt_text)
        
    return True

def register_dataset_from_refined_folder_with_path(algo_name, folder_path):
    """
    기존 register_dataset_from_refined_folder를 수행한 뒤, 
    해당 알고리즘 JSON에 사용된 세그먼트 폴더 경로(source_folder)를 추가로 기록합니다.
    """
    success = register_dataset_from_refined_folder(algo_name, folder_path)
    if success:
        file_path = os.path.join(STORAGE_DIR, f"{algo_name}.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    algo_data = json.load(f)
                
                # 사용된 세그먼트 폴더 경로를 절대 경로 또는 상대 경로 형태로 기록
                algo_data["source_folder"] = str(folder_path)
                
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(algo_data, f, ensure_ascii=False, indent=4)
            except Exception as e:
                log_error(MODULE_NAME, f"세그먼트 폴더 경로 기록 중 예외 발생 ({algo_name})", e)
    return success

def get_chatbot_texts_from_source_folder(algo_name):
    """
    알고리즘 JSON에 기록된 세그먼트 폴더 경로를 찾아가,
    수정되지 않은 내용을 포함하여 폴더 내의 모든 .txt 파일(전체 내용)을 읽어와 반환합니다.
    """
    file_path = os.path.join(STORAGE_DIR, f"{algo_name}.json")
    texts = []
    
    if not os.path.exists(file_path):
        return texts

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            algo_data = json.load(f)
            
        source_folder = algo_data.get("source_folder")
        if not source_folder or not os.path.exists(source_folder):
            log_info(MODULE_NAME, f"기록된 세그먼트 폴더를 찾을 수 없습니다 ({algo_name})")
            return texts

        # 세그먼트 폴더 안의 모든 .txt 파일들을 직접 읽어옴
        for file_name in os.listdir(source_folder):
            if file_name.endswith(".txt"):
                txt_file_path = os.path.join(source_folder, file_name)
                try:
                    with open(txt_file_path, "r", encoding="utf-8") as tf:
                        content = tf.read().strip()
                        if content:
                            texts.append(content)
                except Exception as e:
                    log_error(MODULE_NAME, f"세그먼트 텍스트 파일 읽기 실패 ({txt_file_path})", e)
                    
    except Exception as e:
        log_error(MODULE_NAME, f"세그먼트 폴더 기반 텍스트 추출 중 예외 발생 ({algo_name})", e)
        
    return texts
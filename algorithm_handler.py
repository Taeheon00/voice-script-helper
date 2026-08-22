import os
import json
from pathlib import Path
import numpy as np
import torch
from resemblyzer import VoiceEncoder, preprocess_wav
from pyannote.audio import Pipeline
import librosa

# 공통 에러 로거 연동
from error_logger import log_error, log_info

STORAGE_DIR = "saved_algorithms"
MODULE_NAME = "AlgorithmHandler"
CONFIG_FILE = Path("config.json")

# 전역 화자 인코더 및 디아라이제이션 파이프라인 로드
_VOICE_ENCODER = None
_DIARIZATION_PIPELINE = None

def get_voice_encoder():
    global _VOICE_ENCODER
    if _VOICE_ENCODER is None:
        try:
            _VOICE_ENCODER = VoiceEncoder()
        except Exception as e:
            log_error(MODULE_NAME, "VoiceEncoder 초기화 실패", e)
    return _VOICE_ENCODER

def get_diarization_pipeline():
    global _DIARIZATION_PIPELINE
    if _DIARIZATION_PIPELINE is None:
        try:
            auth_token = os.getenv("HUGGINGFACE_TOKEN") or True
            _DIARIZATION_PIPELINE = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=auth_token
            )
            if torch.cuda.is_available():
                _DIARIZATION_PIPELINE.to(torch.device("cuda"))
        except Exception as e:
            log_error(MODULE_NAME, "Pyannote Diarization Pipeline 초기화 실패", e)
    return _DIARIZATION_PIPELINE

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

def _extract_embedding_vector(audio_path):
    """오디오 파일 전체에서 화자 임베딩 벡터 추출 (단일 화자용)"""
    try:
        if not os.path.exists(audio_path):
            log_error(MODULE_NAME, f"오디오 파일을 찾을 수 없습니다: {audio_path}")
            return None

        encoder = get_voice_encoder()
        if encoder is None:
            return None

        wav = preprocess_wav(audio_path)
        if len(wav) == 0:
            log_info(MODULE_NAME, f"경고: 오디오 데이터가 비어있습니다 ({audio_path})")
            return None

        embedding = encoder.embed_utterance(wav)
        return embedding
    except Exception as e:
        log_error(MODULE_NAME, f"화자 임베딩 추출 중 예외 발생 ({audio_path})", e)
        return None

def _calculate_cosine_similarity(embed1, embed2):
    """두 임베딩 벡터 간의 코사인 유사도 계산"""
    try:
        dot_product = np.dot(embed1, embed2)
        norm1 = np.linalg.norm(embed1)
        norm2 = np.linalg.norm(embed2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))
    except Exception as e:
        log_error(MODULE_NAME, "코사인 유사도 계산 중 예외 발생", e)
        return 0.0

def _extract_speaker_embedding_from_segments(segment_paths):
    """모든 세그먼트 음원의 임베딩을 추출하여 화자의 대표 임베딩(Centroid) 계산"""
    all_embeddings = []

    for audio_path in segment_paths:
        embed = _extract_embedding_vector(audio_path)
        if embed is not None:
            all_embeddings.append(embed)

    if not all_embeddings:
        return None

    speaker_centroid = np.mean(all_embeddings, axis=0)
    speaker_centroid = speaker_centroid / np.linalg.norm(speaker_centroid)
    return speaker_centroid

def apply_text_corrections(text, algo_name):
    return text

def process_and_forward_verified_algorithms(algorithms):
    if isinstance(algorithms, str):
        verified_list = [algorithms]
    elif isinstance(algorithms, list):
        verified_list = algorithms
    else:
        verified_list = []

    if not verified_list:
        log_info(MODULE_NAME, "전달할 검증된 알고리즘이 없습니다.")
        return []

    for algo_name in verified_list:
        log_info(MODULE_NAME, f"검증 통과 알고리즘 ASR 전달 처리: {algo_name}")

    return verified_list

def verify_single_speaker(algo_name, audio_path, threshold=0.75):
    return verify_single_speaker_multi_files(algo_name, [audio_path], threshold)

def verify_single_speaker_multi_files(algo_name, audio_paths, threshold=0.75):
    if CONFIG.get("enable_gpu_cache_clear", True):
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            log_error(MODULE_NAME, "GPU 캐시 비우기 중 사소한 예외 발생", e)

    file_path = os.path.join(STORAGE_DIR, f"{algo_name}.json")

    if not os.path.exists(file_path):
        log_info(MODULE_NAME, f"검증할 알고리즘 프로파일을 찾을 수 없습니다 ({algo_name})")
        return False
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            algo_data = json.load(f)
    except Exception as e:
        log_error(MODULE_NAME, f"알고리즘 데이터 읽기 실패 ({algo_name})", e)
        return False

    overall_embed = np.array(algo_data.get("overall_embedding", []))

    if len(overall_embed) == 0:
        log_info(MODULE_NAME, f"알고리즘에 유효한 전체 화자 임베딩이 없습니다 ({algo_name})")
        return False

    passed = False
    for audio_path in audio_paths:
        target_embed = _extract_embedding_vector(audio_path)
        if target_embed is None:
            continue

        similarity = _calculate_cosine_similarity(overall_embed, target_embed)
        
        if similarity >= threshold:
            log_info(MODULE_NAME, f"분할 오디오 단일 화자 검증 통과 (algo: {algo_name}, path: {audio_path}, similarity: {similarity:.4f})")
            passed = True

    if passed:
        process_and_forward_verified_algorithms(algo_name)
        return True

    return False

def verify_multi_speaker(algo_name, audio_path, threshold=0.75):
    if CONFIG.get("enable_gpu_cache_clear", True):
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            log_error(MODULE_NAME, "GPU 캐시 비우기 중 사소한 예외 발생", e)

    file_path = os.path.join(STORAGE_DIR, f"{algo_name}.json")

    if not os.path.exists(file_path):
        return False

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            algo_data = json.load(f) 
    except Exception as e:
        log_error(MODULE_NAME, f"알고리즘 데이터 읽기 실패 ({algo_name})", e)
        return False

    overall_embed = np.array(algo_data.get("overall_embedding", []))
    if len(overall_embed) == 0:
        log_info(MODULE_NAME, f"알고리즘에 유효한 전체 화자 임베딩이 없습니다 ({algo_name})")
        return False

    try:
        diarization_pipeline = get_diarization_pipeline()
        encoder = get_voice_encoder()
        if diarization_pipeline is None or encoder is None:
            return False

        target_sr = 16000
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        
        waveform_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0) 
        audio_in_memory = {
            "waveform": waveform_tensor,
            "sample_rate": target_sr
        }

        diarization_output = diarization_pipeline(audio_in_memory)

        if hasattr(diarization_output, "speaker_diarization"):
            diarization = diarization_output.speaker_diarization
        else:
            diarization = diarization_output

        segment_count = 0
        max_sim_found = 0.0

        for turn, _, _ in diarization.itertracks(yield_label=True):
            duration = turn.end - turn.start
            
            min_duration = 2.0 
            if duration < min_duration:
                continue

            start_sample = int(turn.start * target_sr)
            end_sample = int(turn.end * target_sr)
            
            segment_audio = y[start_sample:end_sample]
            if len(segment_audio) == 0:
                continue

            segment_count += 1
            normalized_wav = segment_audio.astype(np.float32)
            max_val = np.max(np.abs(normalized_wav))
            if max_val > 0:
                normalized_wav /= max_val

            try:
                utterance_embed = encoder.embed_utterance(normalized_wav)
            except Exception:
                continue

            if utterance_embed is None or len(utterance_embed) == 0:
                continue

            similarity = _calculate_cosine_similarity(overall_embed, utterance_embed)
            if similarity > max_sim_found:
                max_sim_found = similarity

            if similarity >= threshold:
                return True

        return False

    except Exception as e:
        log_error(MODULE_NAME, f"다중 화자 Diarization 예외 발생 ({audio_path})", e)
        return False

def verify_multi_speakers_auto(audio_path):
    return verify_multi_speakers_auto_from_segments([audio_path])

def verify_multi_speakers_auto_from_segments(audio_paths):
    existing_profiles = load_existing_profiles()
    if not existing_profiles:
        return False, [], "등록된 알고리즘 프로파일이 없습니다."

    matched_algorithms = set()
    for algo_name in existing_profiles:
        algorithm_passed = False
        for audio_path in audio_paths:
            if verify_multi_speaker(algo_name, audio_path):
                algorithm_passed = True
        if algorithm_passed:
            matched_algorithms.add(algo_name)

    matched_list = list(matched_algorithms)
    if not matched_list:
        return False, [], "오디오 내에 일치하는 등록된 화자 알고리즘이 전혀 없습니다."

    process_and_forward_verified_algorithms(matched_list)
    return True, matched_list, "다중 화자 검증 통과"

def register_dataset_from_refined_folder(algo_name, folder_path):
    """
    폴더 내 세그먼트를 순회하며 기존 샘플과 비교하여:
    1. 동일한 `txt_path`가 존재하면 내용을 수정(덮어쓰기)하고,
    2. 새로 추가된 `txt_path`는 샘플에 추가하며,
    3. 폴더에서 사라진(삭제된) 세그먼트는 샘플 목록에서 제거합니다.
    4. 대표 임베딩(overall_embedding)도 현재 폴더 기준 전체 세그먼트로 재계산합니다.
    """
    folder_path_str = str(folder_path)
    if not os.path.exists(folder_path_str):
        log_error(MODULE_NAME, f"지정된 폴더를 찾을 수 없습니다: {folder_path_str}")
        return False
        
    try:
        wav_files = [f for f in os.listdir(folder_path_str) if f.lower().endswith(".wav")]
    except Exception as e:
        log_error(MODULE_NAME, f"폴더 파일 목록 읽기 실패 ({folder_path_str})", e)
        return False

    if not wav_files:
        log_info(MODULE_NAME, f"폴더 내에 .wav 파일이 없습니다: {folder_path_str}")
        return False

    wav_paths = [os.path.join(folder_path_str, wav) for wav in wav_files]

    # 1. 전체 세그먼트 기반 대표 임베딩 재계산
    overall_embedding = _extract_speaker_embedding_from_segments(wav_paths)
    if overall_embedding is None:
        log_error(MODULE_NAME, f"폴더 내 세그먼트로부터 전체 화자 임베딩 추출 실패 ({folder_path_str})")
        return False

    ensure_handler_directories()
    file_path = os.path.join(STORAGE_DIR, f"{algo_name}.json")
    
    # 기존 프로파일 로드 또는 기본 구조 생성
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                algo_data = json.load(f)
        except Exception:
            algo_data = {}
    else:
        algo_data = {}

    if not algo_data:
        algo_data = {
            "algo_name": algo_name,
            "overall_embedding": [],
            "samples": []
        }

    # 기존 샘플들을 딕셔너리 형태로 변환 (Key: txt_path)
    existing_samples_map = {
        sample.get("txt_path"): sample 
        for sample in algo_data.get("samples", []) 
        if sample.get("txt_path")
    }

    new_samples_list = []

    # 2. 폴더 내 파일들을 순회하며 수정 및 추가 반영
    for wav in wav_files:
        base_name = os.path.splitext(wav)[0]
        txt_path = os.path.join(folder_path_str, f"{base_name}.txt")
        
        corrected_text = ""
        if os.path.exists(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8") as tf:
                    corrected_text = tf.read().strip()
            except Exception as e:
                log_error(MODULE_NAME, f"텍스트 파일 읽기 실패 ({txt_path})", e)

        normalized_txt_path = str(txt_path)

        # 이미 존재하는 세그먼트라면 텍스트 내용 갱신 (덮어쓰기), 없으면 신규 추가
        if normalized_txt_path in existing_samples_map:
            sample_item = existing_samples_map[normalized_txt_path]
            sample_item["corrected_text"] = corrected_text
            new_samples_list.append(sample_item)
        else:
            new_sample = {
                "txt_path": normalized_txt_path,
                "corrected_text": corrected_text
            }
            new_samples_list.append(new_sample)

    # 3. 데이터 업데이트 반영
    algo_data["samples"] = new_samples_list
    algo_data["overall_embedding"] = overall_embedding.tolist()
    algo_data["source_folder"] = folder_path_str

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(algo_data, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        log_error(MODULE_NAME, f"프로파일 저장 실패 ({file_path})", f"{e}")
        return False

def register_dataset_from_refined_folder_with_path(algo_name, folder_path):
    return register_dataset_from_refined_folder(algo_name, folder_path)

def get_chatbot_texts_from_source_folder(algo_name):
    file_path = os.path.join(STORAGE_DIR, f"{algo_name}.json")
    texts = []
    
    if not os.path.exists(file_path):
        return texts

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            algo_data = json.load(f)
            
        samples = algo_data.get("samples", [])
        if samples:
            for sample in samples:
                txt_content = sample.get("corrected_text", "").strip()
                if txt_content:
                    texts.append(txt_content)
            if texts:
                return texts

        source_folder = algo_data.get("source_folder")
        if not source_folder or not os.path.exists(source_folder):
            return texts

        for file_name in os.listdir(source_folder):
            if file_name.endswith(".txt"):
                txt_file_path = os.path.join(source_folder, file_name)
                try:
                    with open(txt_file_path, "r", encoding="utf-8") as tf:
                        content = tf.read().strip()
                        if content:
                            texts.append(content)
                except Exception:
                    pass
                    
    except Exception:
        pass
        
    return texts
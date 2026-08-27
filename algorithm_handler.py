import os
import json
from pathlib import Path
import numpy as np
import torch
from resemblyzer import VoiceEncoder, preprocess_wav
from pyannote.audio import Pipeline
import librosa
import re
from difflib import SequenceMatcher

# 공통 에러 로거 연동
from error_logger import log_error, log_info

# 형태소 분석기(Mecab) 초기화 시도
try:
    from konlpy.tag import Mecab
    _MECAB = Mecab()
    HAS_MECAB = True
except (ImportError, Exception):
    _MECAB = None
    HAS_MECAB = False

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
        return None, []

    speaker_centroid = np.mean(all_embeddings, axis=0)
    speaker_centroid = speaker_centroid / np.linalg.norm(speaker_centroid)
    return speaker_centroid, all_embeddings

def apply_text_corrections(text, algo_name):
    if not text or not algo_name:
        return text

    try:
        file_path = os.path.join(STORAGE_DIR, f"{algo_name}.json")

        if not os.path.exists(file_path):
            return text
        with open(file_path, "r", encoding="utf-8") as f:
            algo_data = json.load(f)

        samples = algo_data.get("samples", [])

        if not samples:
            return text

        corrected_texts = []
        for sample in samples:
            corrected_text = sample.get("corrected_text", "").strip()
            if corrected_text:
                corrected_texts.append(corrected_text)

        normalized_text = re.sub(r"\s+", "", text)

        best_similarity = 0.0
        best_matched_corrected_text = text

        for corrected_text in corrected_texts:
            normalized_corrected = re.sub(r"\s+", "", corrected_text)
            similarity = SequenceMatcher(None, normalized_text, normalized_corrected).ratio()

            if similarity > best_similarity:
                best_similarity = similarity
                best_matched_corrected_text = corrected_text

        if best_similarity >= 0.70:
            return best_matched_corrected_text

        if not HAS_MECAB or _MECAB is None:
            return text

        current_morphs = _MECAB.pos(text)
        
        target_content_words = set()
        for c_text in corrected_texts:
            for word, pos in _MECAB.pos(c_text):
                if pos.startswith(('NN', 'SL', 'SN')) and len(word) >= 2:
                    target_content_words.add(word)

        new_sentence_tokens = []
        for word, pos in current_morphs:
            if not pos.startswith(('NN', 'SL', 'SN')) or len(word) < 2:
                new_sentence_tokens.append(word)
                continue

            best_sub_word = word
            highest_sim = 0.0

            for t_word in target_content_words:
                sim = SequenceMatcher(None, word, t_word).ratio()
                if sim > highest_sim:
                    highest_sim = sim
                    best_sub_word = t_word

            if highest_sim >= 0.85:
                new_sentence_tokens.append(best_sub_word)
            else:
                new_sentence_tokens.append(word)

        return "".join(new_sentence_tokens)

    except Exception as e:
        log_error(MODULE_NAME, f"알고리즘 텍스트 보정 실패 ({algo_name})", e)
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

def verify_single_speaker(algo_name, audio_path, threshold=0.80):
    return verify_single_speaker_multi_files(algo_name, [audio_path], threshold)

def verify_single_speaker_multi_files(algo_name, audio_paths, threshold=0.80):
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

def verify_multi_speaker(algo_name, audio_path, threshold=0.80):
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

        speaker_embeddings = {}
        speaker_durations = {}

        for turn, _, speaker_label in diarization.itertracks(yield_label=True):

            duration = turn.end - turn.start

            min_duration = 2.0
            if duration < min_duration:
                continue

            start_sample = int(turn.start * target_sr)
            end_sample = int(turn.end * target_sr)

            segment_audio = y[start_sample:end_sample]

            if len(segment_audio) == 0:
                continue

            try:
                normalized_wav = preprocess_wav(segment_audio, source_sr=target_sr)

                if len(normalized_wav) == 0:
                    continue

                utterance_embed = encoder.embed_utterance(normalized_wav)

            except Exception:
                continue

            if utterance_embed is None or len(utterance_embed) == 0:
                continue

            if speaker_label not in speaker_embeddings:
                speaker_embeddings[speaker_label] = []
                speaker_durations[speaker_label] = 0.0

            speaker_embeddings[speaker_label].append(utterance_embed)
            speaker_durations[speaker_label] += duration

        best_similarity = 0.0
        best_speaker = None

        for speaker_label, embeddings in speaker_embeddings.items():

            if not embeddings:
                continue

            speaker_centroid = np.mean(embeddings, axis=0)

            norm = np.linalg.norm(speaker_centroid)

            if norm == 0:
                continue

            speaker_centroid = speaker_centroid / norm

            similarity = _calculate_cosine_similarity(overall_embed, speaker_centroid)

            total_duration = speaker_durations.get(speaker_label, 0.0)

            log_info(MODULE_NAME, f"다중 화자 검증 - 화자: {speaker_label}, 총 발화시간: {total_duration:.2f}초, 구간 수: {len(embeddings)}, 유사도: {similarity:.4f}")

            if similarity > best_similarity:
                best_similarity = similarity
                best_speaker = speaker_label

            if similarity >= threshold:
                log_info(MODULE_NAME, f"다중 화자 검증 통과 (algo: {algo_name}, speaker: {speaker_label}, similarity: {similarity:.4f}, duration: {total_duration:.2f}s)")
                return True

        log_info(MODULE_NAME, f"다중 화자 검증 실패 (algo: {algo_name}, 최고 화자: {best_speaker}, 최고 유사도: {best_similarity:.4f})")
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

    # 신규 폴더 세그먼트 임베딩 추출 및 대표값 계산
    new_centroid, new_embeddings = _extract_speaker_embedding_from_segments(wav_paths)
    if new_centroid is None:
        log_error(MODULE_NAME, f"폴더 내 세그먼트로부터 전체 화자 임베딩 추출 실패 ({folder_path_str})")
        return False

    ensure_handler_directories()
    file_path = os.path.join(STORAGE_DIR, f"{algo_name}.json")
    
    all_embeddings_for_centroid = new_embeddings.copy()
    algo_data = {}
    is_existing_profile = os.path.exists(file_path)

    existing_source_folders = []

    if is_existing_profile:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                algo_data = json.load(f)
            
            existing_embed = np.array(algo_data.get("overall_embedding", []))
            
            raw_old_folder = algo_data.get("source_folder")
            if isinstance(raw_old_folder, list):
                existing_source_folders = raw_old_folder
            elif isinstance(raw_old_folder, str) and raw_old_folder.strip():
                existing_source_folders = [raw_old_folder]

            is_exact_same_files = False
            matched_old_folder = None
            normalized_new_folder = os.path.abspath(folder_path_str)
            new_file_paths = {os.path.abspath(path) for path in wav_paths}

            for old_f in existing_source_folders:
                if not old_f:
                    continue
                try:
                    normalized_old_folder = os.path.abspath(old_f)
                    if normalized_old_folder != normalized_new_folder:
                        continue

                    old_file_paths = {os.path.abspath(os.path.join(old_f, f)) for f in os.listdir(old_f) if f.lower().endswith(".wav")}
                    if old_file_paths == new_file_paths:
                        is_exact_same_files = True
                        matched_old_folder = old_f
                        break
                except Exception:
                    pass
            
            if is_exact_same_files:
                log_info(MODULE_NAME, f"동일한 세부 파일 경로 내 수정/저장 감지: 검증을 생략하고 그대로 업데이트합니다 ({folder_path_str})")
            elif len(existing_embed) > 0:
                similarity_threshold = 0.90  
                similarity = _calculate_cosine_similarity(existing_embed, new_centroid)

                log_info(MODULE_NAME, f"알고리즘 업데이트 화자 검증 - 신규 폴더 대표 임베딩 유사도: {similarity:.4f}")

                if similarity < similarity_threshold:
                    # [변경점] 여기서는 log_error 대신 상세 사유를 담은 info 로그를 남기고 안전하게 False를 반환하여 메시지 중첩 방지
                    log_info(MODULE_NAME, f"알고리즘 업데이트 거부: 신규 경로의 대표 화자가 기존 알고리즘 화자와 일치하지 않습니다. (유사도: {similarity:.4f}, 기준: {similarity_threshold:.2f})")
                    return False

                for old_f in existing_source_folders:
                    if old_f and os.path.exists(old_f):
                        old_wav_files = [os.path.join(old_f, f) for f in os.listdir(old_f) if f.lower().endswith(".wav")]
                        for old_path in old_wav_files:
                            old_embed = _extract_embedding_vector(old_path)
                            if old_embed is not None:
                                all_embeddings_for_centroid.append(old_embed)

        except Exception as e:
            # [변경점] 예외 발생 시 구체적인 에러 메시지를 UI단으로 정확히 전달할 수 있도록 로그 처리 조정
            err_msg = f"기존 임베딩 비교 중 예외 발생 ({algo_name}): {str(e)}"
            log_error(MODULE_NAME, err_msg)
            return False

    # 전체 세그먼트(기존 + 신규)를 종합하여 최종 대표 임베딩 재계산
    if all_embeddings_for_centroid:
        final_centroid = np.mean(all_embeddings_for_centroid, axis=0)
        overall_embedding = final_centroid / np.linalg.norm(final_centroid)
    else:
        overall_embedding = new_centroid

    if not algo_data:
        algo_data = {
            "algo_name": algo_name,
            "overall_embedding": [],
            "samples": []
        }

    existing_samples_map = {
        sample.get("txt_path"): sample 
        for sample in algo_data.get("samples", []) 
        if sample.get("txt_path")
    }

    new_samples_list = []

    # 새로운 폴더가 추가된 경우 기존 샘플들과 병합 유지 처리
    if is_existing_profile and not is_exact_same_files:
        old_samples = algo_data.get("samples", [])
        new_samples_list = old_samples.copy()

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

            new_samples_list.append({"txt_path": str(txt_path), "corrected_text": corrected_text})
    else:
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

    # source_folder 리스트 관리: 기존 목록에 새 폴더가 없으면 추가 (중복 방지)
    if folder_path_str not in existing_source_folders:
        existing_source_folders.append(folder_path_str)

    algo_data["samples"] = new_samples_list
    algo_data["overall_embedding"] = overall_embedding.tolist()
    algo_data["source_folder"] = existing_source_folders if len(existing_source_folders) > 1 else existing_source_folders[0] if existing_source_folders else folder_path_str

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

        source_folders = algo_data.get("source_folder")
        if isinstance(source_folders, str):
            source_folders = [source_folders]
        elif not isinstance(source_folders, list):
            source_folders = []

        for source_folder in source_folders:
            if not source_folder or not os.path.exists(source_folder):
                continue

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
import sys
import shutil
import warnings
import logging
import re
import json
import time
from pathlib import Path
import soundfile as sf
import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.*")
warnings.filterwarnings("ignore", message=".*torchcodec is not installed correctly.*")
warnings.filterwarnings("ignore", message=".*triton not found.*")
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from qwen_asr import Qwen3ASRModel
import algorithm_handler as ah
import asr_postprocessor as post_processor
from error_logger import log_error, log_info

MODULE_NAME = "AudioProcessor"

try:
    from audio_separator.separator import Separator
    HAS_UVR5 = True
except ImportError as e:
    HAS_UVR5 = False
    log_error(MODULE_NAME, "audio-separator 패키지 로드 실패 (UVR5 기능을 사용할 수 없습니다)", e)

try:
    from pyannote.audio import Pipeline
    HAS_PYANNOTE = True
except ImportError as e:
    HAS_PYANNOTE = False
    log_error(MODULE_NAME, "pyannote.audio 패키지 로드 실패 (화자 분리 기능을 사용할 수 없습니다)", e)

AUDIO_DIR = Path("audio")
AUTO_REC_DIR = AUDIO_DIR / "auto_recorded_audio"
MANUAL_UVR5_DIR = AUDIO_DIR / "uvr5"
SEGMENTS_BASE_DIR = Path("segments_base")
ASR_DIR = Path("asr_output")
POST_DIR = Path("post_processing")
ERROR_LOG_DIR = Path("error_log")
TOKEN_FILE = Path("hf_token.txt")
CONFIG_FILE = Path("config.json")

TARGET_SAMPLE_RATE = 16000
DEVICE = "cuda"
DEFAULT_MODEL_PATH = "Qwen/Qwen3-ASR-1.7B"

_cached_asr_model = None
_uvr5_separator_instance = None

def load_config():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log_info(MODULE_NAME, "config.json 로드 실패 (기본 설정 사용)")
    return {}

CONFIG = load_config()

def ensure_directories():
    for d in [
        AUDIO_DIR,
        MANUAL_UVR5_DIR,
        AUTO_REC_DIR,
        SEGMENTS_BASE_DIR,
        ASR_DIR,
        POST_DIR,
        ERROR_LOG_DIR
    ]:
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log_error(MODULE_NAME, f"디렉토리 생성 실패 ({d})", e)
            
    try:
        ah.ensure_handler_directories()
    except Exception as e:
        log_error(MODULE_NAME, "algorithm_handler 디렉토리 생성 실패", e)

def get_huggingface_token():
    if TOKEN_FILE.exists():
        try:
            token = TOKEN_FILE.read_text(encoding="utf-8").strip()
            if token: return token
        except Exception as e:
            log_error(MODULE_NAME, "허깅페이스 토큰 파일 읽기 실패", e)
    return None

def format_time(seconds):
    try:
        if seconds < 60:
            return f"{seconds:.1f}초"
        elif seconds < 3600:
            m = int(seconds // 60)
            s = seconds % 60
            return f"{m}분 {s:.1f}초"
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = seconds % 60
            return f"{h}시간 {m}분 {s:.1f}초"
    except Exception as e:
        return f"{seconds}초"

def format_mmss(seconds):
    try:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"
    except Exception:
        return "00:00"

def save_wav_chunk(audio_data, sample_rate, filename):
    import wave
    try:
        scaled_audio = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
        with wave.open(str(filename), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(scaled_audio.tobytes())
    except Exception as e:
        log_error(MODULE_NAME, f"WAV 청크 저장 실패 ({filename})", e)

def load_asr_model():
    global _cached_asr_model
    if _cached_asr_model is not None:
        return _cached_asr_model

    log_info(MODULE_NAME, f"ASR 모델 로딩 중 ({DEFAULT_MODEL_PATH})")
    try:
        torch.backends.cudnn.benchmark = True
        if CONFIG.get("enable_gpu_cache_clear", True):
            torch.cuda.empty_cache()

        model = Qwen3ASRModel.from_pretrained(
            DEFAULT_MODEL_PATH,
            dtype=torch.float16,
            device_map=DEVICE
        )
        _cached_asr_model = model
        log_info(MODULE_NAME, "ASR 모델 로드 완료")
        return model
    except Exception as e:
        log_error(MODULE_NAME, f"ASR 모델 로드 실패 ({DEFAULT_MODEL_PATH})", e)
        return None

def resample_audio(audio_data, original_sr, target_sr=TARGET_SAMPLE_RATE):
    try:
        if original_sr == target_sr:
            return audio_data.astype(np.float32)
        duration = audio_data.shape[0] / float(original_sr)
        target_length = int(round(duration * target_sr))
        if target_length <= 0:
            return np.zeros((0,), dtype=np.float32)
        orig_times = np.linspace(0.0, duration, num=audio_data.shape[0], endpoint=False)
        target_times = np.linspace(0.0, duration, num=target_length, endpoint=False)
        return np.interp(target_times, orig_times, audio_data).astype(np.float32)
    except Exception as e:
        return audio_data.astype(np.float32)

def apply_uvr5_vocal_extraction(input_audio_path):
    global _uvr5_separator_instance
    if not HAS_UVR5:
        err_msg = "audio-separator 패키지가 없어 UVR5 분기를 수행할 수 없습니다."
        log_info(MODULE_NAME, err_msg)
        raise RuntimeError(err_msg)

    abs_input_path = Path(input_audio_path).resolve()
    
    if AUTO_REC_DIR.resolve() in abs_input_path.parents:
        target_uvr5_dir = abs_input_path.parent / "uvr5"
    else:
        target_uvr5_dir = MANUAL_UVR5_DIR
        
    target_uvr5_dir.mkdir(parents=True, exist_ok=True)

    base_name = abs_input_path.stem
    expected_vocal_path = target_uvr5_dir / f"{base_name}_Vocals.wav"
    if expected_vocal_path.exists():
        return str(expected_vocal_path)

    log_info(MODULE_NAME, f"UVR5 보컬 분리 시작: {base_name}")
    uvr_start_time = time.time()
    
    try:
        if _uvr5_separator_instance is None:
            _uvr5_separator_instance = Separator(output_dir=str(target_uvr5_dir))
        else:
            _uvr5_separator_instance.output_dir = str(target_uvr5_dir)
            
        _uvr5_separator_instance.load_model('UVR-MDX-NET-Voc_FT.onnx')
        output_files = _uvr5_separator_instance.separate(str(abs_input_path))
        
        vocal_file = None
        for f in output_files:
            if "vocals" in f.lower():
                full_path = target_uvr5_dir / f
                new_vocal_path = target_uvr5_dir / f"{base_name}_Vocals.wav"
                if full_path.exists():
                    if new_vocal_path.exists(): new_vocal_path.unlink()
                    full_path.rename(new_vocal_path)
                vocal_file = new_vocal_path
                
        if vocal_file and vocal_file.exists():
            log_info(MODULE_NAME, f"UVR5 보컬 분리 완료 (소요 시간: {format_time(time.time() - uvr_start_time)})")
            return str(vocal_file)
            
        raise RuntimeError(f"UVR5 Vocals 파일을 찾지 못했습니다: {base_name}")
    except Exception as e:
        log_error(MODULE_NAME, f"UVR5 보컬 분리 프로세스 중 예외 발생 ({base_name})", e)
        raise

def configure_strict_analysis_pipeline(audio_data, sample_rate):
    while True:
        print("\n================================================")
        print("                분석 모드 선택")
        print("================================================")
        print(" 0. 기본 분석")
        print(" 1. 단일 화자 알고리즘 분석")
        print(" 2. 다중 화자 알고리즘 분석")
        print(" 3. 메뉴로 돌아가기")
        print("------------------------------------------------")
        try:
            mode = input("선택: ").strip()
        except Exception:
            continue
        
        if mode == "3":
            return None, False, 3
        if mode == "0":
            return [], False, 0

        try:
            existing = ah.load_existing_profiles()
        except Exception:
            existing = []
            
        if mode == "1":
            if not existing:
                print("[알림] 등록된 프로파일이 없습니다.")
                continue
            try:
                matched_speakers = [algo_name for algo_name in existing if ah.verify_single_speaker(algo_name, audio_data)]
            except Exception:
                matched_speakers = []

            if not matched_speakers:
                print("[차단] 일치하는 화자 프로파일이 없습니다.")
                continue
            return [matched_speakers[0]], True, 1
            
        elif mode == "2":
            if not existing:
                print("[알림] 등록된 프로파일이 없습니다.")
                continue
            try:
                success, msg = ah.verify_multi_speakers_auto(audio_data)
            except Exception as e:
                success, msg = False, str(e)

            if not success:
                print(f"[차단] {msg}")
                continue
            return existing, False, 2
        else:
            print("[오류] 올바른 번호를 입력해주세요.")

def print_clean_stage_progress(current_item, total_items, start_time):
    if total_items <= 0:
        percent = 100
        sub_progress = 1.0
    else:
        percent = int((current_item / total_items) * 100)
        percent = min(100, max(0, percent))
        sub_progress = current_item / total_items

    now = time.time()
    elapsed = max(0.0, now - start_time)
    
    if sub_progress > 0.0 and current_item < total_items:
        estimated_total = elapsed / sub_progress
        eta = max(0.0, estimated_total - elapsed)
    else:
        eta = 0.0

    elapsed_str = format_mmss(elapsed)
    eta_str = format_mmss(eta)

    terminal_width = shutil.get_terminal_size(fallback=(120, 24)).columns
    prefix = f"{percent:3d}% |"
    suffix = f"| {current_item}/{total_items} {elapsed_str}<{eta_str}"
    
    fixed_width = len(prefix) + len(suffix)
    bar_len = max(10, terminal_width - fixed_width - 2)
    
    filled_len = int(bar_len * sub_progress)
    filled_len = min(bar_len, max(0, filled_len))
    empty_len = bar_len - filled_len
    
    bar_str = "█" * filled_len + " " * empty_len

    if current_item >= total_items:
        sys.stdout.write("\r" + prefix + bar_str + suffix + "\n")
        sys.stdout.flush()
    else:
        sys.stdout.write("\r" + prefix + bar_str + suffix)
        sys.stdout.flush()

def execute_batch_analysis_flow(model, sub_wavs, active_speakers, is_single, analysis_mode=0, target_folder_name=""):
    log_info(MODULE_NAME, f"통합 일괄 분석 실행: {target_folder_name} (총 {len(sub_wavs)}개 파일)")
    try:
        if model is None:
            log_error(MODULE_NAME, "분석 실행 실패: 모델이 로드되어 있지 않습니다.")
            return

        parent_audio_dir = SEGMENTS_BASE_DIR / target_folder_name
        parent_audio_dir.mkdir(parents=True, exist_ok=True)
            
        prefix_str = "seg_single" if is_single and active_speakers else "segment"
        existing_subdirs = [d for d in parent_audio_dir.iterdir() if d.is_dir() and d.name.startswith(prefix_str)]
        specific_segment_dir = parent_audio_dir / f"{prefix_str}_{len(existing_subdirs) + 1:03d}"
        specific_segment_dir.mkdir(parents=True, exist_ok=True)

        global_time_offset = 0.0
        total_files = len(sub_wavs)
        file_audio_cache = []

        if sub_wavs:
            target_uvr5_dir = Path(sub_wavs[0]).parent / "uvr5"
        else:
            target_uvr5_dir = AUTO_REC_DIR / target_folder_name / "uvr5" if target_folder_name else MANUAL_UVR5_DIR
            
        target_uvr5_dir.mkdir(parents=True, exist_ok=True)

        all_vocals_exist = True
        for file_path in sub_wavs:
            base_name = Path(file_path).stem
            expected_vocal_path = target_uvr5_dir / f"{base_name}_Vocals.wav"
            if not expected_vocal_path.exists():
                all_vocals_exist = False
                break

        if all_vocals_exist and total_files > 0:
            print(f"\n[알림] UVR5 파일이 존재하여 스킵합니다.")
        else:
            log_info(MODULE_NAME, f"UVR5 보컬 분리 시작 (총 {total_files}개 파일)")
            stage1_start = time.time()
            
            for idx, file_path in enumerate(sub_wavs, 1):
                file_name = Path(file_path).name
                print(f"\n[시작] UVR5 보컬 분리 수행 중: {file_name} ({idx}/{total_files})")
                
                try:
                    apply_uvr5_vocal_extraction(file_path)
                    print(f"[완료] UVR5 보컬 분리 완료: {file_name}")
                except Exception as e:
                    log_error(MODULE_NAME, f"특정 파일 UVR5 처리 중 예외 발생 ({file_name})", e)
                    print(f"[실패] UVR5 보컬 분리 중 오류 발생: {file_name}")
            
            log_info(MODULE_NAME, f"UVR5 보컬 분리 통합 완료 (총 {total_files}개 파일, 소요 시간: {format_time(time.time() - stage1_start)})")

        for file_path in sub_wavs:
            file_path_obj = Path(file_path)
            base_name = file_path_obj.stem
            expected_vocal_path = target_uvr5_dir / f"{base_name}_Vocals.wav"
            processed_vocal_path = expected_vocal_path if expected_vocal_path.exists() else file_path_obj

            vocal_audio_data, sr = sf.read(str(processed_vocal_path))
            if vocal_audio_data.ndim > 1:
                vocal_audio_data = np.mean(vocal_audio_data, axis=1)
            vocal_audio_data = vocal_audio_data.astype(np.float32)
            vocal_audio_data = resample_audio(vocal_audio_data, sr, TARGET_SAMPLE_RATE)

            max_val = np.max(np.abs(vocal_audio_data))
            if max_val > 0.0001:
                vocal_audio_data = vocal_audio_data / max_val

            file_duration = vocal_audio_data.shape[0] / TARGET_SAMPLE_RATE
            file_audio_cache.append((file_path_obj, vocal_audio_data, file_duration))

        print(f"\n화자 분리 수행 중 (총 {total_files}개 파일)")
        stage2_start = time.time()
        batch_speaker_turns = []
        for f_idx, (file_path_obj, vocal_audio_data, file_duration) in enumerate(file_audio_cache, 1):
            print_clean_stage_progress(f_idx, total_files, stage2_start)

            raw_speaker_turns = []
            if HAS_PYANNOTE and not is_single:
                token = get_huggingface_token()
                if token:
                    try:
                        if CONFIG.get("enable_gpu_cache_clear", True):
                            torch.cuda.empty_cache()

                        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
                        pipeline.to(torch.device(DEVICE))
                        tensor_audio = torch.from_numpy(vocal_audio_data).unsqueeze(0).to(torch.float32).to(DEVICE)
                        diarization = pipeline({"waveform": tensor_audio, "sample_rate": TARGET_SAMPLE_RATE})
                        annotation = getattr(diarization, "speaker_diarization", diarization)
                        
                        for turn, _, speaker in annotation.itertracks(yield_label=True):
                            if (turn.end - turn.start) >= 1.0:
                                raw_speaker_turns.append((turn.start + global_time_offset, turn.end + global_time_offset, speaker))
                    except Exception as ex:
                        log_error(MODULE_NAME, f"화자 분리 수행 중 예외 발생 ({file_path_obj.name})", ex)
            
            if not raw_speaker_turns:
                raw_speaker_turns.append((global_time_offset, global_time_offset + file_duration, "SPEAKER_00"))

            batch_speaker_turns.extend(raw_speaker_turns)
            global_time_offset += file_duration

        speaker_mapping = {}
        unique_orig_speakers = sorted({s for _, _, s in batch_speaker_turns})
        for idx_s, orig_s in enumerate(unique_orig_speakers):
            if active_speakers and len(active_speakers) > 1:
                speaker_mapping[orig_s] = active_speakers[idx_s] if idx_s < len(active_speakers) else orig_s
            elif active_speakers and len(active_speakers) == 1:
                speaker_mapping[orig_s] = active_speakers[0]
            else:
                speaker_mapping[orig_s] = orig_s

        total_turns = len(batch_speaker_turns)
        print(f"\n구간별 ASR 인식 및 문장 정제 시작 (총 {total_turns}개 구간)")
        stage3_start = time.time()
        temp_segments = []
        recent_texts = [] 
        full_vocal_audio_data = np.concatenate([v_data for _, v_data, _ in file_audio_cache])

        for idx, (start, end, orig_speaker) in enumerate(batch_speaker_turns, 1):
            print_clean_stage_progress(idx, total_turns, stage3_start)

            mapped_speaker = speaker_mapping.get(orig_speaker, "SPEAKER_00")
            start_sample = int(start * TARGET_SAMPLE_RATE)
            end_sample = int(end * TARGET_SAMPLE_RATE)
            chunk_audio = full_vocal_audio_data[start_sample:end_sample]
            
            chunk_text = ""
            if chunk_audio.size > 0:
                try:
                    from transformers import logging as hf_logging
                    hf_logging.set_verbosity_error()
                    
                    with torch.no_grad():
                        res_chunk = model.transcribe((chunk_audio, TARGET_SAMPLE_RATE))
                    
                    if hasattr(res_chunk, "text"):
                        chunk_text = res_chunk.text
                    elif isinstance(res_chunk, list):
                        chunk_text = " ".join([str(item.get("text", item)) if isinstance(item, dict) else str(getattr(item, "text", item)) for item in res_chunk])
                    else:
                        chunk_text = str(res_chunk)
                except Exception as e:
                    log_error(MODULE_NAME, f"구간 ASR 변환 실패 ({start:.1f}s ~ {end:.1f}s)", e)

            chunk_txt = post_processor.process_segment_text(
                chunk_text,
                recent_texts=recent_texts,
                current_start=start,
                mapped_speaker=mapped_speaker
            )

            if not chunk_txt.strip():
                continue
            
            temp_segments.append({
                "start": start,
                "end": end,
                "speaker": mapped_speaker,
                "text": chunk_txt,
                "audio": chunk_audio
            })

        final_segments = []
        for seg in temp_segments:
            if not final_segments:
                final_segments.append(seg)
                continue

            prev = final_segments[-1]
            time_gap = seg["start"] - prev["end"]
            prev_text_stripped = prev["text"].strip()
            
            is_sentence_incomplete = not prev_text_stripped.endswith(('.', '!', '?', '~'))
            is_same_speaker = (prev["speaker"] == seg["speaker"])
            is_close_gap = (time_gap < 0.5)

            if is_same_speaker and is_sentence_incomplete and is_close_gap:
                prev["end"] = seg["end"]
                prev["text"] = f"{prev['text']} {seg['text']}".strip()
                try:
                    combined_audio = np.concatenate([prev["audio"], seg["audio"]])
                    prev["audio"] = combined_audio
                except Exception:
                    pass
            else:
                final_segments.append(seg)

        POST_DIR.mkdir(exist_ok=True)
        ASR_DIR.mkdir(exist_ok=True)
        
        post_file_path = POST_DIR / f"{target_folder_name}_post_001.txt"
        asr_file_path = ASR_DIR / f"{target_folder_name}_asr_001.txt"
        
        p_idx = 1
        while post_file_path.exists():
            post_file_path = POST_DIR / f"{target_folder_name}_post_{p_idx:03d}.txt"
            p_idx += 1
        a_idx = 1
        while asr_file_path.exists():
            asr_file_path = ASR_DIR / f"{target_folder_name}_asr_{a_idx:03d}.txt"
            a_idx += 1

        total_segments = len(final_segments)
        print(f"\n최종 세그먼트 및 대화 로그 저장 시작 (총 {total_segments}개 세그먼트)")
        stage4_start = time.time()
        saved_segment_count = 0

        with open(post_file_path, "w", encoding="utf-8") as post_file_obj, open(asr_file_path, "w", encoding="utf-8") as asr_file_obj:
            post_file_obj.write(f"=== 대화 로그 (자동녹화: {target_folder_name}) ===\n\n")

            for idx, seg in enumerate(final_segments, 1):
                print_clean_stage_progress(idx, total_segments, stage4_start)

                saved_segment_count += 1
                safe_speaker_name = str(seg["speaker"]).replace("/", "_").replace("\\", "_")
                
                if analysis_mode == 0:
                    base_seg_name = f"seg_sub_{saved_segment_count-1:03d}_{safe_speaker_name}_{seg['start']:.1f}s-{seg['end']:.1f}s"
                elif analysis_mode == 1:
                    base_seg_name = f"seg_sub_{saved_segment_count-1:03d}_A_{safe_speaker_name}_{seg['start']:.1f}s-{seg['end']:.1f}s"
                else:
                    base_seg_name = f"seg_sub_{saved_segment_count-1:03d}_B_{safe_speaker_name}_{seg['start']:.1f}s-{seg['end']:.1f}s"

                save_wav_chunk(seg["audio"], TARGET_SAMPLE_RATE, specific_segment_dir / f"{base_seg_name}.wav")
                (specific_segment_dir / f"{base_seg_name}.txt").write_text(seg['text'], encoding="utf-8")

                log_line = f"[{seg['speaker']}] ({format_time(seg['start'])} ~ {format_time(seg['end'])}): {seg['text']}"
                post_file_obj.write(log_line + "\n")
                asr_file_obj.write(log_line + "\n")

        log_info(MODULE_NAME, f"통합 분석 완료! 유효 세그먼트: {saved_segment_count}개")
        print(f"\n[알림] 저장이 완료되었습니다. {specific_segment_dir}")

    except Exception as e:
        log_error(MODULE_NAME, f"통합 분석 실행 플로우 중 치명적인 예외 발생 ({target_folder_name})", e)

def execute_analysis_flow(model, file_path, active_speakers, is_single, analysis_mode=0):
    log_info(MODULE_NAME, f"분석 실행: {Path(file_path).name}")
    try:
        if model is None:
            return

        raw_source_stem = Path(file_path).stem
        clean_source_name = re.sub(r'_Vocals$', '', raw_source_stem, flags=re.IGNORECASE)
        
        parent_audio_dir = SEGMENTS_BASE_DIR / clean_source_name
        parent_audio_dir.mkdir(parents=True, exist_ok=True)
            
        prefix_str = "seg_single" if is_single and active_speakers else "segment"
        existing_subdirs = [d for d in parent_audio_dir.iterdir() if d.is_dir() and d.name.startswith(prefix_str)]
        specific_segment_dir = parent_audio_dir / f"{prefix_str}_{len(existing_subdirs) + 1:03d}"
        specific_segment_dir.mkdir(parents=True, exist_ok=True)

        abs_input_path = Path(file_path).resolve()
        if AUTO_REC_DIR.resolve() in abs_input_path.parents:
            target_uvr5_dir = abs_input_path.parent / "uvr5"
        else:
            target_uvr5_dir = MANUAL_UVR5_DIR
            
        expected_vocal_path = target_uvr5_dir / f"{abs_input_path.stem}_Vocals.wav"

        if expected_vocal_path.exists():
            print("\n[알림] UVR5 파일이 존재하여 스킵합니다.")
            processed_vocal_path = str(expected_vocal_path)
        else:
            print(f"\n[시작] 단일 파일 UVR5 보컬 분리 수행 중: {Path(file_path).name}")
            stage1_start = time.time()
            processed_vocal_path = apply_uvr5_vocal_extraction(file_path)
            print(f"[완료] UVR5 보컬 분리 완료 (소요 시간: {format_time(time.time() - stage1_start)})")
        
        vocal_audio_data, sr = sf.read(str(processed_vocal_path))
        if vocal_audio_data.ndim > 1:
            vocal_audio_data = np.mean(vocal_audio_data, axis=1)
        vocal_audio_data = vocal_audio_data.astype(np.float32)
        vocal_audio_data = resample_audio(vocal_audio_data, sr, TARGET_SAMPLE_RATE)

        max_val = np.max(np.abs(vocal_audio_data))
        if max_val > 0.0001:
            vocal_audio_data = vocal_audio_data / max_val

        total_duration = vocal_audio_data.shape[0] / TARGET_SAMPLE_RATE

        print(f"\n화자 분리 수행 중")
        stage2_start = time.time()
        print_clean_stage_progress(1, 1, stage2_start)
        raw_speaker_turns = []
        
        if HAS_PYANNOTE and not is_single:
            token = get_huggingface_token()
            if token:
                try:
                    if CONFIG.get("enable_gpu_cache_clear", True):
                        torch.cuda.empty_cache()

                    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
                    pipeline.to(torch.device(DEVICE))
                    tensor_audio = torch.from_numpy(vocal_audio_data).unsqueeze(0).to(torch.float32).to(DEVICE)
                    diarization = pipeline({"waveform": tensor_audio, "sample_rate": TARGET_SAMPLE_RATE})
                    annotation = getattr(diarization, "speaker_diarization", diarization)
                    
                    for turn, _, speaker in annotation.itertracks(yield_label=True):
                        if (turn.end - turn.start) >= 1.0:
                            raw_speaker_turns.append((turn.start, turn.end, speaker))
                except Exception as ex:
                    log_error(MODULE_NAME, "화자 분리 수행 중 예외 발생", ex)
        
        if not raw_speaker_turns:
            raw_speaker_turns.append((0.0, total_duration, "SPEAKER_00"))

        speaker_mapping = {}
        unique_orig_speakers = sorted({s for _, _, s in raw_speaker_turns})
        for idx, orig_s in enumerate(unique_orig_speakers):
            if active_speakers and len(active_speakers) > 1:
                speaker_mapping[orig_s] = active_speakers[idx] if idx < len(active_speakers) else orig_s
            elif active_speakers and len(active_speakers) == 1:
                speaker_mapping[orig_s] = active_speakers[0]
            else:
                speaker_mapping[orig_s] = orig_s

        total_turns = len(raw_speaker_turns)
        print(f"\n구간별 ASR 인식 및 문장 정제 시작 (총 {total_turns}개 구간)")
        stage3_start = time.time()
        temp_segments = []
        recent_texts = [] 
        
        for idx, (start, end, orig_speaker) in enumerate(raw_speaker_turns, 1):
            print_clean_stage_progress(idx, total_turns, stage3_start)

            mapped_speaker = speaker_mapping.get(orig_speaker, "SPEAKER_00")
            start_sample = int(start * TARGET_SAMPLE_RATE)
            end_sample = int(end * TARGET_SAMPLE_RATE)
            chunk_audio = vocal_audio_data[start_sample:end_sample]
            
            chunk_text = ""
            if chunk_audio.size > 0:
                try:
                    from transformers import logging as hf_logging
                    hf_logging.set_verbosity_error()
                    
                    with torch.no_grad():
                        res_chunk = model.transcribe((chunk_audio, TARGET_SAMPLE_RATE))
                    
                    if hasattr(res_chunk, "text"):
                        chunk_text = res_chunk.text
                    elif isinstance(res_chunk, list):
                        chunk_text = " ".join([str(item.get("text", item)) if isinstance(item, dict) else str(getattr(item, "text", item)) for item in res_chunk])
                    else:
                        chunk_text = str(res_chunk)
                except Exception as e:
                    log_error(MODULE_NAME, f"구간 {idx} ASR 변환 실패 ({start:.1f}s ~ {end:.1f}s)", e)

            chunk_txt = post_processor.process_segment_text(
                chunk_text,
                recent_texts=recent_texts,
                current_start=start,
                mapped_speaker=mapped_speaker
            )

            temp_segments.append({
                "start": start,
                "end": end,
                "speaker": mapped_speaker,
                "text": chunk_txt,
                "audio": chunk_audio
            })

        merged_segments = []
        for seg in temp_segments:
            if not merged_segments:
                merged_segments.append(seg)
                continue

            prev = merged_segments[-1]
            time_gap = seg["start"] - prev["end"]
            prev_text_stripped = prev["text"].strip()
            
            is_sentence_incomplete = not prev_text_stripped.endswith(('.', '!', '?', '~'))
            is_same_speaker = (prev["speaker"] == seg["speaker"])
            is_close_gap = (time_gap < 0.5)

            if is_same_speaker and is_sentence_incomplete and is_close_gap:
                start_sample = int(prev["start"] * TARGET_SAMPLE_RATE)
                end_sample = int(seg["end"] * TARGET_SAMPLE_RATE)
                new_chunk_audio = vocal_audio_data[start_sample:end_sample]

                prev["end"] = seg["end"]
                prev["text"] = f"{prev['text']} {seg['text']}".strip()
                prev["audio"] = new_chunk_audio
            else:
                merged_segments.append(seg)

        POST_DIR.mkdir(exist_ok=True)
        ASR_DIR.mkdir(exist_ok=True)
        
        post_file_path = POST_DIR / f"{clean_source_name}_post_{len(list(POST_DIR.glob(f'{clean_source_name}_post_*.txt'))) + 1:03d}.txt"
        asr_file_path = ASR_DIR / f"{clean_source_name}_asr_{len(list(ASR_DIR.glob(f'{clean_source_name}_asr_*.txt'))) + 1:03d}.txt"

        stage4_start = time.time()
        saved_segment_count = 0
        total_segments = len(merged_segments)

        print(f"\n최종 세그먼트 및 대화 로그 저장 시작 (총 {total_segments}개 세그먼트)")
        with open(post_file_path, "w", encoding="utf-8") as post_file_obj, open(asr_file_path, "w", encoding="utf-8") as asr_file_obj:
            post_file_obj.write(f"=== 대화 로그 ({file_path}) ===\n\n")

            for idx, seg in enumerate(merged_segments, 1):
                print_clean_stage_progress(idx, total_segments, stage4_start)

                saved_segment_count += 1
                safe_speaker_name = str(seg["speaker"]).replace("/", "_").replace("\\", "_")
                
                if analysis_mode == 0:
                    base_seg_name = f"seg_sub_{saved_segment_count-1:03d}_{safe_speaker_name}_{seg['start']:.1f}s-{seg['end']:.1f}s"
                elif analysis_mode == 1:
                    base_seg_name = f"seg_sub_{saved_segment_count-1:03d}_A_{safe_speaker_name}_{seg['start']:.1f}s-{seg['end']:.1f}s"
                else:
                    base_seg_name = f"seg_sub_{saved_segment_count-1:03d}_B_{safe_speaker_name}_{seg['start']:.1f}s-{seg['end']:.1f}s"

                save_wav_chunk(seg["audio"], TARGET_SAMPLE_RATE, specific_segment_dir / f"{base_seg_name}.wav")
                (specific_segment_dir / f"{base_seg_name}.txt").write_text(seg['text'], encoding="utf-8")

                log_line = f"[{seg['speaker']}] ({format_time(seg['start'])} ~ {format_time(seg['end'])}): {seg['text']}"
                post_file_obj.write(log_line + "\n")
                asr_file_obj.write(log_line + "\n")

        print(f"\n[알림] 저장이 완료되었습니다. {specific_segment_dir}")
        log_info(MODULE_NAME, f"분석 완료! 유효 세그먼트: {saved_segment_count}개")

    except Exception as e:
        log_error(MODULE_NAME, f"분석 실행 플로우 중 치명적인 예외 발생 ({file_path})", e)

def select_and_process_audio_file(model=None):
    ensure_directories()
    
    if not AUDIO_DIR.exists():
        print("\n[알림] 'audio' 폴더가 존재하지 않습니다.")
        return

    def process_target_file(target_file):
        current_model = model or load_asr_model()
        if current_model is None:
            print("[오류] 모델을 로드할 수 없습니다.")
            return

        try:
            sample_audio_data, sr = sf.read(str(target_file))
            if sample_audio_data.ndim > 1:
                sample_audio_data = np.mean(sample_audio_data, axis=1)
            sample_audio_data = sample_audio_data.astype(np.float32)
            sample_audio_data = resample_audio(sample_audio_data, sr, TARGET_SAMPLE_RATE)
                
            active_speakers, is_single, analysis_mode = configure_strict_analysis_pipeline(sample_audio_data, TARGET_SAMPLE_RATE)
            if active_speakers is not None or analysis_mode != 3:
                execute_analysis_flow(current_model, str(target_file), active_speakers, is_single, analysis_mode)
            else:
                print("[알림] 분석이 취소되었습니다.")
        except Exception as e:
            log_error(MODULE_NAME, f"파일 처리 중 예외 발생 ({target_file.name})", e)

    while True:
        try:
            root_wavs = sorted([f for f in AUDIO_DIR.glob("*.wav") if "vocal" not in f.name.lower() and "instrumental" not in f.name.lower()], key=lambda x: x.name)
        except Exception as e:
            root_wavs = []

        menu_items = [('auto', '자동녹화')] + [('file', wf) for wf in root_wavs]

        print("\n==================================================")
        print("                    오디오")
        print("==================================================")
        for idx, item in enumerate(menu_items, 1):
            name = item[1].name if item[0] == 'file' else item[1]
            print(f" {idx:2d}. {name}")
        
        back_idx = len(menu_items) + 1
        print(f" {back_idx:2d}. 메뉴로 돌아가기")
        print("--------------------------------------------------")
        
        choice = input("분석할 항목 번호를 선택하세요: ").strip()
        if not choice.isdigit():
            print("[오류] 올바른 번호를 입력해주세요.")
            continue
            
        choice_val = int(choice)
        if choice_val == back_idx:
            return
            
        adjusted_idx = choice_val - 1
        if not (0 <= adjusted_idx < len(menu_items)):
            print("[오류] 잘못된 번호입니다. 다시 선택해주세요.")
            continue
            
        item_type, item_data = menu_items[adjusted_idx]
        
        if item_type == 'auto':
            while True:
                try:
                    auto_subdirs = sorted([d for d in AUTO_REC_DIR.iterdir() if d.is_dir() and d.name.lower() != "uvr5"], key=lambda x: x.name) if AUTO_REC_DIR.exists() else []
                except Exception:
                    auto_subdirs = []

                if not auto_subdirs:
                    print("\n[알림] 생성된 자동녹화 폴더가 존재하지 않습니다.")
                    break

                sub_menu_items = [('folder', sd) for sd in auto_subdirs]

                print("\n==================================================")
                print("                자동녹화 폴더 목록")
                print("==================================================")
                for idx, item in enumerate(sub_menu_items, 1):
                    print(f" {idx:2d}. {item[1].name}")
                
                sub_back_idx = len(sub_menu_items) + 1
                print(f" {sub_back_idx:2d}. 메뉴로 돌아가기")
                print("--------------------------------------------------")
                
                sub_choice = input("탐색할 폴더 번호를 선택하세요: ").strip()
                if not sub_choice.isdigit():
                    print("[오류] 올바른 번호를 입력해주세요.")
                    continue
                    
                sub_val = int(sub_choice)
                if sub_val == sub_back_idx:
                    break
                    
                sub_adjusted_idx = sub_val - 1
                if not (0 <= sub_adjusted_idx < len(sub_menu_items)):
                    print("[오류] 잘못된 번호입니다. 다시 선택해주세요.")
                    continue
                    
                target_folder = sub_menu_items[sub_adjusted_idx][1]
                try:
                    sub_wavs = sorted(list(target_folder.glob("*.wav")), key=lambda x: x.name)
                except Exception:
                    sub_wavs = []

                if not sub_wavs:
                    print("[알림] 해당 폴더에 WAV 파일이 없습니다.")
                    input("계속하려면 엔터를 누르세요...")
                    continue

                current_model = model or load_asr_model()
                if current_model is None:
                    print("[오류] 모델을 로드할 수 없습니다.")
                    return

                try:
                    sample_audio_data, sr = sf.read(str(sub_wavs[0]))
                    if sample_audio_data.ndim > 1:
                        sample_audio_data = np.mean(sample_audio_data, axis=1)
                    sample_audio_data = sample_audio_data.astype(np.float32)
                    sample_audio_data = resample_audio(sample_audio_data, sr, TARGET_SAMPLE_RATE)
                        
                    active_speakers, is_single, analysis_mode = configure_strict_analysis_pipeline(sample_audio_data, TARGET_SAMPLE_RATE)
                    if active_speakers is None and analysis_mode == 3:
                        print("[알림] 분석이 취소되었습니다.")
                        return
                except Exception as e:
                    log_error(MODULE_NAME, "분석 파이프라인 설정 중 예외 발생", e)
                    return

                execute_batch_analysis_flow(current_model, sub_wavs, active_speakers, is_single, analysis_mode, target_folder.name)
                return
        else:
            process_target_file(item_data)
            return
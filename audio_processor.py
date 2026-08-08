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

# 내부 라이브러리 로그 출력 차단
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.*")
warnings.filterwarnings("ignore", message=".*torchcodec is not installed correctly.*")
warnings.filterwarnings("ignore", message=".*triton not found.*")
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from qwen_asr import Qwen3ASRModel
import algorithm_handler as ah

# ASR 후처리 및 정제 모듈 연동
import asr_postprocessor as post_processor

# 표준 공통 에러 로거 연동
from error_logger import log_error, log_info

MODULE_NAME = "AudioProcessor"

try:
    from audio_separator.separator import Separator
    HAS_UVR5 = True
except ImportError:
    HAS_UVR5 = False

try:
    from pyannote.audio import Pipeline
    HAS_PYANNOTE = True
except ImportError:
    HAS_PYANNOTE = False

# 디렉토리 구조 설정
AUDIO_DIR = Path("audio")
AUTO_REC_DIR = AUDIO_DIR / "auto_recorded_audio"
AUTO_UVR5_DIR = AUTO_REC_DIR / "uvr5"
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
            log_error(MODULE_NAME, "config.json 로드 실패", e)
    return {}

CONFIG = load_config()

def ensure_directories():
    for d in [
        AUDIO_DIR,
        MANUAL_UVR5_DIR,
        AUTO_REC_DIR,
        AUTO_UVR5_DIR,
        SEGMENTS_BASE_DIR,
        ASR_DIR,
        POST_DIR,
        ERROR_LOG_DIR
    ]:
        d.mkdir(parents=True, exist_ok=True)
    ah.ensure_handler_directories()

def get_huggingface_token():
    if TOKEN_FILE.exists():
        try:
            token = TOKEN_FILE.read_text(encoding="utf-8").strip()
            if token: return token
        except Exception as e:
            log_error(MODULE_NAME, "허깅페이스 토큰 파일 읽기 실패", e)
    return None

def format_time(seconds):
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

    log_info(MODULE_NAME, f"ASR 모델 로딩 중... ({DEFAULT_MODEL_PATH})")
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
    if original_sr == target_sr:
        return audio_data.astype(np.float32)
    duration = audio_data.shape[0] / float(original_sr)
    target_length = int(round(duration * target_sr))
    if target_length <= 0:
        return np.zeros((0,), dtype=np.float32)
    orig_times = np.linspace(0.0, duration, num=audio_data.shape[0], endpoint=False)
    target_times = np.linspace(0.0, duration, num=target_length, endpoint=False)
    return np.interp(target_times, orig_times, audio_data).astype(np.float32)

def apply_uvr5_vocal_extraction(input_audio_path):
    global _uvr5_separator_instance
    if not HAS_UVR5:
        err_msg = "audio-separator 패키지가 없어 UVR5 분기를 수행할 수 없습니다."
        log_error(MODULE_NAME, err_msg)
        raise RuntimeError(err_msg)

    ensure_directories()
    abs_input_path = Path(input_audio_path).resolve()
    
    target_uvr5_dir = AUTO_UVR5_DIR if AUTO_REC_DIR.resolve() in abs_input_path.parents else MANUAL_UVR5_DIR
    target_uvr5_dir.mkdir(parents=True, exist_ok=True)

    base_name = abs_input_path.stem
    
    expected_vocal_path = target_uvr5_dir / f"{base_name}_Vocals.wav"
    if expected_vocal_path.exists():
        log_info(MODULE_NAME, f"UVR5 보컬 파일 존재: {expected_vocal_path.name}")
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

def split_speaker_turn_by_silence(
    vocal_audio_data,
    start,
    end,
    sample_rate=TARGET_SAMPLE_RATE,
    silence_threshold=0.05,
    min_silence_duration=0.1
):
    start_sample = max(0, int(start * sample_rate))
    end_sample = min(len(vocal_audio_data), int(end * sample_rate))

    if end_sample <= start_sample:
        return []

    chunk = vocal_audio_data[start_sample:end_sample]
    if chunk.size == 0:
        return []

    amplitude = np.abs(chunk)
    silent_mask = amplitude < silence_threshold
    min_silence_samples = max(1, int(min_silence_duration * sample_rate))

    split_points = []
    silence_start = None

    for i, is_silent in enumerate(silent_mask):
        if is_silent:
            if silence_start is None:
                silence_start = i
        else:
            if silence_start is not None:
                silence_length = i - silence_start
                if silence_length >= min_silence_samples:
                    split_points.append((silence_start, i))
                silence_start = None

    if silence_start is not None:
        silence_length = len(silent_mask) - silence_start
        if silence_length >= min_silence_samples:
            split_points.append((silence_start, len(silent_mask)))

    if not split_points:
        return [(start, end)]

    result = []
    current_start_sample = 0

    for silence_start_sample, silence_end_sample in split_points:
        segment_start_sample = current_start_sample
        segment_end_sample = silence_start_sample

        if segment_end_sample > segment_start_sample:
            segment_start = start + segment_start_sample / sample_rate
            segment_end = start + segment_end_sample / sample_rate
            if segment_end > segment_start:
                result.append((segment_start, segment_end))

        current_start_sample = silence_end_sample

    if current_start_sample < len(chunk):
        segment_start = start + current_start_sample / sample_rate
        segment_end = end
        if segment_end > segment_start:
            result.append((segment_start, segment_end))

    return result

def configure_strict_analysis_pipeline(audio_data, sample_rate):
    while True:
        print("\n==============================================")
        print("            분석 모드 선택")
        print("==============================================")
        print(" 0. 기본 분석")
        print(" 1. 단일 화자 알고리즘 분석")
        print(" 2. 다중 화자 알고리즘 분석")
        print(" 3. 메뉴로 돌아가기")
        print("----------------------------------------------")
        mode = input("선택: ").strip()
        
        if mode == "3":
            return None, False, 3
        if mode == "0":
            return [], False, 0

        existing = ah.load_existing_profiles()
            
        if mode == "1":
            if not existing:
                print("[알림] 등록된 프로파일이 없습니다.")
                continue
            matched_speakers = [algo_name for algo_name in existing if ah.verify_single_speaker(algo_name, audio_data)]
            if not matched_speakers:
                print("[차단] 일치하는 화자 프로파일이 없습니다.")
                continue
            return [matched_speakers[0]], True, 1
            
        elif mode == "2":
            if not existing:
                print("[알림] 등록된 프로파일이 없습니다.")
                continue
            success, msg = ah.verify_multi_speakers_auto(audio_data)
            if not success:
                print(f"[차단] {msg}")
                continue
            return existing, False, 2
        else:
            print("[오류] 올바른 번호를 입력해주세요.")

def execute_analysis_flow(model, file_path, active_speakers, is_single, analysis_mode=0):
    log_info(MODULE_NAME, f"분석 실행: {Path(file_path).name}")
    try:
        if model is None:
            log_error(MODULE_NAME, "분석 실행 실패: 모델이 로드되어 있지 않습니다.")
            return

        audio_data, sr = sf.read(str(file_path))
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        audio_data = audio_data.astype(np.float32)
        audio_data = resample_audio(audio_data, sr, TARGET_SAMPLE_RATE)
        
        max_val = np.max(np.abs(audio_data))
        if max_val > 1e-5:
            audio_data = audio_data / max_val

        processed_vocal_path = apply_uvr5_vocal_extraction(file_path)
        
        vocal_audio_data, sr = sf.read(str(processed_vocal_path))
        if vocal_audio_data.ndim > 1:
            vocal_audio_data = np.mean(vocal_audio_data, axis=1)
        vocal_audio_data = vocal_audio_data.astype(np.float32)
        vocal_audio_data = resample_audio(vocal_audio_data, sr, TARGET_SAMPLE_RATE)

        # -----------------------------------------------------------------
        # 전체 음원 1차 ASR 검수 분석 + 동적 게이지바 적용 구간 복구
        # -----------------------------------------------------------------
        log_info(MODULE_NAME, "전체 음원 1차 ASR 검수 분석 수행 중...")
        
        chunk_duration = 30.0
        chunk_samples = int(chunk_duration * TARGET_SAMPLE_RATE)
        total_chunks = max(1, int(np.ceil(vocal_audio_data.shape[0] / chunk_samples)))
        global_texts = []
        global_start_time = time.time()

        for idx in range(total_chunks):
            start_sample = idx * chunk_samples
            end_sample = min(vocal_audio_data.shape[0], (idx + 1) * chunk_samples)
            chunk_audio = vocal_audio_data[start_sample:end_sample]
            if chunk_audio.size == 0:
                continue
                
            current_chunk_num = idx + 1
            percent = int((current_chunk_num / total_chunks) * 100)
            elapsed = time.time() - global_start_time
            avg_time = elapsed / current_chunk_num if current_chunk_num > 0 else 0
            remaining_sec = avg_time * (total_chunks - current_chunk_num)

            eta_str = format_time(remaining_sec)
            elapsed_str = format_time(elapsed)

            terminal_width = shutil.get_terminal_size(fallback=(120, 24)).columns
            prefix = f"{percent:3d}% |"
            suffix = f"| {current_chunk_num}/{total_chunks} {elapsed_str}<{eta_str}"
            fixed_width = len(prefix) + len(suffix)
            bar_len = max(1, terminal_width - fixed_width - 5)
            filled_len = int(bar_len * current_chunk_num / total_chunks)
            empty_len = bar_len - filled_len
            bar_str = "█" * filled_len + " " * empty_len

            progress_msg = prefix + bar_str + suffix
            sys.stdout.write("\r" + progress_msg)
            sys.stdout.flush()

            try:
                from transformers import logging as hf_logging
                hf_logging.set_verbosity_error()
                
                with torch.no_grad():
                    res_chunk = model.transcribe((chunk_audio, TARGET_SAMPLE_RATE))
                
                chunk_text = ""
                if hasattr(res_chunk, "text"):
                    chunk_text = res_chunk.text
                elif isinstance(res_chunk, list):
                    chunk_text = " ".join([str(item.get("text", item)) if isinstance(item, dict) else str(getattr(item, "text", item)) for item in res_chunk])
                else:
                    chunk_text = str(res_chunk)
                    
                if chunk_text.strip():
                    global_texts.append(chunk_text.strip())
            except Exception:
                pass
                
            if torch.cuda.is_available() and CONFIG.get("enable_gpu_cache_clear", True):
                torch.cuda.empty_cache()

        print() # 게이지바 줄바꿈
        global_inspected_text = " ".join(global_texts)
        if global_inspected_text:
            log_info(MODULE_NAME, f"전체 음원 1차 검수 완료 (글자 수: {len(global_inspected_text)})")
        # -----------------------------------------------------------------

        raw_source_stem = Path(file_path).stem
        clean_source_name = re.sub(r'_Vocals$', '', raw_source_stem, flags=re.IGNORECASE)
        
        parent_audio_dir = SEGMENTS_BASE_DIR / clean_source_name
        parent_audio_dir.mkdir(parents=True, exist_ok=True)
            
        prefix_str = "seg_single" if is_single and active_speakers else "segment"
        existing_subdirs = [d for d in parent_audio_dir.iterdir() if d.is_dir() and d.name.startswith(prefix_str)]
        specific_segment_dir = parent_audio_dir / f"{prefix_str}_{len(existing_subdirs) + 1:03d}"
        specific_segment_dir.mkdir(parents=True, exist_ok=True)

        log_info(MODULE_NAME, "화자 분리 수행 중...")
        diar_start_time = time.time()
        total_duration = vocal_audio_data.shape[0] / TARGET_SAMPLE_RATE
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
                    
                    tracks = list(annotation.itertracks(yield_label=True))
                    total_tracks = len(tracks)

                    for idx, (turn, _, speaker) in enumerate(tracks, 1):
                        percent = int((idx / total_tracks) * 100) if total_tracks > 0 else 100
                        elapsed = time.time() - diar_start_time
                        avg_time = elapsed / idx if idx > 0 else 0
                        remaining_sec = avg_time * (total_tracks - idx)

                        eta_str = format_time(remaining_sec)
                        elapsed_str = format_time(elapsed)

                        terminal_width = shutil.get_terminal_size(fallback=(120, 24)).columns
                        prefix = f"{percent:3d}% |"
                        suffix = f"| {idx}/{total_tracks} {elapsed_str}<{eta_str}"
                        fixed_width = len(prefix) + len(suffix)
                        bar_len = max(1, terminal_width - fixed_width - 5)
                        filled_len = int(bar_len * idx / total_tracks) if total_tracks > 0 else bar_len
                        empty_len = bar_len - filled_len
                        bar_str = "█" * filled_len + " " * empty_len

                        progress_msg = prefix + bar_str + suffix
                        sys.stdout.write("\r" + progress_msg)
                        sys.stdout.flush()

                        split_turns = split_speaker_turn_by_silence(
                            vocal_audio_data,
                            turn.start,
                            turn.end,
                            sample_rate=TARGET_SAMPLE_RATE,
                            silence_threshold=0.015,
                            min_silence_duration=0.22
                        )

                        if len(split_turns) > 1:
                            for split_start, split_end in split_turns:
                                raw_speaker_turns.append((split_start, split_end, speaker))
                        else:
                            raw_speaker_turns.append((turn.start, turn.end, speaker))
                        
                    print()
                    log_info(MODULE_NAME, f"화자 분리 완료 (소요 시간: {format_time(time.time() - diar_start_time)})")

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

        log_info(MODULE_NAME, f"구간별 ASR 인식 및 문장 정제 시작 (총 {len(raw_speaker_turns)}개 구간)")
        asr_start_time = time.time()
        
        temp_segments = []
        recent_texts = [] 
        total_turns = len(raw_speaker_turns)
        
        for idx, (start, end, orig_speaker) in enumerate(raw_speaker_turns, 1):
            segment_duration = end - start
            if segment_duration < 0.3:
                continue

            percent = int((idx / total_turns) * 100)
            elapsed = time.time() - asr_start_time
            avg_time = elapsed / idx if idx > 0 else 0
            remaining_sec = avg_time * (total_turns - idx)

            eta_str = format_time(remaining_sec)
            elapsed_str = format_time(elapsed)

            terminal_width = shutil.get_terminal_size(fallback=(120, 24)).columns
            prefix = f"{percent:3d}% |"
            suffix = f"| {idx}/{total_turns} {elapsed_str}<{eta_str}"
            fixed_width = len(prefix) + len(suffix)
            bar_len = max(1, terminal_width - fixed_width - 5)
            filled_len = int(bar_len * idx / total_turns)
            empty_len = bar_len - filled_len
            bar_str = "█" * filled_len + " " * empty_len

            progress_msg = prefix + bar_str + suffix
            sys.stdout.write("\r" + progress_msg)
            sys.stdout.flush()

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

            chunk_txt = post_processor.clean_hallucination_text(chunk_text)
            chunk_txt = post_processor.sanitize_asr_output(
                chunk_txt,
                recent_texts=recent_texts,
                current_start=start
            )
            if not chunk_txt.strip():
                continue

            chunk_txt = post_processor.clean_and_convert_numbers(chunk_txt)
            chunk_txt = post_processor.convert_korean_numbers(chunk_txt)
            chunk_txt = ah.apply_text_corrections(chunk_txt, mapped_speaker)
            
            temp_segments.append({
                "start": start,
                "end": end,
                "speaker": mapped_speaker,
                "text": chunk_txt,
                "audio": chunk_audio
            })

        print()

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

        log_info(MODULE_NAME, f"최종 세그먼트 및 대화 로그 저장 시작 (총 {len(merged_segments)}개 세그먼트)")
        POST_DIR.mkdir(exist_ok=True)
        ASR_DIR.mkdir(exist_ok=True)
        
        post_file_path = POST_DIR / f"{clean_source_name}_post_{len(list(POST_DIR.glob(f'{clean_source_name}_post_*.txt'))) + 1:03d}.txt"
        all_full_texts = []
        saved_segment_count = 0

        post_proc_start_time = time.time()
        total_segments = len(merged_segments)

        with open(post_file_path, "w", encoding="utf-8") as post_file_obj:
            post_file_obj.write(f"=== 대화 로그 ({file_path}) ===\n\n")

            for idx, seg in enumerate(merged_segments, 1):
                percent = int((idx / total_segments) * 100) if total_segments > 0 else 100
                elapsed = time.time() - post_proc_start_time
                avg_time = elapsed / idx if idx > 0 else 0
                remaining_sec = avg_time * (total_segments - idx)

                eta_str = format_time(remaining_sec)
                elapsed_str = format_time(elapsed)

                terminal_width = shutil.get_terminal_size(fallback=(120, 24)).columns
                prefix = f"{percent:3d}% |"
                suffix = f"| {idx}/{total_segments} {elapsed_str}<{eta_str}"
                fixed_width = len(prefix) + len(suffix)
                bar_len = max(1, terminal_width - fixed_width - 5)
                filled_len = int(bar_len * idx / total_segments) if total_segments > 0 else bar_len
                empty_len = bar_len - filled_len
                bar_str = "█" * filled_len + " " * empty_len

                progress_msg = prefix + bar_str + suffix
                sys.stdout.write("\r" + progress_msg)
                sys.stdout.flush()

                saved_segment_count += 1
                safe_speaker_name = str(seg["speaker"]).replace("/", "_").replace("\\", "_")
                
                if analysis_mode == 0:
                    base_seg_name = f"seg_sub_{saved_segment_count-1:03d}_{safe_speaker_name}_{seg['start']:.1f}s-{seg['end']:.1f}s"
                elif analysis_mode == 1:
                    base_seg_name = f"seg_sub_{saved_segment_count-1:03d}_A_{safe_speaker_name}_{seg['start']:.1f}s-{seg['end']:.1f}s"
                else:
                    base_seg_name = f"seg_sub_{saved_segment_count-1:03d}_B_{safe_speaker_name}_{seg['start']:.1f}s-{seg['end']:.1f}s"

                # WAV 파일 및 순수 텍스트 세그먼트 파일 정상 저장
                save_wav_chunk(seg["audio"], TARGET_SAMPLE_RATE, specific_segment_dir / f"{base_seg_name}.wav")
                (specific_segment_dir / f"{base_seg_name}.txt").write_text(seg['text'], encoding="utf-8")

                log_line = f"[{seg['speaker']}] ({format_time(seg['start'])} ~ {format_time(seg['end'])}): {seg['text']}"
                post_file_obj.write(log_line + "\n")
                all_full_texts.append(log_line)

        print()

        asr_raw_file = ASR_DIR / f"{clean_source_name}_asr_{len(list(ASR_DIR.glob(f'{clean_source_name}_asr_*.txt'))) + 1:03d}.txt"
        asr_raw_file.write_text("\n".join(all_full_texts), encoding="utf-8")

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
            root_wavs = sorted([f for f in AUDIO_DIR.glob("*.wav") if "vocal" not in f.name.lower() and "instrumental" not in f.name.lower()], key=lambda x: x.stat().st_mtime, reverse=True)
        except Exception as e:
            log_error(MODULE_NAME, "audio 폴더 내 WAV 파일 검색 실패", e)
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
                    auto_subdirs = sorted([d for d in AUTO_REC_DIR.iterdir() if d.is_dir() and d.name.lower() != "uvr5"], key=lambda x: x.stat().st_mtime, reverse=True) if AUTO_REC_DIR.exists() else []
                except Exception as e:
                    log_error(MODULE_NAME, "자동녹화 하위 폴더 검색 실패", e)
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
                    sub_wavs = sorted(list(target_folder.glob("*.wav")))
                except Exception as e:
                    log_error(MODULE_NAME, f"폴더 내 WAV 검색 실패 ({target_folder.name})", e)
                    sub_wavs = []

                if not sub_wavs:
                    print("[알림] 해당 폴더에 WAV 파일이 없습니다.")
                    input("계속하려면 엔터를 누르세요...")
                    continue
                
                process_target_file(sub_wavs[0])
                return
        else:
            process_target_file(item_data)
            return
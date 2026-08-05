import os
import sys
import queue
import time
import wave
import traceback
import threading
import re
import warnings
import logging
from pathlib import Path
from datetime import datetime

# 💡 내부 라이브러리 로그 출력 차단
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.*")
warnings.filterwarnings("ignore", message=".*torchcodec is not installed correctly.*")
warnings.filterwarnings("ignore", message=".*triton not found.*")
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

import numpy as np
import torch
from qwen_asr import Qwen3ASRModel
import pyaudiowpatch as pyaudio
import algorithm_handler as ah

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

TARGET_SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL_PATH = "Qwen/Qwen3-ASR-1.7B"

AUDIO_DIR = Path("audio")
MANUAL_UVR5_DIR = AUDIO_DIR / "uvr5"
AUTO_REC_DIR = AUDIO_DIR / "auto_recorded_audio"
AUTO_UVR5_DIR = AUTO_REC_DIR / "uvr5"

SEGMENTS_BASE_DIR = Path("segments_base")
ASR_DIR = Path("asr_output")
POST_DIR = Path("post_processing")
ERROR_LOG_DIR = Path("error_log")
TOKEN_FILE = Path("hf_token.txt")

_cached_asr_model = None
_uvr5_separator_instance = None

def ensure_directories():
    for d in [AUDIO_DIR, MANUAL_UVR5_DIR, AUTO_REC_DIR, AUTO_UVR5_DIR, SEGMENTS_BASE_DIR, ASR_DIR, POST_DIR, ERROR_LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    ah.ensure_handler_directories()

def write_error_log(context_name, error_exception):
    ensure_directories()
    error_files = list(ERROR_LOG_DIR.glob("error_log_*.txt"))
    file_path = ERROR_LOG_DIR / f"error_log_{len(error_files) + 1:03d}.txt"
    tb_str = traceback.format_exc()
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("=== voice-script-helper-error-report ===\n")
        f.write(f"일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"작업 위치: {context_name}\n")
        f.write("----------------------------------------\n")
        f.write(f"[에러 메시지]\n{str(error_exception)}\n\n")
        f.write(f"[트레이스백]\n{tb_str}\n")

def get_huggingface_token():
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if token: return token
    return None

def clean_hallucination_text(text):
    if not text:
        return ""
    if re.search(r'[\u4e00-\u9fff\u3040-\u30ff\u31f0-\u31ff]', text):
        return ""
    return text.strip()

def load_asr_model():
    global _cached_asr_model
    if _cached_asr_model is not None:
        print("[*] 이미 메모리에 로드된 ASR 모델을 재사용합니다.")
        return _cached_asr_model

    print(f"\n[*] 고성능 ASR 모델({DEFAULT_MODEL_PATH}) 로딩 중... (장치: {DEVICE})")
    try:
        if DEVICE == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()

        model = Qwen3ASRModel.from_pretrained(
            DEFAULT_MODEL_PATH,
            dtype=torch.float16,
            device_map=DEVICE
        )
        _cached_asr_model = model
        print("[+] ASR 모델 로드 완료!")
        return model
    except Exception as e:
        write_error_log("ASR 모델 로드 단계", e)
        print(f"[오류] 모델 로드 실패: {e}")
        return None

def resample_audio(audio_data, orig_sr, target_sr=16000):
    if orig_sr == target_sr:
        return audio_data.astype(np.float32)
    duration = audio_data.shape[0] / float(orig_sr)
    target_length = int(round(duration * target_sr))
    if target_length <= 0:
        return np.zeros((0,), dtype=np.float32)
    orig_times = np.linspace(0.0, duration, num=audio_data.shape[0], endpoint=False)
    target_times = np.linspace(0.0, duration, num=target_length, endpoint=False)
    return np.interp(target_times, orig_times, audio_data).astype(np.float32)

def save_wav_chunk(audio_data, sample_rate, filename):
    try:
        scaled_audio = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
        with wave.open(str(filename), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(scaled_audio.tobytes())
    except Exception as e:
        print(f"[경고] 세그먼트 WAV 저장 실패: {e}")

def format_time(seconds):
    """초(second) 단위를 입력받아 시:분:초 또는 분:초 형태로 보기 쉽게 변환합니다."""
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

def apply_uvr5_vocal_extraction(input_audio_path):
    global _uvr5_separator_instance
    if not HAS_UVR5:
        raise RuntimeError("audio-separator 패키지가 설치되어 있지 않아 UVR5 분리를 수행할 수 없습니다.")

    ensure_directories()
    abs_input_path = Path(input_audio_path).resolve()
    
    target_uvr5_dir = AUTO_UVR5_DIR if AUTO_REC_DIR.resolve() in abs_input_path.parents else MANUAL_UVR5_DIR
    target_uvr5_dir.mkdir(parents=True, exist_ok=True)

    base_name = abs_input_path.stem
    print(f"\n[⏳ UVR5 전처리: 배경음 및 보컬 정밀 분리 중... ({base_name})]")
    
    if _uvr5_separator_instance is None:
        _uvr5_separator_instance = Separator()
    
    _uvr5_separator_instance.output_dir = str(target_uvr5_dir)
    _uvr5_separator_instance.load_model('UVR-MDX-NET-Voc_FT.onnx')
        
    output_files = _uvr5_separator_instance.separate(str(abs_input_path))
    
    vocal_file = None
    for f in output_files:
        f_lower = f.lower()
        full_path = target_uvr5_dir / f
        if "vocals" in f_lower:
            new_vocal_path = target_uvr5_dir / f"{base_name}_Vocals.wav"
            if full_path.exists():
                if new_vocal_path.exists(): new_vocal_path.unlink()
                full_path.rename(new_vocal_path)
            vocal_file = new_vocal_path
        elif "instrumental" in f_lower or "background" in f_lower or "no_vocals" in f_lower:
            new_inst_path = target_uvr5_dir / f"{base_name}_Instrumental.wav"
            if full_path.exists():
                if new_inst_path.exists(): new_inst_path.unlink()
                full_path.rename(new_inst_path)
            
    if vocal_file and vocal_file.exists():
        print(f"[+ 성공] 보컬 분리 완료: {vocal_file}")
        return str(vocal_file)
    else:
        raise RuntimeError(f"UVR5 분리 완료 후 'Vocals' 파일을 찾을 수 없습니다: {base_name}")

def process_audio_pipeline(model, audio_data, sample_rate, source_identifier="audio", active_speakers=None, is_single_speaker_target=False):
    try:
        ensure_directories()
        max_val = np.max(np.abs(audio_data))
        if max_val > 1e-5:
            audio_data = audio_data / max_val

        raw_source_stem = Path(source_identifier).stem
        clean_source_name = re.sub(r'_Vocals$', '', raw_source_stem, flags=re.IGNORECASE)
        
        parent_audio_dir = SEGMENTS_BASE_DIR / clean_source_name
        parent_audio_dir.mkdir(parents=True, exist_ok=True)
            
        prefix_str = "seg_single" if is_single_speaker_target and active_speakers else "segment"
        existing_subdirs = [d for d in parent_audio_dir.iterdir() if d.is_dir() and d.name.startswith(prefix_str)]
        specific_segment_dir = parent_audio_dir / f"{prefix_str}_{len(existing_subdirs) + 1:03d}"
        specific_segment_dir.mkdir(parents=True, exist_ok=True)

        print("\n[⏳ 1단계: 화자 분리(Diarization) 수행 중...]")
        total_duration = audio_data.shape[0] / sample_rate
        
        raw_speaker_turns = []
        if HAS_PYANNOTE and not is_single_speaker_target:
            token = get_huggingface_token()
            if token:
                try:
                    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
                    pipeline.to(torch.device(DEVICE))
                    tensor_audio = torch.from_numpy(audio_data).unsqueeze(0).to(torch.float32)
                    diarization = pipeline({"waveform": tensor_audio, "sample_rate": sample_rate})
                    
                    annotation = getattr(diarization, "speaker_diarization", diarization)
                    for turn, _, speaker in annotation.itertracks(yield_label=True):
                        if (turn.end - turn.start) >= 1.0:
                            raw_speaker_turns.append((turn.start, turn.end, speaker))
                    print(f"[+] 유효 발화 구간 {len(raw_speaker_turns)}개 추출됨")
                except Exception as ex:
                    print(f"[경고] 화자 분리 중 오류: {ex}")
        
        if not raw_speaker_turns:
            raw_speaker_turns.append((0.0, total_duration, "speaker_1"))

        speaker_mapping = {}
        if active_speakers and len(active_speakers) > 1:
            unique_orig_speakers = sorted({s for _, _, s in raw_speaker_turns})
            for idx, orig_s in enumerate(unique_orig_speakers):
                speaker_mapping[orig_s] = active_speakers[idx] if idx < len(active_speakers) else f"speaker_{idx+1}"
        else:
            s_name = active_speakers[0] if active_speakers else "speaker_1"
            for _, _, orig_speaker in raw_speaker_turns:
                speaker_mapping[orig_speaker] = s_name

        print("\n[⏳ 2단계: 각 구간별 개별 ASR 인식 및 세그먼트 저장 시작]")
        all_full_texts = []
        POST_DIR.mkdir(exist_ok=True)
        
        existing_posts = list(POST_DIR.glob(f"{clean_source_name}_post_*.txt"))
        post_file_path = POST_DIR / f"{clean_source_name}_post_{len(existing_posts) + 1:03d}.txt"
        
        with open(post_file_path, "w", encoding="utf-8") as post_file_obj:
            post_file_obj.write(f"=== 정제된 화자별 대화 로그 (출처: {source_identifier}) ===\n\n")

            for idx, (start, end, orig_speaker) in enumerate(raw_speaker_turns):
                mapped_speaker = speaker_mapping.get(orig_speaker, "speaker_1")
                
                start_sample = int(start * sample_rate)
                end_sample = int(end * sample_rate)
                chunk_audio = audio_data[start_sample:end_sample]
                
                chunk_text = ""
                if chunk_audio.size > sample_rate * 0.2:
                    try:
                        res_chunk = model.transcribe((chunk_audio, sample_rate))
                        if hasattr(res_chunk, "text"):
                            chunk_text = res_chunk.text
                        elif isinstance(res_chunk, list):
                            chunk_text = " ".join([str(item.get("text", item)) if isinstance(item, dict) else str(getattr(item, "text", item)) for item in res_chunk])
                        else:
                            chunk_text = str(res_chunk)
                    except Exception as e:
                        print(f"[경고] {idx}번 구간 ASR 처리 실패: {e}")

                chunk_txt = clean_hallucination_text(chunk_text)
                chunk_txt = ah.apply_text_corrections(chunk_txt, mapped_speaker)
                if not chunk_txt:
                    chunk_txt = "(음성 인식 내용 없음)"

                base_seg_name = f"seg_sub_{idx:03d}_{mapped_speaker}_{start:.1f}s-{end:.1f}s"
                seg_wav_filename = specific_segment_dir / f"{base_seg_name}.wav"
                seg_txt_filename = specific_segment_dir / f"{base_seg_name}.txt"
                
                save_wav_chunk(chunk_audio, sample_rate, seg_wav_filename)
                seg_txt_filename.write_text(chunk_txt, encoding="utf-8")

                # 보기 편하게 포맷팅 함수 적용
                start_str = format_time(start)
                end_str = format_time(end)

                log_line = f"[{mapped_speaker}] ({start_str} ~ {end_str}): {chunk_txt}"
                post_file_obj.write(log_line + "\n")
                all_full_texts.append(log_line)
                print(f"  - 구간 처리 완료 [{idx+1}/{len(raw_speaker_turns)}] {mapped_speaker} ({start_str} ~ {end_str})")

        ASR_DIR.mkdir(exist_ok=True)
        existing_asrs = list(ASR_DIR.glob(f"{clean_source_name}_asr_*.txt"))
        asr_raw_file = ASR_DIR / f"{clean_source_name}_asr_{len(existing_asrs) + 1:03d}.txt"
        asr_raw_file.write_text("\n".join(all_full_texts), encoding="utf-8")

        print(f"\n[💾 후처리 로그 저장 완료] {post_file_path}")
        print(f"[💾 ASR 전체 결과 저장 완료] {asr_raw_file}")
        print(f"[🎉 분석 완료! 결과 폴더: {specific_segment_dir}/]")
        return str(specific_segment_dir)

    except Exception as e:
        write_error_log("파이프라인 실행 중", e)
        print(f"[오류] {e}")
        return None

def configure_strict_analysis_pipeline(audio_data, sample_rate):
    while True:
        print("\n==============================================")
        print("              음원 분석 모드 선택")
        print("============================================== ")
        print(" 0. 기본 분석")
        print(" 1. 단일 화자 알고리즘 분석")
        print(" 2. 다중 화자 알고리즘 분석")
        print(" 3. 메뉴로 돌아가기")
        print("----------------------------------------------")
        mode = input("선택: ").strip()
        
        if mode == "3":
            print("[*] 메인 메뉴로 돌아갑니다.")
            return None, False
        if mode == "0":
            print("[*] 기본 분석 모드로 진행합니다.")
            return [], False

        existing = ah.load_existing_profiles()
        print(f"[*] 시스템에 등록된 프로파일 목록: {existing if existing else '없음'}")
            
        if mode == "1":
            if not existing:
                print("\n[알림] 등록된 알고리즘 프로파일이 없습니다.\n")
                continue
            matched_speakers = [algo_name for algo_name in existing if ah.verify_single_speaker(algo_name, audio_data)]
            if not matched_speakers:
                print("[🚫 분석 차단] 일치하는 화자 프로파일이 없습니다.")
                continue
            print(f"[+] 일치하는 화자 프로파일 감지됨: {matched_speakers[0]}")
            return [matched_speakers[0]], True
            
        elif mode == "2":
            if not existing:
                print("\n[알림] 등록된 알고리즘 프로파일이 없습니다.\n")
                continue
            success, msg = ah.verify_multi_speakers_auto(audio_data)
            if not success:
                print(f"[🚫 분석 차단] {msg}")
                continue
            print(f"[+] 다중 화자 검증 통과. 등록된 전체 프로파일({existing})을 적용합니다.")
            return existing, False
        else:
            print("[오류] 올바른 번호를 입력해주세요.")

def execute_analysis_flow(model, target_file):
    try:
        with wave.open(target_file, 'rb') as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0 if sample_width == 2 else np.frombuffer(frames, dtype=np.float32)
            if n_channels > 1:
                audio_data = np.mean(audio_data.reshape(-1, n_channels), axis=1)
        audio_data = resample_audio(audio_data, sr, TARGET_SAMPLE_RATE)
    except Exception as e:
        write_error_log("오디오 파일 로드 단계", e)
        print(f"\n[오류] 오디오 파일을 읽어오는 중 문제가 발생했습니다: {e}")
        return

    active_speakers, is_single = configure_strict_analysis_pipeline(audio_data, TARGET_SAMPLE_RATE)
    if active_speakers is None:
        return

    try:
        processed_vocal_path = apply_uvr5_vocal_extraction(target_file)
    except Exception as e:
        write_error_log("UVR5 보컬 분리 단계", e)
        print(f"\n[🚫 분석 차단] UVR5 분리 과정 오류로 작업을 중단합니다.\n[상세 오류]: {e}")
        return

    with wave.open(processed_vocal_path, 'rb') as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
        vocal_audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0 if sample_width == 2 else np.frombuffer(frames, dtype=np.float32)
        if n_channels > 1:
            vocal_audio_data = np.mean(vocal_audio_data.reshape(-1, n_channels), axis=1)

    vocal_audio_data = resample_audio(vocal_audio_data, sr, TARGET_SAMPLE_RATE)
    process_audio_pipeline(model, vocal_audio_data, TARGET_SAMPLE_RATE, source_identifier=processed_vocal_path, active_speakers=active_speakers, is_single_speaker_target=is_single)

def record_and_transcribe(model):
    print("\n==============================================")
    print("                녹화 방식 선택")
    print("============================================== ")
    print(" 1. 실시간 녹화 ")
    print(" 2. 시간 선택 자동 녹화 ")
    print("----------------------------------------------")
    rec_mode = input("선택: ").strip()
    
    target_total_seconds = 0.0
    min_input = 0.0
    if rec_mode == "2":
        try:
            min_input = float(input("총 녹화 시간을 분(Minute) 단위로 입력하세요 (예: 15, 60): ").strip())
            target_total_seconds = min_input * 60.0
        except ValueError:
            print("[오류] 올바른 숫자를 입력해주세요. 직접 녹화 종료 모드로 전환합니다.")
            rec_mode = "1"

    p = pyaudio.PyAudio()
    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = p.get_device_info_by_index(wasapi_info['defaultOutputDevice'])
        
        loopback_device = None
        if not default_speakers.get("isLoopbackDevice", False):
            for loopback in p.get_loopback_device_info_generator():
                if default_speakers['name'] in loopback['name']:
                    loopback_device = loopback
                    break
            if loopback_device is None:
                loopback_device = next(p.get_loopback_device_info_generator())
        else:
            loopback_device = default_speakers
            
        native_sr = int(loopback_device['defaultSampleRate'])
        channels = loopback_device['maxInputChannels']
        print(f"[*] 브라우저 오디오 루프백 캡처 장치 확정: [{loopback_device['name']}] (샘플레이트: {native_sr}Hz)")
    except Exception as e:
        print(f"[오류] 브라우저 루프백 장치 설정 실패: {e}")
        p.terminate()
        return

    print(f"\n[*] 준비되었습니다.\n💡 브라우저 소리만 녹화됩니다.")
    if rec_mode == "1":
        input("👉 소리가 나는 상태에서 엔터(Enter) 키를 누르면 녹화가 시작됩니다...")
        print(f"\n[🔴 브라우저 녹화 중...] (끝내려면 Enter)")
    else:
        input(f"👉 엔터를 누르면 [{min_input}분] 동안 자동 녹화가 시작됩니다.")
        print(f"\n[🔴 브라우저 자동 녹화 중...] (목표 시간: {min_input}분)")

    stop_event = threading.Event()
    audio_queue = queue.Queue()
    buffer = []
    is_recording_flag = True
    start_time = time.time()
    
    SPLIT_INTERVAL_SECONDS = 300.0
    saved_file_paths = []
    
    current_auto_session_dir = AUTO_REC_DIR
    if rec_mode == "2":
        AUTO_REC_DIR.mkdir(parents=True, exist_ok=True)
        existing_subdirs = [d for d in AUTO_REC_DIR.iterdir() if d.is_dir() and d.name.startswith("auto_recorded_")]
        current_auto_session_dir = AUTO_REC_DIR / f"auto_recorded_{len(existing_subdirs) + 1:03d}"
        current_auto_session_dir.mkdir(parents=True, exist_ok=True)

    def audio_callback(in_data, frame_count, time_info, status):
        if is_recording_flag:
            audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    stream = p.open(
        format=pyaudio.paFloat32,
        channels=channels,
        rate=native_sr,
        input=True,
        input_device_index=loopback_device['index'],
        frames_per_buffer=CHUNK_SIZE,
        stream_callback=audio_callback
    )
    stream.start_stream()
    threading.Thread(target=lambda: (input(), stop_event.set()), daemon=True).start()

    def save_current_buffer(current_buf):
        if not current_buf:
            return
        raw_audio = np.concatenate(current_buf, axis=0)
        if channels > 1:
            raw_audio = np.mean(raw_audio.reshape(-1, channels), axis=1)
        
        target_dir = current_auto_session_dir if rec_mode == "2" else AUDIO_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        
        clean_wavs = [f for f in target_dir.glob("audio_*.wav") if "vocal" not in f.name.lower() and "instrumental" not in f.name.lower()]
        temp_recorded_path = target_dir / f"audio_{len(clean_wavs) + 1:03d}.wav"
        
        scaled_audio = np.clip(raw_audio * 32767, -32768, 32767).astype(np.int16)
        with wave.open(str(temp_recorded_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(native_sr)
            wf.writeframes(scaled_audio.tobytes())
        print(f"\n[💾 오디오 파일 저장 완료] {temp_recorded_path}")
        saved_file_paths.append(str(temp_recorded_path))

    try:
        last_split_time = start_time
        while not stop_event.is_set():
            current_time = time.time()
            elapsed = current_time - start_time
            
            if rec_mode == "2" and elapsed >= target_total_seconds:
                print(f"\n[+] 목표 녹화 시간({min_input}분)에 도달하여 녹화를 자동으로 종료합니다.")
                break

            if rec_mode == "2" and (current_time - last_split_time >= SPLIT_INTERVAL_SECONDS):
                if buffer:
                    save_current_buffer(buffer)
                    buffer = []
                last_split_time = current_time

            current_chunk_rms = 0.0
            while not audio_queue.empty():
                chunk = audio_queue.get()
                audio_np = np.frombuffer(chunk, dtype=np.float32)
                buffer.append(audio_np)
                if audio_np.size > 0:
                    current_chunk_rms = np.sqrt(np.mean(audio_np**2))

            bar_length = 20
            filled_length = int(min(current_chunk_rms * 50, 1.0) * bar_length)
            gauge = "█" * filled_length + "-" * (bar_length - filled_length)

            elapsed_str = format_time(elapsed)
            sys.stdout.write(f"\r[🔴 녹화 중] 시간: {elapsed_str} | 신호 [{gauge}] (종료: Enter)")
            sys.stdout.flush()
            time.sleep(0.1)
    except Exception as e:
        print(f"\n[경고] 녹화 루프 중 예외 발생: {e}")
    finally:
        is_recording_flag = False
        stream.stop_stream()
        stream.close()
        p.terminate()
        print()

    if buffer:
        save_current_buffer(buffer)

    if not saved_file_paths:
        print("\n[알림] 녹음된 오디오가 없습니다.")
        return

    print(f"\n[+ 성공] 모든 녹화 파일 저장 완료! (총 {len(saved_file_paths)}개 파일)")
    for target_file in saved_file_paths:
        print(f"\n----------------------------------------------")
        print(f"[*] 대상 파일 분석 시작: {target_file}")
        print(f"----------------------------------------------")
        execute_analysis_flow(model, target_file)

def select_and_process_audio_file(model):
    ensure_directories()
    while True:
        print("\n==============================================")
        print("          오디오 파일 선택 및 분석")
        print("============================================== ")
        print(" 0. 자동녹화 폴더 선택")
        
        normal_files = []
        if AUDIO_DIR.exists():
            for f in AUDIO_DIR.glob("*.wav"):
                if "vocal" not in f.name.lower() and "instrumental" not in f.name.lower():
                    normal_files.append(f)
        
        for idx, filepath in enumerate(normal_files, 1):
            print(f" {idx}. {filepath.name}")
            
        back_option_num = len(normal_files) + 1
        print(f" {back_option_num}. 메인 메뉴로 돌아가기")
        print("----------------------------------------------")
        
        choice = input("선택: ").strip()
        if not choice.isdigit():
            print("[오류] 올바른 번호를 선택해주세요.")
            continue
            
        choice_val = int(choice)
        if choice_val == back_option_num:
            print("[*] 메인 메뉴로 돌아갑니다.")
            return
            
        if choice_val == 0:
            if not AUTO_REC_DIR.exists():
                print("\n[알림] 생성된 자동녹화 폴더가 없습니다.")
                continue
                
            sub_sessions = sorted([d for d in AUTO_REC_DIR.iterdir() if d.is_dir() and d.name.startswith("auto_recorded_")])
            if not sub_sessions:
                print("\n[알림] 자동녹화된 세션 폴더가 없습니다.")
                continue
                
            print("\n==============================================")
            print("          자동녹화 세션 폴더 목록")
            print("============================================== ")
            for s_idx, s_dir in enumerate(sub_sessions, 1):
                print(f" {s_idx}. {s_dir.name}")
            print(f" {len(sub_sessions) + 1}. 이전 메뉴로")
            print("----------------------------------------------")
            
            s_choice = input("선택: ").strip()
            if not s_choice.isdigit():
                continue
                
            s_idx_val = int(s_choice) - 1
            if s_idx_val == len(sub_sessions):
                continue
            if not (0 <= s_idx_val < len(sub_sessions)):
                print("[오류] 범위를 벗어난 선택입니다.")
                continue
                
            chosen_session_dir = sub_sessions[s_idx_val]
            session_wavs = sorted([
                f for f in chosen_session_dir.glob("*.wav") 
                if "vocal" not in f.name.lower() and "instrumental" not in f.name.lower()
            ])
            
            if not session_wavs:
                print(f"\n[알림] 선택한 폴더 내에 분석할 WAV 파일이 없습니다.")
                continue
                
            print(f"\n[+] 총 {len(session_wavs)}개의 분할된 오디오 파일을 순차적으로 분석합니다.")
            for target_file in session_wavs:
                print(f"\n----------------------------------------------")
                print(f"[*] 대상 파일 분석 중: {target_file.name}")
                print(f"----------------------------------------------")
                execute_analysis_flow(model, str(target_file))
            break
        else:
            idx_val = choice_val - 1
            if not (0 <= idx_val < len(normal_files)):
                print("[오류] 올바른 번호를 입력해주세요.")
                continue
            target_file = normal_files[idx_val]
            execute_analysis_flow(model, str(target_file))
            break

if __name__ == "__main__":
    ensure_directories()
    asr_model = load_asr_model()
    if not asr_model:
        print("[오류] ASR 모델을 불러오지 못해 프로그램을 종료합니다.")
        sys.exit(1)

    while True:
        print("\n==============================================")
        print("         Voice Script Helper (메인)")
        print("============================================== ")
        print(" 1. 브라우저 오디오 녹화 및 자동 분석")
        print(" 2. 기존 오디오 파일 선택 및 분석")
        print(" 3. 종료")
        print("----------------------------------------------")
        sel = input("선택: ").strip()
        if sel == "1":
            record_and_transcribe(asr_model)
        elif sel == "2":
            select_and_process_audio_file(asr_model)
        elif sel == "3":
            print("[*] 프로그램을 종료합니다.")
            break
        else:
            print("[오류] 올바른 번호를 선택해주세요.")
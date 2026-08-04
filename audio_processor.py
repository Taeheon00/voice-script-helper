# audio_processor.py
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

# 불필요한 경고 메시지 화면 출력 원천 차단
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.*")
warnings.filterwarnings("ignore", message=".*torchcodec is not installed correctly.*")
warnings.filterwarnings("ignore", message=".*triton not found.*")
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
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

AUDIO_DIR = "audio"
UVR5_OUTPUT_DIR = os.path.join(AUDIO_DIR, "uvr5")
ASR_DIR = "asr_output"
POST_DIR = "post_processing"
SEGMENTS_BASE_DIR = "segments_base"
ERROR_LOG_DIR = "error_log"
TOKEN_FILE = "hf_token.txt"

audio_queue = queue.Queue()
is_recording = False

def ensure_directories():
    for d in [AUDIO_DIR, UVR5_OUTPUT_DIR, ASR_DIR, POST_DIR, SEGMENTS_BASE_DIR, ERROR_LOG_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)
    ah.ensure_handler_directories()

def get_next_filename(directory, prefix, extension="txt"):
    ensure_directories()
    existing_files = os.listdir(directory)
    count = sum(1 for f in existing_files if f.startswith(prefix) and f.endswith(f".{extension}"))
    return os.path.join(directory, f"{prefix}_{count + 1:03d}.{extension}")

def get_next_audio_filename(directory, extension="wav"):
    ensure_directories()
    prefix = "audio"
    existing_files = os.listdir(directory)
    count = sum(1 for f in existing_files if f.startswith(prefix) and f.endswith(f".{extension}") and not "vocal" in f and not "Instrumental" in f)
    return os.path.join(directory, f"{prefix}_{count + 1:03d}.{extension}")

def write_error_log(context_name, error_exception):
    ensure_directories()
    file_path = get_next_filename(ERROR_LOG_DIR, ERROR_LOG_DIR, "txt")
    tb_str = traceback.format_exc()
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("=== 커스텀 챗봇 오류 발생 보고 ===\n")
        f.write(f"일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"작업 위치: {context_name}\n")
        f.write("----------------------------------------\n")
        f.write(f"[에러 메시지]\n{str(error_exception)}\n\n")
        f.write(f"[트레이스백]\n{tb_str}\n")

def set_huggingface_token_menu():
    print("\n==============================================")
    print("       허깅페이스(HuggingFace) 토큰 설정")
    print("==============================================")
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r", encoding="utf-8") as f:
            current_token = f.read().strip()
            if current_token:
                print(f"[*] 현재 등록된 토큰: {current_token[:6]}...")
    token = input("HuggingFace Read 토큰 입력 (hf_...): ").strip()
    if token:
        with open(TOKEN_FILE, "w", encoding="utf-8") as f:
            f.write(token)
        print("[+] 토큰이 성공적으로 저장되었습니다!")

def get_huggingface_token():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r", encoding="utf-8") as f:
            token = f.read().strip()
            if token: return token
    return None

def clean_hallucination_text(text):
    if not text:
        return ""
    if re.search(r'[\u4e00-\u9fff\u3040-\u30ff\u31f0-\u31ff]', text):
        return ""
    return text.strip()

def load_asr_model():
    print(f"\n[*] 고성능 ASR 모델({DEFAULT_MODEL_PATH}) 로딩 중... (장치: {DEVICE})")
    try:
        model = Qwen3ASRModel.from_pretrained(
            DEFAULT_MODEL_PATH,
            dtype=torch.float16,
            device_map=DEVICE
        )
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
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(scaled_audio.tobytes())
    except Exception as e:
        print(f"[경고] 세그먼트 WAV 저장 실패: {e}")

def apply_uvr5_vocal_extraction(input_audio_path):
    if not HAS_UVR5:
        raise RuntimeError("audio-separator 패키지가 설치되어 있지 않아 UVR5 분리를 수행할 수 없습니다.")

    ensure_directories()
    base_name = os.path.splitext(os.path.basename(input_audio_path))[0]
    print(f"\n[⏳ UVR5 전처리: 고성능 모델(Voc_FT)로 배경음 및 보컬 정밀 분리 중...]")
    
    separator = Separator(output_dir=UVR5_OUTPUT_DIR)
    separator.load_model('UVR-MDX-NET-Voc_FT.onnx')
    output_files = separator.separate(input_audio_path)
    
    vocal_file = None
    instrumental_file = None
    
    for f in output_files:
        f_lower = f.lower()
        full_path = os.path.join(UVR5_OUTPUT_DIR, f)
        if "vocals" in f_lower:
            new_vocal_name = f"{base_name}_Vocals.wav"
            new_vocal_path = os.path.join(UVR5_OUTPUT_DIR, new_vocal_name)
            if os.path.exists(full_path):
                if os.path.exists(new_vocal_path): os.remove(new_vocal_path)
                os.rename(full_path, new_vocal_path)
            vocal_file = new_vocal_path
        elif "instrumental" in f_lower or "background" in f_lower or "no_vocals" in f_lower:
            new_inst_name = f"{base_name}_Instrumental.wav"
            new_inst_path = os.path.join(UVR5_OUTPUT_DIR, new_inst_name)
            if os.path.exists(full_path):
                if os.path.exists(new_inst_path): os.remove(new_inst_path)
                os.rename(full_path, new_inst_path)
            instrumental_file = new_inst_path
            
    if vocal_file and os.path.exists(vocal_file):
        print(f"[+ 성공] 보컬 및 배경음 정밀 분리 완료:")
        print(f"    - 보컬 파일(Vocals): {vocal_file}")
        if instrumental_file:
            print(f"    - 배경음 파일(Instrumental): {instrumental_file}")
        return vocal_file
    else:
        raise RuntimeError("UVR5 분리 프로세스는 완료되었으나, 결과물에서 'Vocals' 파일을 찾을 수 없습니다.")

def process_audio_pipeline(model, audio_data, sample_rate, source_identifier="audio", active_speakers=None, is_single_speaker_target=False):
    try:
        max_val = np.max(np.abs(audio_data))
        if max_val > 0.0001:
            audio_data = audio_data / max_val

        clean_source_name = os.path.splitext(os.path.basename(source_identifier))[0]
        parent_audio_dir = os.path.join(SEGMENTS_BASE_DIR, clean_source_name)
        if not os.path.exists(parent_audio_dir):
            os.makedirs(parent_audio_dir)
            
        prefix_str = "seg_single" if is_single_speaker_target and active_speakers else "세그먼트"
        existing_subdirs = [d for d in os.listdir(parent_audio_dir) if os.path.isdir(os.path.join(parent_audio_dir, d)) and d.startswith(prefix_str)]
        next_seg_num = len(existing_subdirs) + 1
        specific_segment_dir = os.path.join(parent_audio_dir, f"{prefix_str}_{next_seg_num}")
        os.makedirs(specific_segment_dir, exist_ok=True)

        print(f"\n[⏳ 1단계: 전체 보컬 오디오 통합 분석 중 (긴 음원 구간별 처리)]")
        total_duration = audio_data.shape[0] / sample_rate
        chunk_duration = 30.0 
        full_extracted = []
        
        if total_duration > chunk_duration:
            num_chunks = int(np.ceil(total_duration / chunk_duration))
            with tqdm(total=num_chunks, desc="[🎯 전체 보컬 분석]", unit="구간") as pbar:
                for i in range(num_chunks):
                    start_sec = i * chunk_duration
                    end_sec = min((i + 1) * chunk_duration, total_duration)
                    
                    s_idx = int(start_sec * sample_rate)
                    e_idx = int(end_sec * sample_rate)
                    sub_audio = audio_data[s_idx:e_idx]
                    
                    if len(sub_audio) > 0:
                        try:
                            res_sub = model.transcribe((sub_audio, sample_rate))
                        except Exception:
                            res_sub = ""

                        if isinstance(res_sub, list):
                            for item in res_sub:
                                full_extracted.append(str(getattr(item, "text", item.get("text", item) if isinstance(item, dict) else item)))
                        else:
                            full_extracted.append(str(getattr(res_sub, "text", res_sub.get("text", res_sub) if isinstance(res_sub, dict) else res_sub)))
                    pbar.update(1)
        else:
            try:
                res_full = model.transcribe((audio_data, sample_rate))
            except Exception:
                res_full = ""

            if isinstance(res_full, list):
                for item in res_full:
                    full_extracted.append(str(getattr(item, "text", item.get("text", item) if isinstance(item, dict) else item)))
            else:
                full_extracted.append(str(getattr(res_full, "text", res_full.get("text", res_full) if isinstance(res_full, dict) else res_full)))
        
        raw_full_text = "\n".join([clean_hallucination_text(line) for line in full_extracted if clean_hallucination_text(line)]).strip()
        
        asr_raw_file = get_next_filename(ASR_DIR, ASR_DIR, "txt")
        with open(asr_raw_file, "w", encoding="utf-8") as f:
            f.write(raw_full_text)
        print(f"[💾 원본 ASR 저장 완료] {asr_raw_file}")

        print(f"\n[⏳ 2단계: 화자 분리 및 1초 미만 노이즈 구간 필터링]")
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
                        duration = turn.end - turn.start
                        if duration >= 1.0:
                            raw_speaker_turns.append((turn.start, turn.end, speaker))
                    print(f"[+] 유효 발화 구간 {len(raw_speaker_turns)}개 추출됨")
                except Exception as ex:
                    print(f"[경고] 화자 분리 중 오류: {ex}")
        else:
            raw_speaker_turns.append((0.0, total_duration, "단일화자"))

        speaker_mapping = {}
        if active_speakers and len(active_speakers) > 1:
            unique_orig_speakers = sorted(list(set([s for _, _, s in raw_speaker_turns])))
            for idx, orig_s in enumerate(unique_orig_speakers):
                if idx < len(active_speakers):
                    speaker_mapping[orig_s] = active_speakers[idx]
                else:
                    speaker_mapping[orig_s] = f"화자{idx+1}"
        else:
            s_name = active_speakers[0] if active_speakers else "화자1"
            for _, _, orig_speaker in raw_speaker_turns:
                speaker_mapping[orig_speaker] = s_name

        post_file = get_next_filename(POST_DIR, POST_DIR, "txt")
        with open(post_file, "w", encoding="utf-8") as f:
            f.write(f"=== 정제된 화자별 대화 로그 (출처: {source_identifier}, 회차: {prefix_str}{next_seg_num}) ===\n\n")
            if raw_speaker_turns:
                print(f"\n[⏳ 3단계: 조각별 개별 ASR 분석 및 세그먼트 저장]")
                with tqdm(total=len(raw_speaker_turns), desc="[🎯 음성 분석 추출]", unit="구간") as pbar:
                    for idx, (start, end, orig_speaker) in enumerate(raw_speaker_turns):
                        mapped_speaker = speaker_mapping.get(orig_speaker, "화자1")
                        
                        start_sample = int(start * sample_rate)
                        end_sample = int(end * sample_rate)
                        chunk_audio = audio_data[start_sample:end_sample]
                        
                        try:
                            res_chunk = model.transcribe((chunk_audio, sample_rate))
                            chunk_texts = []
                            if isinstance(res_chunk, list):
                                for item in res_chunk:
                                    chunk_texts.append(str(getattr(item, "text", item.get("text", item) if isinstance(item, dict) else item)))
                            else:
                                chunk_texts.append(str(getattr(res_chunk, "text", res_chunk.get("text", res_chunk) if isinstance(res_chunk, dict) else res_chunk)))
                            
                            chunk_txt = " ".join([clean_hallucination_text(t) for t in chunk_texts if clean_hallucination_text(t)])
                        except Exception as asr_err:
                            chunk_txt = f"(ASR 오류: {asr_err})"

                        if chunk_txt and chunk_txt != "(음성 인식 내용 없음)":
                            base_seg_name = f"seg_{idx:03d}_{mapped_speaker}_{start:.1f}s-{end:.1f}s"
                            seg_wav_filename = os.path.join(specific_segment_dir, f"{base_seg_name}.wav")
                            seg_txt_filename = os.path.join(specific_segment_dir, f"{base_seg_name}.txt")
                            
                            save_wav_chunk(chunk_audio, sample_rate, seg_wav_filename)
                            
                            try:
                                with open(seg_txt_filename, "w", encoding="utf-8") as st_f:
                                    st_f.write(chunk_txt)
                            except Exception:
                                pass

                            log_line = f"[{mapped_speaker}] ({start:.1f}초 ~ {end:.1f}초): {chunk_txt}"
                            f.write(log_line + "\n")
                        pbar.update(1)
            else:
                f.write(raw_full_text)
                
        print(f"\n[🎉 분석 완료! 결과 파일: {post_file}]")
        print(f"[📂 데이터셋 세그먼트 저장 폴더: {specific_segment_dir}/]")
    except Exception as e:
        write_error_log("파이프라인 실행 중", e)
        print(f"[오류] {e}")

def configure_strict_analysis_pipeline(audio_data, sample_rate):
    print("\n==============================================")
    print("               음원 분석 모드 선택")
    print("==============================================")
    print(" 0. 기본 분석")
    print(" 1. 단일 화자 알고리즘 분석")
    print(" 2. 다중 화자 알고리즘 분석")
    print("----------------------------------------------")
    mode = input("선택: ").strip()
    
    existing = ah.load_existing_profiles()
    print(f"[*] 시스템에 등록된 프로파일 목록: {existing if existing else '없음'}")
    
    if mode == "0":
        print("[*] 기본 분석 모드로 진행합니다.")
        return [], False
        
    elif mode == "1":
        s_name = input("적용할 단일 화자 이름 입력: ").strip()
        if not s_name:
            print("[오류] 화자 이름이 입력되지 않았습니다.")
            return None, True
            
        if not ah.verify_speaker_with_audio(s_name, audio_data, sample_rate):
            print("[🚫 분석 차단] 음역대 검증에 실패하여 분석 프로세스가 강제 중단됩니다.")
            return None, True
            
        return [s_name], True
        
    elif mode == "2":
        if not existing:
            print("\n[알림] 등록된 알고리즘 프로파일이 없습니다.")
            return None, False

        try:
            count_str = input("참여할 다중 화자 인원 수 입력: ").strip()
            if not count_str.isdigit():
                print("[오류] 숫자로 입력해주세요.")
                return None, False
            count = int(count_str)
        except ValueError:
            return None, False
            
        speakers = []
        for i in range(count):
            s_name = input(f"{i+1}번째 참여 화자 이름 입력: ").strip()
            speakers.append(s_name)
            
        if not ah.verify_multi_speakers_strict(speakers, audio_data, sample_rate):
            print("[🚫 분석 차단] 다중 화자 중 음역대 불일치 항목이 있어 분석을 차단합니다.")
            return None, False
            
        return speakers, False
        
    return [], False

def record_and_transcribe(model):
    global is_recording
    p = pyaudio.PyAudio()
    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = p.get_device_info_by_index(wasapi_info['defaultOutputDevice'])
        
        # 크로미움 기반 브라우저(크롬, 웨일 등) 간 충돌/전체 시스템 유입을 방지하고
        # 오직 독립적인 브라우저 소리(루프백)만 정확히 타겟팅하기 위한 디바이스 분기 처리
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

    print(f"\n[*] 준비되었습니다.")
    print("💡 브라우저 소리만 녹화됩니다.")
    print("💡 유튜브, 동영상, 스트리밍 등 오디오를 재생해 주세요!")
    input("👉 소리가 나는 상태에서 엔터(Enter) 키를 누르세요...")
    
    buffer = []
    is_recording = True
    start_time = time.time()
    
    def audio_callback(in_data, frame_count, time_info, status):
        if is_recording:
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

    print(f"\n[🔴 브라우저 소리 단독 녹화 중...] (끝내려면 Enter)")
    stop_event = threading.Event()
    threading.Thread(target=lambda: (input(), stop_event.set()), daemon=True).start()
    
    try:
        while not stop_event.is_set():
            elapsed = time.time() - start_time
            audio_level = 0.0
            while not audio_queue.empty():
                chunk = audio_queue.get()
                audio_np = np.frombuffer(chunk, dtype=np.float32)
                buffer.append(audio_np)
                if audio_np.size > 0:
                    audio_level = np.max(np.abs(audio_np))
            
            bar_len = 30
            filled_len = int(min(audio_level * 150, bar_len))
            bar_str = "█" * filled_len + "-" * (bar_len - filled_len)
            sys.stdout.write(f"\r[🔴 녹화 중] 시간: {elapsed:4.1f}초 | 볼륨: [{bar_str}] ({audio_level:.4f})")
            sys.stdout.flush()
            time.sleep(0.05)
    except Exception as e:
        print(f"\n[경고] 녹화 루프 중 예외 발생: {e}")
    finally:
        is_recording = False
        stream.stop_stream()
        stream.close()
        p.terminate()
        print()

    if not buffer:
        print("\n[알림] 녹음된 오디오가 없습니다.")
        return

    raw_audio = np.concatenate(buffer, axis=0)
    if channels > 1:
        raw_audio = np.mean(raw_audio.reshape(-1, channels), axis=1)
    
    ensure_directories()
    temp_recorded_path = get_next_audio_filename(AUDIO_DIR, "wav")
    scaled_audio = np.clip(raw_audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(temp_recorded_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(native_sr)
        wf.writeframes(scaled_audio.tobytes())
    print(f"[💾 녹음 원본 저장 완료] {temp_recorded_path}")

    try:
        processed_vocal_path = apply_uvr5_vocal_extraction(temp_recorded_path)
    except Exception as e:
        write_error_log("UVR5 보컬 분리 단계 (녹음)", e)
        print(f"\n[🚫 분석 차단] UVR5 분리 과정 오류로 작업을 중단합니다.\n[상세 오류]: {e}")
        return

    with wave.open(processed_vocal_path, 'rb') as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
        audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0 if sample_width == 2 else np.frombuffer(frames, dtype=np.float32)
        if n_channels > 1:
            audio_data = np.mean(audio_data.reshape(-1, n_channels), axis=1)

    audio_data = resample_audio(audio_data, sr, TARGET_SAMPLE_RATE)
    active_speakers, is_single = configure_strict_analysis_pipeline(audio_data, TARGET_SAMPLE_RATE)
    if active_speakers is None:
        return

    process_audio_pipeline(model, audio_data, TARGET_SAMPLE_RATE, source_identifier=processed_vocal_path, active_speakers=active_speakers, is_single_speaker_target=is_single)

def select_and_process_audio_file(model):
    ensure_directories()
    files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith('.wav') and not "vocal" in f.lower() and not "instrumental" in f.lower()]
    if not files:
        print(f" [!] '{AUDIO_DIR}' 폴더에 분석 가능한 원본 WAV 파일이 없습니다.")
        return

    for idx, filename in enumerate(files, 1): print(f" {idx}. {filename}")
    choice = input("분석할 파일 번호 입력: ").strip()
    if not choice.isdigit() or int(choice)-1 >= len(files): return
    
    selected_filename = files[int(choice)-1]
    target_file = os.path.join(AUDIO_DIR, selected_filename)
    
    try:
        processed_vocal_path = apply_uvr5_vocal_extraction(target_file)
    except Exception as e:
        write_error_log("UVR5 보컬 분리 단계 (파일 선택)", e)
        print(f"\n[🚫 분석 차단] UVR5 분리 과정 오류로 작업을 중단합니다.\n[상세 오류]: {e}")
        return
    
    with wave.open(processed_vocal_path, 'rb') as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
        audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0 if sample_width == 2 else np.frombuffer(frames, dtype=np.float32)
        if n_channels > 1:
            audio_data = np.mean(audio_data.reshape(-1, n_channels), axis=1)

    audio_data = resample_audio(audio_data, sr, TARGET_SAMPLE_RATE)
    active_speakers, is_single = configure_strict_analysis_pipeline(audio_data, TARGET_SAMPLE_RATE)
    if active_speakers is None:
        return

    process_audio_pipeline(model, audio_data, TARGET_SAMPLE_RATE, source_identifier=processed_vocal_path, active_speakers=active_speakers, is_single_speaker_target=is_single)
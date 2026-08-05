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

# 📁 폴더 구조 정의 (사용자 요구사항 반영)
AUDIO_DIR = "audio"
MANUAL_UVR5_DIR = os.path.join(AUDIO_DIR, "uvr5")

AUTO_REC_DIR = os.path.join(AUDIO_DIR, "auto_recorded_audio")
AUTO_UVR5_DIR = os.path.join(AUTO_REC_DIR, "uvr5")

SEGMENTS_BASE_DIR = "segments_base"
ASR_DIR = "asr_output"
POST_DIR = "post_processing"
ERROR_LOG_DIR = "error_log"
TOKEN_FILE = "hf_token.txt"

audio_queue = queue.Queue()
is_recording = False

# 🚀 최적화 캐시 변수 (중복 로딩 방지용 싱글톤)
_cached_asr_model = None
_uvr5_separator_instance = None

def ensure_directories():
    for d in [AUDIO_DIR, MANUAL_UVR5_DIR, AUTO_REC_DIR, AUTO_UVR5_DIR, SEGMENTS_BASE_DIR, ASR_DIR, POST_DIR, ERROR_LOG_DIR]:
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
    count = sum(1 for f in existing_files if f.startswith(prefix) and f.endswith(f".{extension}") and not "vocal" in f.lower() and not "instrumental" in f.lower())
    return os.path.join(directory, f"{prefix}_{count + 1:03d}.{extension}")

def write_error_log(context_name, error_exception):
    ensure_directories()
    file_path = get_next_filename(ERROR_LOG_DIR, ERROR_LOG_DIR, "txt")
    tb_str = traceback.format_exc()
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("=== voice-script-helper-error-report ===\n")
        f.write(f"일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"작업 위치: {context_name}\n")
        f.write("----------------------------------------\n")
        f.write(f"[에러 메시지]\n{str(error_exception)}\n\n")
        f.write(f"[트레이스백]\n{tb_str}\n")

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
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(scaled_audio.tobytes())
    except Exception as e:
        print(f"[경고] 세그먼트 WAV 저장 실패: {e}")

def apply_uvr5_vocal_extraction(input_audio_path):
    global _uvr5_separator_instance
    if not HAS_UVR5:
        raise RuntimeError("audio-separator 패키지가 설치되어 있지 않아 UVR5 분리를 수행할 수 없습니다.")

    ensure_directories()
    
    # 🔍 사용자 요구사항에 따른 경로 분기 처리
    # auto_recorded_audio 폴더 안의 파일이면 -> audio/auto_recorded_audio/uvr5/ 에 저장
    # 그 외(일반 실시간 녹화 또는 메뉴 2 파일 분석 등 audio/ 최상위 파일)면 -> audio/uvr5/ 에 저장
    abs_input_path = os.path.abspath(input_audio_path)
    abs_auto_dir = os.path.abspath(AUTO_REC_DIR)
    
    if abs_input_path.startswith(abs_auto_dir):
        target_uvr5_dir = AUTO_UVR5_DIR
    else:
        target_uvr5_dir = MANUAL_UVR5_DIR

    if not os.path.exists(target_uvr5_dir):
        os.makedirs(target_uvr5_dir)

    base_name = os.path.splitext(os.path.basename(input_audio_path))[0]
    print(f"\n[⏳ UVR5 전처리: 고성능 모델(Voc_FT)로 배경음 및 보컬 정밀 분리 중... ({base_name})]")
    
    if _uvr5_separator_instance is None:
        _uvr5_separator_instance = Separator()
    
    _uvr5_separator_instance.output_dir = target_uvr5_dir
    _uvr5_separator_instance.load_model('UVR-MDX-NET-Voc_FT.onnx')
        
    output_files = _uvr5_separator_instance.separate(input_audio_path)
    
    vocal_file = None
    for f in output_files:
        f_lower = f.lower()
        full_path = os.path.join(target_uvr5_dir, f)
        if "vocals" in f_lower:
            new_vocal_name = f"{base_name}_Vocals.wav"
            new_vocal_path = os.path.join(target_uvr5_dir, new_vocal_name)
            if os.path.exists(full_path):
                if os.path.exists(new_vocal_path): os.remove(new_vocal_path)
                os.rename(full_path, new_vocal_path)
            vocal_file = new_vocal_path
        elif "instrumental" in f_lower or "background" in f_lower or "no_vocals" in f_lower:
            new_inst_name = f"{base_name}_Instrumental.wav"
            new_inst_path = os.path.join(target_uvr5_dir, new_inst_name)
            if os.path.exists(full_path):
                if os.path.exists(new_inst_path): os.remove(new_inst_path)
                os.rename(full_path, new_inst_path)
            
    if vocal_file and os.path.exists(vocal_file):
        print(f"[+ 성공] 보컬 분리 완료: {vocal_file}")
        return vocal_file
    else:
        raise RuntimeError(f"UVR5 분리 완료 후 'Vocals' 파일을 찾을 수 없습니다: {base_name}")

def process_audio_pipeline(model, audio_data, sample_rate, source_identifier="audio", active_speakers=None, is_single_speaker_target=False):
    try:
        max_val = np.max(np.abs(audio_data))
        if max_val > 0.0001:
            audio_data = audio_data / max_val

        clean_source_name = os.path.splitext(os.path.basename(source_identifier))[0]
        parent_audio_dir = os.path.join(SEGMENTS_BASE_DIR, clean_source_name)
        if not os.path.exists(parent_audio_dir):
            os.makedirs(parent_audio_dir)
            
        prefix_str = "seg_single" if is_single_speaker_target and active_speakers else "segment"
        existing_subdirs = [d for d in os.listdir(parent_audio_dir) if os.path.isdir(os.path.join(parent_audio_dir, d)) and d.startswith(prefix_str)]
        next_seg_num = len(existing_subdirs) + 1
        specific_segment_dir = os.path.join(parent_audio_dir, f"{prefix_str}_{next_seg_num}")
        os.makedirs(specific_segment_dir, exist_ok=True)

        print(f"\n[⏳ 1단계: 전체 보컬 오디오 통합 분석 중 (구간별 처리)]")
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

        print(f"\n[⏳ 2단계: 화자 분리 및 노이즈 구간 필터링]")
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
            raw_speaker_turns.append((0.0, total_duration, "single_speaker"))

        speaker_mapping = {}
        if active_speakers and len(active_speakers) > 1:
            unique_orig_speakers = sorted(list(set([s for _, _, s in raw_speaker_turns])))
            for idx, orig_s in enumerate(unique_orig_speakers):
                if idx < len(active_speakers):
                    speaker_mapping[orig_s] = active_speakers[idx]
                else:
                    speaker_mapping[orig_s] = f"speaker_{idx+1}"
        else:
            s_name = active_speakers[0] if active_speakers else "speaker_1"
            for _, _, orig_speaker in raw_speaker_turns:
                speaker_mapping[orig_speaker] = s_name

        post_file_path = get_next_filename(POST_DIR, POST_DIR, "txt")
        with open(post_file_path, "w", encoding="utf-8") as post_file_obj:
            post_file_obj.write(f"=== 정제된 화자별 대화 로그 (출처: {source_identifier}) ===\n\n")

            if raw_speaker_turns:
                print(f"\n[⏳ 3단계: 조각별 개별 ASR 분석 및 세그먼트 저장]")
                with tqdm(total=len(raw_speaker_turns), desc="[🎯 음성 분석 추출]", unit="구간") as pbar:
                    for idx, (start, end, orig_speaker) in enumerate(raw_speaker_turns):
                        mapped_speaker = speaker_mapping.get(orig_speaker, "speaker_1")
                        
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
                            
                            raw_chunk_txt = " ".join([clean_hallucination_text(t) for t in chunk_texts if clean_hallucination_text(t)])
                        except Exception as asr_err:
                            raw_chunk_txt = f"(ASR 오류: {asr_err})"

                        chunk_txt = ah.apply_text_corrections(raw_chunk_txt, mapped_speaker)

                        if chunk_txt and chunk_txt != "(음성 인식 내용 없음)":
                            base_seg_name = f"seg_sub_{idx:03d}_{mapped_speaker}_{start:.1f}s-{end:.1f}s"
                            seg_wav_filename = os.path.join(specific_segment_dir, f"{base_seg_name}.wav")
                            seg_txt_filename = os.path.join(specific_segment_dir, f"{base_seg_name}.txt")
                            seg_raw_txt_filename = os.path.join(specific_segment_dir, f"{base_seg_name}.raw_txt")
                            
                            save_wav_chunk(chunk_audio, sample_rate, seg_wav_filename)
                            
                            try:
                                with open(seg_txt_filename, "w", encoding="utf-8") as st_f:
                                    st_f.write(chunk_txt)
                                with open(seg_raw_txt_filename, "w", encoding="utf-8") as srt_f:
                                    srt_f.write(raw_chunk_txt)
                            except Exception:
                                pass

                            log_line = f"[{mapped_speaker}] ({start:.1f}초 ~ {end:.1f}초): {chunk_txt}"
                            post_file_obj.write(log_line + "\n")
                        pbar.update(1)
        
        print(f"\n[🎉 분석 완료! 단일 통합 결과 폴더: {specific_segment_dir}/]")
        return specific_segment_dir

    except Exception as e:
        write_error_log("파이프라인 실행 중", e)
        print(f"[오류] {e}")
        return None

def configure_strict_analysis_pipeline(audio_data, sample_rate):
    while True:
        print("\n==============================================")
        print("               음원 분석 모드 선택")
        print("==============================================")
        print(" 0. 기본 분석")
        print(" 1. 단일 화자 알고리즘 분석")
        print(" 2. 다중 화자 알고리즘 분석")
        print(" 3. 메뉴로 돌아가기")
        print("----------------------------------------------")
        mode = input("선택: ").strip()
        
        if mode == "3":
            print("[*] 메인 메뉴로 돌아갑니다.")
            return None, False

        existing = ah.load_existing_profiles()
        print(f"[*] 시스템에 등록된 프로파일 목록: {existing if existing else '없음'}")
        
        if mode == "0":
            print("[*] 기본 분석 모드로 진행합니다.")
            return [], False
            
        elif mode == "1":
            if not existing:
                print("\n[알림] 등록된 알고리즘 프로파일이 없습니다. 다른 분석 모드를 선택해주세요.\n")
                continue
                
            matched_speakers = []
            for algo_name in existing:
                if ah.verify_single_speaker(algo_name, audio_data):
                    matched_speakers.append(algo_name)
            
            if not matched_speakers:
                print("[🚫 분석 차단] 등록된 단일 화자 알고리즘 중 유연한 피치 매칭을 만족하는 프로파일이 없습니다.")
                continue
                
            print(f"[+] 일치하는 화자 프로파일 감지됨: {matched_speakers[0]}")
            return [matched_speakers[0]], True
            
        elif mode == "2":
            if not existing:
                print("\n[알림] 등록된 알고리즘 프로파일이 없습니다. 다른 분석 모드를 선택해주세요.\n")
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
    # 1. 원본 오디오 파일 로드
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

    # 2. UVR5 분리 전에 [음원 분석 모드 선택] 메뉴를 먼저 출력하고 사용자 선택 받기
    active_speakers, is_single = configure_strict_analysis_pipeline(audio_data, TARGET_SAMPLE_RATE)
    if active_speakers is None:
        return

    # 3. 분석 모드 선택 후, UVR5 보컬 분리 수행 (정해진 폴더 규칙 적용)
    try:
        processed_vocal_path = apply_uvr5_vocal_extraction(target_file)
    except Exception as e:
        write_error_log("UVR5 보컬 분리 단계", e)
        print(f"\n[🚫 분석 차단] UVR5 분리 과정 오류로 작업을 중단합니다.\n[상세 오류]: {e}")
        return

    # 4. 분리된 보컬 파일 로드 후 ASR 파이프라인 진행
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
    global is_recording
    
    print("\n==============================================")
    print("                녹화 방식 선택")
    print("==============================================")
    print(" 1. 실시간 녹화 ")
    print(" 2. 시간 선택 자동 녹화 ")
    print("----------------------------------------------")
    rec_mode = input("선택: ").strip()
    
    target_total_seconds = 0
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

    print(f"\n[*] 준비되었습니다.")
    print("💡 브라우저 소리만 녹화됩니다.")
    print("💡 유튜브, 동영상, 스트리밍 등 오디오를 재생해 주세요!")
    
    if rec_mode == "1":
        input("👉 소리가 나는 상태에서 엔터(Enter) 키를 누르면 녹화가 시작됩니다...")
        print(f"\n[🔴 브라우저 녹화 중...] (끝내려면 Enter)")
    else:
        input(f"👉 엔터를 누르면 [{min_input}분] 동안 자동 녹화가 시작됩니다. (중간 종료하려면 Enter)")
        print(f"\n[🔴 브라우저 자동 녹화 중...] (목표 시간: {min_input}분 / 중간 종료하려면 Enter)")

    stop_event = threading.Event()
    buffer = []
    is_recording = True
    start_time = time.time()
    
    SPLIT_INTERVAL_SECONDS = 300.0
    saved_file_paths = []
    
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

    threading.Thread(target=lambda: (input(), stop_event.set()), daemon=True).start()

    def save_current_buffer(current_buf):
        if not current_buf:
            return
        raw_audio = np.concatenate(current_buf, axis=0)
        if channels > 1:
            raw_audio = np.mean(raw_audio.reshape(-1, channels), axis=1)
        
        ensure_directories()
        
        # 📂 저장 경로 분기 규칙 적용
        # rec_mode가 "2"(자동 녹화)이면 audio/auto_recorded_audio/ 에 저장
        # rec_mode가 "1"(실시간 녹화)이면 audio/ 에 저장
        target_dir = AUTO_REC_DIR if rec_mode == "2" else AUDIO_DIR
        
        temp_recorded_path = get_next_audio_filename(target_dir, "wav")
        scaled_audio = np.clip(raw_audio * 32767, -32768, 32767).astype(np.int16)
        with wave.open(temp_recorded_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(native_sr)
            wf.writeframes(scaled_audio.tobytes())
        print(f"\n[💾 오디오 파일 저장 완료] {temp_recorded_path}")
        saved_file_paths.append(temp_recorded_path)

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

    if buffer:
        save_current_buffer(buffer)

    if not saved_file_paths:
        print("\n[알림] 녹음된 오디오가 없습니다.")
        return

    target_save_loc = AUTO_REC_DIR if rec_mode == "2" else AUDIO_DIR
    print(f"\n[+ 성공] 모든 녹화 파일 저장 완료! (총 {len(saved_file_paths)}개 파일)")
    print(f"[*] 저장 위치: {target_save_loc}/")
    
    # 메뉴 1번 흐름: 녹화가 끝나면 곧바로 분석 흐름(분석 메뉴 선택 ➔ UVR5 ➔ ASR)으로 연계
    print(f"\n[*] 녹화된 파일에 대한 분석 모드를 시작합니다.")
    for target_file in saved_file_paths:
        print(f"\n----------------------------------------------")
        print(f"[*] 대상 파일 분석 시작: {target_file}")
        print(f"----------------------------------------------")
        execute_analysis_flow(model, target_file)

def select_and_process_audio_file(model):
    ensure_directories()
    files = []
    for root, dirs, filenames in os.walk(AUDIO_DIR):
        # auto_recorded_audio 폴더 내의 파일은 메뉴 2번 파일 분석 리스트에서 제외 (메뉴 1 자동녹화 전용)
        if AUTO_REC_DIR in root:
            continue
        for f in filenames:
            if f.lower().endswith('.wav') and not "vocal" in f.lower() and not "instrumental" in f.lower():
                files.append(os.path.join(root, f))
                
    if not files:
        print(f" [!] '{AUDIO_DIR}' 폴더 내에 분석 가능한 원본 WAV 파일이 없습니다.")
        return

    for idx, filepath in enumerate(files, 1):
        print(f" {idx}. {filepath}")
    choice = input("분석할 파일 번호 입력 (메뉴로 돌아가려면 엔터 또는 기타 키): ").strip()
    if not choice.isdigit() or int(choice)-1 >= len(files): return
    
    target_file = files[int(choice)-1]
    execute_analysis_flow(model, target_file)
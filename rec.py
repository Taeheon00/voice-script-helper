import sys
import time
import threading
import soundfile as sf
import numpy as np
import os

# 공통 에러 로거 안전 연동 (독립 실행 대비 fallback 포함)
try:
    from error_logger import log_error, log_info
except ImportError:
    def log_error(mod, msg, exc=None, debug=False):
        print(f"[오류][{mod}] {msg} {exc if exc else ''}")
    def log_info(mod, msg):
        print(f"[*] [{mod}] {msg}")

MODULE_NAME = "Recorder"

# ProcessAudioCapture 패키지 연동 확인
try:
    from process_audio_capture import ProcessAudioCapture
    HAS_PROCESS_CAPTURE = True
except ImportError:
    HAS_PROCESS_CAPTURE = False

# 외부 모듈(audio_processor) 참조
try:
    import audio_processor as ap
except ImportError:
    ap = None

def ensure_directories():
    """main.py와의 호환성을 위해 추가된 디렉토리 생성 함수"""
    dirs = ["audio", "segments_base", "asr_output", "post_processing", "saved_algorithms", "error_log"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    if ap:
        if hasattr(ap, "ensure_directories"):
            ap.ensure_directories()

def _auto_detect_target_pid():
    if not HAS_PROCESS_CAPTURE:
        log_error(MODULE_NAME, "process-audio-capture 패키지가 설치되어 있지 않습니다.")
        return None
    try:
        audio_procs = ProcessAudioCapture.enumerate_audio_processes()
    except Exception as e:
        log_error(MODULE_NAME, "활성 오디오 프로세스 스캔 실패", e, debug=True)
        return None
    if not audio_procs:
        return None
    target_names = ["chrome.exe", "msedge.exe", "whale.exe", "firefox.exe"]
    for p in audio_procs:
        p_name_lower = p.name.lower()
        if any(b in p_name_lower for b in target_names):
            print(f"[+] 자동 감지된 타겟 브라우저: {p.name} (PID: {p.pid})")
            return p.pid
    first_proc = audio_procs[0]
    print(f"[+] 대체 오디오 프로세스 자동 선택: {first_proc.name} (PID: {first_proc.pid})")
    return first_proc.pid

def record_and_transcribe(model=None):
    # 디렉토리 보장 로직 호출
    ensure_directories()
    
    while True:
        print("\n==============================================")
        print("                녹화 방식 선택")
        print("============================================== ")
        print(" 1. 실시간 녹화 ")
        print(" 2. 시간 선택 자동 녹화 (5분 단위 분할 저장) ")
        print(" 3. 메뉴로 돌아가기 ")
        print("----------------------------------------------")
        
        sys.stdin.flush()
        rec_mode = input("선택: ").strip()
        
        if rec_mode == '3':
            return
        elif rec_mode in ['1', '2']:
            break
        else:
            print("잘못된 입력입니다. 1, 2, 3 중에서 선택해주세요.")

    target_total_seconds = 0.0
    min_input = 0.0
    if rec_mode == "2":
        try:
            sys.stdin.flush()
            min_input = float(input("총 녹화 시간을 분(Minute) 단위로 입력하세요 (예: 15, 60): ").strip())
            target_total_seconds = min_input * 60.0
        except ValueError as e:
            log_error(MODULE_NAME, "녹화 시간 입력 값 파싱 오류 (숫자 아님)", e)
            print("[오류] 올바른 숫자를 입력해주세요. 실시간 녹화 모드로 전환합니다.")
            rec_mode = "1"

    if not HAS_PROCESS_CAPTURE or not ProcessAudioCapture.is_supported():
        log_error(MODULE_NAME, "Process Loopback을 지원하지 않거나 패키지가 없습니다.")
        return

    print("\n[오디오 프로세스 자동 스캔 중...]")
    target_pid = _auto_detect_target_pid()
    if not target_pid:
        print("[오류] 현재 소리를 출력 중인 브라우저/오디오 프로세스가 없습니다. 소리 재생 후 다시 시도하세요.")
        return

    current_auto_session_dir = None
    try:
        if rec_mode == "2":
            ap.AUTO_REC_DIR.mkdir(parents=True, exist_ok=True)
            existing_subdirs = [d for d in ap.AUTO_REC_DIR.iterdir() if d.is_dir() and d.name.startswith("auto_recorded_")]
            current_auto_session_dir = ap.AUTO_REC_DIR / f"auto_recorded_{len(existing_subdirs) + 1:03d}"
            current_auto_session_dir.mkdir(parents=True, exist_ok=True)
            rec_file = current_auto_session_dir / "audio_001.wav"
        else:
            ap.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            clean_wavs = [f for f in ap.AUDIO_DIR.glob("audio_*.wav") if "vocal" not in f.name.lower() and "instrumental" not in f.name.lower()]
            rec_file = ap.AUDIO_DIR / f"audio_{len(clean_wavs) + 1:03d}.wav"
    except Exception as e:
        log_error(MODULE_NAME, "녹화 저장 경로 생성 중 예외 발생", e, debug=True)
        return

    print(f"[*] 타겟 PID [{target_pid}] 지정 완료.")
    
    sys.stdin.flush()
    if rec_mode == "1":
        input("👉 소리가 나는 상태에서 엔터(Enter) 키를 누르면 녹화가 시작됩니다...")
        print(f"\n[🔴 브라우저 녹화 중...] (끝내려면 Enter)")
    else:
        input(f"👉 엔터를 누르면 [{min_input}분] 동안 자동 녹화가 시작됩니다. (중간에 Enter를 누르면 언제든 조기 종료 가능)")
        print(f"\n[🔴 브라우저 자동 녹화 중...] (목표 시간: {min_input}분)")

    stop_event = threading.Event()
    threading.Thread(target=lambda: (sys.stdin.read(1) if sys.stdin.readable() else input(), stop_event.set()), daemon=True).start()

    start_time = time.time()
    current_db = [-60.0]
    segment_duration = 300.0  # 5분 단위 분할
    file_counter = 1

    def on_level_callback(db_value):
        current_db[0] = db_value

    def record_worker():
        nonlocal rec_file, file_counter
        try:
            segment_start_time = time.time()
            capture = ProcessAudioCapture(pid=target_pid, output_path=str(rec_file), level_callback=on_level_callback)
            capture.start()

            while not stop_event.is_set():
                now = time.time()
                elapsed_total = now - start_time
                segment_elapsed = now - segment_start_time

                if rec_mode == "2" and elapsed_total >= target_total_seconds:
                    print(f"\n[+] 목표 녹화 시간({min_input}분)에 도달하여 녹화를 정상 종료합니다.")
                    stop_event.set()
                    break

                if rec_mode == "2" and segment_elapsed >= segment_duration:
                    capture.stop()
                    print(f"\n[+] 5분 경과: 세그먼트 파일 저장 완료 -> {rec_file.resolve()}")
                    
                    file_counter += 1
                    rec_file = current_auto_session_dir / f"audio_{file_counter:03d}.wav"
                    
                    capture = ProcessAudioCapture(pid=target_pid, output_path=str(rec_file), level_callback=on_level_callback)
                    capture.start()
                    segment_start_time = time.time()
                    print(f"[🔴 다음 세그먼트 녹화 중...] (새 파일: {rec_file.name})")

                db = current_db[0]
                normalized_val = min(max((db + 60.0) / 60.0, 0.0), 1.0) if db > -59.0 else 0.0
                active_blocks = int(normalized_val * 20)
                gauge = "█" * active_blocks + "-" * (20 - active_blocks)

                time_str = ap.format_time(elapsed_total) if ap and hasattr(ap, 'format_time') else f"{int(elapsed_total)}초"
                sys.stdout.write(f"\r[🔴 녹화 중] 총시간: {time_str} | 신호 [{gauge}] (종료: Enter)")
                sys.stdout.flush()
                time.sleep(0.1)

            capture.stop()
        except Exception as e:
            log_error(MODULE_NAME, "녹화 워커 스레드 실행 중 예외 발생", e, debug=True)

    record_worker()

    print(f"\n" + "="*46)
    if rec_mode == "2" and current_auto_session_dir:
        print(f"💾 자동 녹화 및 분할 파일 저장 완료!")
        print(f"📁 저장 폴더: {current_auto_session_dir.resolve()}")
    else:
        print(f"💾 실시간 녹화 파일 저장 완료!")
        print(f"📁 저장 경로: {rec_file.resolve()}")
    print("="*46)

    # 🔗 녹화 직후 분석 모드 진입
    target_files_to_analyze = []
    try:
        if rec_mode == "2" and current_auto_session_dir and current_auto_session_dir.exists():
            target_files_to_analyze = sorted([f for f in current_auto_session_dir.glob("*.wav") if "vocal" not in f.name.lower() and "instrumental" not in f.name.lower()])
        elif rec_file.exists():
            target_files_to_analyze = [rec_file]
    except Exception as e:
        log_error(MODULE_NAME, "녹화 직후 분석 대상 파일 검색 실패", e)

    if target_files_to_analyze and ap:
        print(f"\n[✨ 분석 모드 진입] 총 {len(target_files_to_analyze)}개의 오디오 파일 분석을 준비합니다.")
        
        current_model = model or ap.load_asr_model()
        if current_model is None:
            log_error(MODULE_NAME, "ASR 모델이 로드되지 않아 녹화 직후 분석을 진행할 수 없습니다.")
            return

        sample_file = target_files_to_analyze[0]
        try:
            sample_audio_data, sr = sf.read(str(sample_file))
            if sample_audio_data.ndim > 1:
                sample_audio_data = np.mean(sample_audio_data, axis=1)
            sample_audio_data = sample_audio_data.astype(np.float32)
            
            target_sr = sr
            if hasattr(ap, 'resample_audio') and hasattr(ap, 'TARGET_SAMPLE_RATE'):
                sample_audio_data = ap.resample_audio(sample_audio_data, sr, ap.TARGET_SAMPLE_RATE)
                target_sr = ap.TARGET_SAMPLE_RATE

            active_speakers, is_single = None, False
            if hasattr(ap, 'configure_strict_analysis_pipeline'):
                active_speakers, is_single = ap.configure_strict_analysis_pipeline(sample_audio_data, target_sr)

            if active_speakers is not None:
                for target_file in target_files_to_analyze:
                    print(f"\n----------------------------------------------")
                    print(f"[*] 대상 파일 분석 중: {target_file.name}")
                    print(f"----------------------------------------------")
                    if hasattr(ap, 'execute_analysis_flow'):
                        ap.execute_analysis_flow(current_model, str(target_file), active_speakers, is_single)
            else:
                print("[*] 분석 파이프라인 구성 실패로 분석을 건너뜁니다.")

        except Exception as e:
            log_error(MODULE_NAME, "녹화 후 분석 연동 중 예외 발생", e, debug=True)

def run_rec_menu(model=None):
    try:
        record_and_transcribe(model)
    except Exception as e:
        log_error(MODULE_NAME, "녹화 메뉴 실행 중 치명적 예외 발생", e, debug=True)
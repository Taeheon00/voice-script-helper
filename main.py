import sys
import os
import json
import ui_studio as us
import rec
import audio_processor as ap

# 공통 에러 로거 연동 (단일화)
from error_logger import log_error, log_info

TOKEN_FILE = "hf_token.txt"
CONFIG_FILE = "config.json"
MODULE_NAME = "Main"

# 16GB 환경 및 장시간(50분 이상) 오디오 분석 최적화 기본 설정
DEFAULT_CONFIG = {
    "enable_gpu_cache_clear": True,        # 구간별 처리 후 GPU 캐시 강제 비우기
    "max_chunk_duration_sec": 30.0,        # 메모리 폭발 방지를 위한 ASR 세그먼트 최대 길이 제한
    "chunk_batch_size": 4,                 # 동시 처리 배치 사이즈 제한
    "uvr5_segment_size": 1048576,          # UVR5 보컬 분리 시 메모리 분할 처리 단위 (오버플로우 방지)
    "torch_threads": 4                     # CPU 오버헤드 방지를 위한 코어 스레드 제한
}

def init_default_config():
    """애플리케이션 구동 시 기본 config.json이 없으면 최적화 값으로 생성합니다."""
    if not os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_CONFIG, f, indent=4, ensure_ascii=False)
            print("[+] 16GB 시스템 최적화 기본 설정 파일(config.json)이 생성되었습니다.")
        except Exception as e:
            log_error(MODULE_NAME, "기본 config.json 생성 실패", e)
    else:
        # 기존 파일이 있더라도 누락된 키가 있다면 보완
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            updated = False
            for k, v in DEFAULT_CONFIG.items():
                if k not in config_data:
                    config_data[k] = v
                    updated = True
            if updated:
                with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                    json.dump(config_data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            log_error(MODULE_NAME, "config.json 검증 및 업데이트 실패", e)

def set_huggingface_token_menu():
    print("\n==============================================")
    print("              허깅페이스 토큰 설정")
    print("==============================================")
    
    current_token = ""
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "r", encoding="utf-8") as f:
                current_token = f.read().strip()
        except Exception as e:
            log_error(MODULE_NAME, "토큰 파일 읽기 실패", e)
            
    if current_token:
        try:
            masked = current_token[:4] + "*" * (len(current_token) - 4) if len(current_token) > 4 else "****"
            print(f"[*] 현재 설정된 토큰: {masked}")
        except Exception as e:
            log_error(MODULE_NAME, "토큰 마스킹 처리 중 예외 발생", e)
            print("[*] 현재 설정된 토큰: (표시 불가)")
    else:
        print("[*] 현재 설정된 토큰: 없음 (비어있음)")
        
    print("----------------------------------------------")
    try:
        new_token = input("새로운 허깅페이스 토큰을 입력하세요 (취소하려면 엔터): ").strip()
    except Exception as e:
        log_error(MODULE_NAME, "토큰 입력 수신 중 예외 발생", e)
        new_token = ""

    if new_token:
        try:
            with open(TOKEN_FILE, "w", encoding="utf-8") as f:
                f.write(new_token)
            print("[+ 성공] 허깅페이스 토큰이 성공적으로 저장되었습니다.")
        except Exception as e:
            log_error(MODULE_NAME, "토큰 저장 실패", e)
    else:
        print("[알림] 토큰 설정이 변경되지 않았습니다.")
        log_info(MODULE_NAME, "토큰 설정이 변경되지 않음 (사용자 취소 또는 빈 값)")
    print("==============================================")

def handle_audio_selection(model):
    """audio 폴더 파일 선택 및 분석 프로세스 처리 함수"""
    try:
        if hasattr(ap, "select_and_process_audio_file"):
            ap.select_and_process_audio_file(model)
            return
        elif hasattr(ap, "run_audio_file_selection"):
            ap.run_audio_file_selection(model)
            return
    except Exception as e:
        log_error(MODULE_NAME, "audio_processor 내부 파일 선택 함수 실행 중 예외 발생", e)

    print("\n[안내] 'audio' 폴더 파일 선택 분석 기능을 실행합니다.")
    try:
        audio_dir = getattr(ap, "AUDIO_DIR", None)
        if audio_dir and audio_dir.exists():
            try:
                wav_files = sorted([
                    f for f in audio_dir.glob("*.wav") 
                    if "vocal" not in f.name.lower() and "instrumental" not in f.name.lower()
                ])
            except Exception as e:
                log_error(MODULE_NAME, "audio 폴더 내 WAV 파일 검색 실패", e)
                wav_files = []

            if wav_files:
                print("\n[audio 폴더 WAV 파일 목록]")
                print("----------------------------------------------")
                for idx, wf in enumerate(wav_files, 1):
                    print(f" {idx:2d}. {wf.name}")
                print("----------------------------------------------")
                
                try:
                    sub_choice = input("분석할 파일 번호를 선택하세요 (취소는 엔터): ").strip()
                except Exception as e:
                    log_error(MODULE_NAME, "WAV 파일 선택 번호 입력 수신 중 예외 발생", e)
                    sub_choice = ""

                if sub_choice.isdigit():
                    try:
                        idx_val = int(sub_choice) - 1
                        if 0 <= idx_val < len(wav_files):
                            target_f = wav_files[idx_val]
                            print(f"[*] 선택된 파일: {target_f.name}")
                            if hasattr(ap, "execute_analysis_flow"):
                                try:
                                    ap.execute_analysis_flow(model, str(target_f), [], True)
                                except Exception as ex:
                                    log_error(MODULE_NAME, f"선택 파일 분석 플로우 실행 실패 ({target_f.name})", ex)
                            else:
                                log_error(MODULE_NAME, "분석 실행 함수(execute_analysis_flow)를 찾을 수 없습니다.", AttributeError("execute_analysis_flow missing"))
                        else:
                            print("[알림] 올바르지 않은 번호입니다.")
                            log_error(MODULE_NAME, f"범위를 벗어난 WAV 파일 번호 입력됨: {sub_choice}", ValueError("Choice out of range"))
                    except Exception as e:
                        log_error(MODULE_NAME, "WAV 파일 번호 파싱 또는 실행 중 예외 발생", e)
                elif sub_choice != "":
                    print("[알림] 올바른 숫자를 입력해주세요.")
                    log_error(MODULE_NAME, f"잘못된 WAV 파일 번호 형식 입력됨: {sub_choice}", ValueError("Invalid input format"))
            else:
                print("[알림] 'audio' 폴더에 분석 가능한 WAV 파일이 없습니다.")
                log_error(MODULE_NAME, "'audio' 폴더에 분석 가능한 WAV 파일이 없습니다.", FileNotFoundError("No WAV files found"))
        else:
            print("[알림] 'audio' 폴더가 존재하지 않거나 경로를 찾을 수 없습니다.")
            log_error(MODULE_NAME, "'audio' 폴더 경로를 찾을 수 없습니다.", FileNotFoundError("audio directory missing"))
    except Exception as e:
        log_error(MODULE_NAME, "handle_audio_selection 함수 실행 중 치명적 예외 발생", e, debug=True)

def main():
    try:
        # 1. 환경 설정 초기화 (가장 먼저 수행하여 하위 모듈들이 참조 가능하게 함)
        init_default_config()
    except Exception as e:
        log_error(MODULE_NAME, "환경 설정 초기화 중 예외 발생", e)

    # 2. 필수 디렉토리 검증 및 생성
    try:
        rec.ensure_directories()
    except Exception as e:
        log_error(MODULE_NAME, "rec.ensure_directories 실행 실패", e)

    try:
        ap.ensure_directories()
    except Exception as e:
        log_error(MODULE_NAME, "ap.ensure_directories 실행 실패", e)
        
    model = None
    
    while True:
        print("\n============================================== ")
        print("                  메인 메뉴                    ")
        print("============================================== ")
        print(" 0. 허깅페이스 토큰 설정")
        print(" 1. 실시간 시스템 음성 녹화 및 분석 (UVR5 보컬 전용)")
        print(" 2. 'audio' 폴더 WAV 파일 선택하여 분석 (UVR5 보컬 전용)")
        print(" 3. 데이터 정제 및 화자 알고리즘 등록/업데이트 (Web UI 스튜디오)")
        print(" 4. 프로그램 종료")
        print("============================================== ")
        
        try:
            choice = input("선택: ").strip()
        except Exception as e:
            log_error(MODULE_NAME, "메인 메뉴 선택 입력 수신 중 예외 발생", e)
            continue
        
        if choice == "0":
            try:
                set_huggingface_token_menu()
            except Exception as e:
                log_error(MODULE_NAME, "토큰 설정 메뉴 실행 중 예외 발생", e)
        elif choice == "1":
            try:
                rec.run_rec_menu(model)
            except Exception as e:
                log_error(MODULE_NAME, "녹화 메뉴 실행 중 예외 발생", e)
        elif choice == "2":
            try:
                handle_audio_selection(model)
            except Exception as e:
                log_error(MODULE_NAME, "오디오 선택 기능 실행 중 예외 발생", e)
        elif choice == "3":
            try:
                us.run_data_refinement_webui()
            except Exception as e:
                log_error(MODULE_NAME, "웹 UI 스튜디오 실행 중 예외 발생", e)
        elif choice == "4":
            print("[안내] 프로그램을 종료합니다.")
            log_info(MODULE_NAME, "프로그램 정상 종료")
            sys.exit(0)
        else:
            print("[알림] 올바른 메뉴 번호를 입력해주세요.")
            log_error(MODULE_NAME, f"잘못된 메인 메뉴 번호 입력됨: {choice}", ValueError("Invalid main menu choice"))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_error(MODULE_NAME, "애플리케이션 메인 실행 루프 중 치명적 예외 발생", e, debug=True)
        sys.exit(1)
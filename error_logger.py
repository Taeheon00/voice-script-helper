import sys
import traceback
from pathlib import Path
from datetime import datetime
import uuid

# 에러 로그를 저장할 기본 디렉토리 설정
ERROR_LOG_DIR = Path("error_log")

# 프로그램이 켜진 시점에 고유 세션 ID 생성 (재시작될 때마다 새로 생성됨)
_SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
_SESSION_INITIALIZED = False

def _ensure_log_dir():
    try:
        ERROR_LOG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def _get_current_log_file():
    """
    현재 프로그램 실행 세션에 대응하는 고유 로그 파일 경로를 반환합니다.
    (프로그램이 켜져 있는 동안에는 동일한 파일 경로 유지, 새로 켜지면 새로운 파일 생성)
    """
    _ensure_log_dir()
    # 세션별로 독립된 파일명을 사용하되, 날짜와 세션ID를 조합하여 고유성 확보
    file_timestamp = datetime.now().strftime("%Y%m%d")
    return ERROR_LOG_DIR / f"error_{file_timestamp}_{_SESSION_ID}.txt"

def log_error(module_name, message, exception=None, debug=False):
    """
    공통 에러 로그 출력 및 파일 기록 함수
    """
    global _SESSION_INITIALIZED
    _ensure_log_dir()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file_path = _get_current_log_file()
    
    error_msg = f"[오류][{module_name}] {message}"
    if exception:
        error_msg += f" | 상세: {exception}"
    
    # 1. 터미널 출력
    print(error_msg, file=sys.stderr)
    
    if debug and exception:
        traceback.print_exc()

    # 2. error_log 폴더 내에 텍스트 파일로 기록 (세션별 유지)
    try:
        log_content = ""
        # 프로그램 실행 후 해당 파일에 처음 기록하는 경우 세션 시작 헤더 추가
        if not _SESSION_INITIALIZED:
            log_content += f"="*60 + f"\n[SESSION START] 새로운 프로그램 실행 세션 감지 (Session ID: {_SESSION_ID})\n" + "="*60 + "\n"
            _SESSION_INITIALIZED = True

        log_content += f"[{timestamp}] [{module_name}] {message}"
        if exception:
            log_content += f"\n  - 상세 예외: {exception}"
            if debug:
                tb_str = traceback.format_exc()
                if tb_str and tb_str.strip() != "NoneType: None":
                    log_content += f"\n  - 트레이스백:\n{tb_str}"
        log_content += "\n" + "-"*60 + "\n"
        
        # 'a' 모드로 열어 현재 실행 중 발생하는 에러들은 계속 이어 붙임 (덮어쓰지 않음)
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(log_content)
    except Exception as e:
        print(f"[시스템 경고] 에러 로그 파일 저장 중 추가 예외 발생: {e}", file=sys.stderr)

def log_info(module_name, message):
    """
    공통 일반 안내 로그 출력 함수
    """
    print(f"[*] [{module_name}] {message}")
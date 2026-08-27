import sys
import traceback
from pathlib import Path
from datetime import datetime
import uuid

ERROR_LOG_DIR = Path("error_log")
_SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
_SESSION_INITIALIZED = False

# 직전에 기록된 에러 메시지를 기억하여 중복 출력 방지
_LAST_LOGGED_ERROR = None

def _ensure_log_dir():
    try:
        ERROR_LOG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def _get_current_log_file():
    _ensure_log_dir()
    file_timestamp = datetime.now().strftime("%Y%m%d")
    return ERROR_LOG_DIR / f"error_{file_timestamp}_{_SESSION_ID}.txt"

def log_error(module_name, message, exception=None, debug=False):
    global _SESSION_INITIALIZED, _LAST_LOGGED_ERROR
    _ensure_log_dir()
    
    error_msg = f"[오류][{module_name}] {message}"
    if exception:
        error_msg += f" | 상세: {exception}"
    
    # 동일한 에러 메시지가 연속해서 들어오는 경우 중복 출력/기록 차단
    if _LAST_LOGGED_ERROR == error_msg:
        return
    _LAST_LOGGED_ERROR = error_msg

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file_path = _get_current_log_file()
    
    # 1. 터미널 출력
    print(error_msg, file=sys.stderr)
    
    if debug and exception:
        traceback.print_exc()

    # 2. error_log 폴더 내에 텍스트 파일로 기록
    try:
        log_content = ""
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
        
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(log_content)
    except Exception as e:
        print(f"[시스템 경고] 에러 로그 파일 저장 중 추가 예외 발생: {e}", file=sys.stderr)

def log_info(module_name, message):
    print(f"[*] [{module_name}] {message}")
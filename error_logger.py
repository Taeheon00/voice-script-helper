import sys
import traceback
from pathlib import Path
from datetime import datetime

# 에러 로그를 저장할 기본 디렉토리 설정
ERROR_LOG_DIR = Path("error_log")

def _ensure_log_dir():
    try:
        ERROR_LOG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def log_error(module_name, message, exception=None, debug=False):
    """
    공통 에러 로그 출력 및 파일 기록 함수
    """
    _ensure_log_dir()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_timestamp = datetime.now().strftime("%Y%m%d")
    
    error_msg = f"[오류][{module_name}] {message}"
    if exception:
        error_msg += f" | 상세: {exception}"
    
    # 1. 터미널 출력
    print(error_msg, file=sys.stderr)
    
    if debug and exception:
        traceback.print_exc()

    # 2. error_log 폴더 내에 텍스트 파일로 기록 (날짜별 통합 파일)
    try:
        log_file_path = ERROR_LOG_DIR / f"error_{file_timestamp}.txt"
        log_content = f"[{timestamp}] [{module_name}] {message}"
        if exception:
            log_content += f"\n  - 상세 예외: {exception}"
            if debug:
                # 트레이스백 내용을 문자열로 가져와서 기록
                tb_str = traceback.format_exc()
                if tb_str and tb_str.strip() != "NoneType: None":
                    log_content += f"\n  - 트레이스백:\n{tb_str}"
        log_content += "\n" + "-"*60 + "\n"
        
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(log_content)
    except Exception as e:
        print(f"[시스템 경고] 에러 로그 파일 저장 중 추가 예외 발생: {e}", file=sys.stderr)

def log_info(module_name, message):
    """
    공통 일반 안내 로그 출력 함수
    """
    print(f"[*] [{module_name}] {message}")
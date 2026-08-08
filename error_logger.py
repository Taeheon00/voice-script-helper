import sys
import traceback

def log_error(module_name, message, exception=None, debug=False):
    """
    공통 에러 로그 출력 함수
    """
    error_msg = f"[오류][{module_name}] {message}"
    if exception:
        error_msg += f" | 상세: {exception}"
    
    print(error_msg, file=sys.stderr)
    
    if debug and exception:
        traceback.print_exc()

def log_info(module_name, message):
    """
    공통 일반 안내 로그 출력 함수
    """
    print(f"[*] [{module_name}] {message}")
import sys
import os
import audio_processor as ap
import ui_studio as us

TOKEN_FILE = "hf_token.txt"

def set_huggingface_token_menu():
    print("\n==============================================")
    print("             허깅페이스 토큰 설정")
    print("==============================================")
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r", encoding="utf-8") as f:
            current_token = f.read().strip()
        if current_token:
            masked = current_token[:4] + "*" * (len(current_token) - 4) if len(current_token) > 4 else "****"
            print(f"[*] 현재 설정된 토큰: {masked}")
        else:
            print("[*] 현재 설정된 토큰: 없음 (비어있음)")
    else:
        print("[*] 현재 설정된 토큰: 없음")
        
    print("----------------------------------------------")
    new_token = input("새로운 허깅페이스 토큰을 입력하세요 (취소하려면 엔터): ").strip()
    if new_token:
        with open(TOKEN_FILE, "w", encoding="utf-8") as f:
            f.write(new_token)
        print("[+ 성공] 허깅페이스 토큰이 성공적으로 저장되었습니다.")
    else:
        print("[알림] 토큰 설정이 변경되지 않았습니다.")
    print("==============================================")

def main():
    ap.ensure_directories()
    model = None
    while True:
        print("\n==============================================")
        print("                 메인 메뉴")
        print("==============================================")
        print(" 0. 허깅페이스 토큰 설정")
        print(" 1. 실시간 시스템 음성 녹화 및 분석 (UVR5 보컬 전용)")
        print(" 2. 'audio' 폴더 WAV 파일 선택하여 분석 (UVR5 보컬 전용)")
        print(" 3. 데이터 정제 및 화자 알고리즘 등록/업데이트 (Web UI 스튜디오)")
        print(" 4. 프로그램 종료")
        print("==============================================")
        choice = input("선택: ").strip()
        
        if choice == "0": 
            set_huggingface_token_menu()
        elif choice == "1":
            if model is None: model = ap.load_asr_model()
            if model: ap.record_and_transcribe(model)
        elif choice == "2":
            if model is None: model = ap.load_asr_model()
            if model: ap.select_and_process_audio_file(model)
        elif choice == "3":
            us.run_data_refinement_webui()
        elif choice == "4": 
            sys.exit(0)

if __name__ == "__main__":
    main()
import sys
import os
import ui_studio as us
import rec               # 녹화 및 캡처 전용 모듈 연동
import audio_processor as ap  # 오디오 분석 전용 모듈 연동

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
    rec.ensure_directories()
    ap.ensure_directories()  # 분석 모듈 전용 디렉토리 생성 보장
    model = None             # 모델 객체 (필요 시 로드)
    
    while True:
        print("\n==============================================")
        print("             음성 학습 프로그램")
        print("==============================================")
        print("                 메인 메뉴")
        print("==============================================")
        print(" 0. 허깅페이스 토큰 설정")
        print(" 1. 실시간 시스템 음성 녹화 및 분석 (UVR5 보컬 전용)")
        print(" 2. 'audio' 폴더 WAV 파일 선택하여 분석 (UVR5 보컬 전용)")
        print(" 3. 데이터 정제 및 화자 알고리즘 등록/업데이트 (Web UI 스튜디오)")
        print(" 4. 프로그램 종료")
        print("============================================== ")
        choice = input("선택: ").strip()
        
        if choice == "0": 
            set_huggingface_token_menu()
        elif choice == "1":
            # 실시간 녹화 및 분석 메뉴 실행 (rec.py)
            rec.run_rec_menu(model)
        elif choice == "2":
            # 'audio' 폴더 내 WAV 파일 선택 분석 기능 실행 (audio_processor.py)
            if hasattr(ap, "select_and_process_audio_file"):
                ap.select_and_process_audio_file(model)
            elif hasattr(ap, "run_audio_file_selection"):
                ap.run_audio_file_selection(model)
            else:
                print("\n[안내] 'audio' 폴더 파일 선택 분석 기능을 실행합니다.")
                audio_dir = ap.AUDIO_DIR
                if audio_dir.exists():
                    wav_files = sorted([f for f in audio_dir.glob("*.wav") if "vocal" not in f.name.lower() and "instrumental" not in f.name.lower()])
                    if wav_files:
                        print("\n[audio 폴더 WAV 파일 목록]")
                        for idx, wf in enumerate(wav_files, 1):
                            print(f" {idx}. {wf.name}")
                        sub_choice = input("분석할 파일 번호를 선택하세요 (취소는 엔터): ").strip()
                        if sub_choice.isdigit():
                            idx_val = int(sub_choice) - 1
                            if 0 <= idx_val < len(wav_files):
                                target_f = wav_files[idx_val]
                                print(f"[*] 선택된 파일: {target_f.name}")
                                if hasattr(ap, "execute_analysis_flow"):
                                    ap.execute_analysis_flow(model, str(target_f), [], True)
                                else:
                                    print("[오류] 분석 실행 함수(execute_analysis_flow)를 찾을 수 없습니다.")
                    else:
                        print("[알림] 'audio' 폴더에 분석 가능한 WAV 파일이 없습니다.")
                else:
                    print("[알림] 'audio' 폴더가 존재하지 않습니다.")
        elif choice == "3":
            us.run_data_refinement_webui()
        elif choice == "4": 
            sys.exit(0)

if __name__ == "__main__":
    main()
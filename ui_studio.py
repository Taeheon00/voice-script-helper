import os
import re
import tempfile
import sys
import asyncio
import logging
import json
import numpy as np
import gradio as gr
from pydub import AudioSegment
import algorithm_handler as ah
from error_logger import log_error, log_info

MODULE_NAME = "RefinementStudioUI"

SEGMENTS_BASE_DIR = "segments_base"
ALGORITHM_DIR = "saved_algorithms"  # algorithm_handler의 STORAGE_DIR과 경로 일치시킴
ITEMS_PER_PAGE = 10

if not hasattr(ah, "split_audio_segment"):
    def _fallback_split_audio_segment(wav_path, txt_path):
        return True, "성공", []
    ah.split_audio_segment = _fallback_split_audio_segment

if not hasattr(ah, "merge_audio_segments"):
    def _fallback_merge_audio_segments(wav_path1, wav_path2, txt_path1, txt_path2):
        return True, "성공", None
    ah.merge_audio_segments = _fallback_merge_audio_segments

def has_wav_files(directory):
    """해당 디렉토리 내부에 WAV 파일이 존재하는지 확인"""
    if not os.path.exists(directory):
        return False
    try:
        return any(f.lower().endswith('.wav') for f in os.listdir(directory))
    except Exception:
        return False

def get_single_speaker_folders():
    """단일 화자 분석 폴더(실제 WAV 파일이 있는 폴더) 조회"""
    if not os.path.exists(SEGMENTS_BASE_DIR):
        log_error(MODULE_NAME, f"단일 화자 폴더 탐색 실패: 기본 디렉토리가 존재하지 않습니다 ({SEGMENTS_BASE_DIR})", FileNotFoundError())
        return []
    folders = []
    try:
        for root, dirs, files in os.walk(SEGMENTS_BASE_DIR):
            for d in dirs:
                d_lower = d.lower()
                if d_lower.startswith("single_segment") or d_lower.startswith("single_auto_recorded"):
                    target_path = os.path.join(root, d)
                    if has_wav_files(target_path):
                        rel_path = os.path.relpath(target_path, SEGMENTS_BASE_DIR)
                        folders.append(rel_path)
    except Exception as e:
        log_error(MODULE_NAME, "단일 화자 폴더 탐색 중 예외 발생", e)
    return folders

def get_basic_segment_folders():
    """기본 분석 폴더(실제 WAV 파일이 있는 폴더) 조회 (원본 탐색 로직 복구)"""
    if not os.path.exists(SEGMENTS_BASE_DIR):
        log_error(MODULE_NAME, f"기본 분석 폴더 탐색 실패: 기본 디렉토리가 존재하지 않습니다 ({SEGMENTS_BASE_DIR})", FileNotFoundError())
        return []
    folders = []
    try:
        for root, dirs, files in os.walk(SEGMENTS_BASE_DIR):
            for d in dirs:
                d_lower = d.lower()
                if "single" in d_lower or "multi" in d_lower:
                    continue
                if d_lower.startswith("segment") or d_lower.startswith("auto_recorded"):
                    target_path = os.path.join(root, d)
                    if has_wav_files(target_path):
                        rel_path = os.path.relpath(target_path, SEGMENTS_BASE_DIR)
                        folders.append(rel_path)
    except Exception as e:
        log_error(MODULE_NAME, "기본 분석 폴더 탐색 중 예외 발생", e)
    return folders

def get_algorithm_json_files():
    """알고리즘 JSON 파일 목록 조회 및 '-- 선택 안함 --' 기본 옵션 추가"""
    files = []
    if os.path.exists(ALGORITHM_DIR):
        try:
            files = [f for f in os.listdir(ALGORITHM_DIR) if f.lower().endswith('.json')]
        except Exception as e:
            log_error(MODULE_NAME, "알고리즘 JSON 파일 탐색 중 예외 발생", e)
            return ["-- 선택 안함 --"]
    return ["-- 선택 안함 --"] + files

def clone_items(items):
    return [item.copy() for item in items]

def extract_speaker_from_filename(filename):
    """파일명에서 화자 번호를 추출하고 1을 더해 반환 (예: speaker_00 -> 1)"""
    matches = re.findall(r'speaker_(\d+)', filename, re.IGNORECASE)
    if matches:
        try:
            num = int(matches[0]) + 1
            return str(num), True
        except ValueError:
            pass
    return "1", False

def check_folder_speaker_type(target_dir):
    """
    폴더 전체의 파일명을 스캔하여 화자 번호가 몇 종류인지 확인합니다.
    """
    if not os.path.exists(target_dir):
        return "1", False
        
    try:
        files = os.listdir(target_dir)
        all_speakers = set()
        for f in files:
            spk_str, found = extract_speaker_from_filename(f)
            if found:
                all_speakers.add(spk_str)
                
        if not all_speakers:
            return "1", False
            
        unique_speakers = sorted(list(all_speakers), key=lambda x: int(x) if x.isdigit() else x)
        if len(unique_speakers) == 1:
            return unique_speakers[0], False  # 단일 화자
        else:
            return ", ".join(unique_speakers), True  # 다중 화자
    except Exception as e:
        log_error(MODULE_NAME, "폴더 화자 유형 분석 중 예외 발생", e)
        return "1", False

def get_segment_label(item, segment_number):
    """
    다중 화자일 경우 개별 파일의 화자 번호(1부터 시작)를 반영하여 '세그먼트 N-화자X' 형태로 표시합니다.
    """
    is_mixed = item.get("is_mixed", False)
    speaker_num = item.get("speaker_num", "1")
    if is_mixed:
        return f"세그먼트 {segment_number}-화자{speaker_num}"
    return f"세그먼트 {segment_number}"

def inspect_all_files(target_dir):
    if not os.path.exists(target_dir):
        msg = f"존재하지 않는 폴더입니다: {target_dir}"
        log_error(MODULE_NAME, msg, FileNotFoundError(target_dir))
        return False, msg
    try:
        files = os.listdir(target_dir)
        wav_files = [f for f in files if f.lower().endswith('.wav')]
        if not wav_files:
            msg = "해당 폴더에 WAV 파일이 없습니다."
            log_error(MODULE_NAME, f"{msg} ({target_dir})", ValueError(msg))
            return False, msg
        issues = []
        for w_f in wav_files:
            base_name = os.path.splitext(w_f)[0]
            t_path = os.path.join(target_dir, f"{base_name}.txt")
            if not os.path.exists(t_path):
                issues.append(f"누락된 텍스트 파일: {base_name}.txt")
        if issues:
            msg = f"전체 검사 실패 (총 {len(issues)}개 문제 발견):\n" + "\n".join(issues[:3])
            log_error(MODULE_NAME, f"파일 전체 검사 문제 발생: {target_dir} - {msg}", ValueError(msg))
            return False, msg
        return True, ""
    except Exception as e:
        log_error(MODULE_NAME, f"파일 전체 검사 중 오류 발생: {target_dir}", e)
        return False, f"검사 중 오류 발생: {e}"

class ConnectionDisconnectFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if record.exc_info:
            exc_type, exc_value, _ = record.exc_info
            if exc_type:
                message += f" {exc_type.__name__} {str(exc_value)}"
        ignore_keywords = [
            "ConnectionResetError", "WinError 10054", "connection reset",
            "connection closed", "broken pipe", "websockets.exceptions", "peer closed connection"
        ]
        lower_msg = message.lower()
        for kw in ignore_keywords:
            if kw.lower() in lower_msg:
                return False
        return True

class StderrDisconnectFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.ignore_keywords = [
            "ConnectionResetError", "WinError 10054", "connection reset",
            "connection closed", "broken pipe", "websockets.exceptions", "peer closed connection"
        ]

    def write(self, text):
        lower_text = text.lower()
        if any(kw.lower() in lower_text for kw in self.ignore_keywords):
            return
        self.original_stderr.write(text)

    def flush(self):
        self.original_stderr.flush()

def setup_connection_log_suppression():
    target_loggers = ["uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.asgi", "websockets"]
    filter_instance = ConnectionDisconnectFilter()
    for logger_name in target_loggers:
        lg = logging.getLogger(logger_name)
        lg.addFilter(filter_instance)
        for handler in lg.handlers:
            if filter_instance not in handler.filters:
                handler.addFilter(filter_instance)
    root_lg = logging.getLogger()
    root_lg.addFilter(filter_instance)
    for handler in root_lg.handlers:
        if filter_instance not in handler.filters:
            handler.addFilter(filter_instance)
    if not isinstance(sys.stderr, StderrDisconnectFilter):
        sys.stderr = StderrDisconnectFilter(sys.stderr)

def run_data_refinement_webui():
    log_info(MODULE_NAME, "Web UI 스튜디오를 실행합니다...")
    setup_connection_log_suppression()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    single_folders = get_single_speaker_folders()
    basic_folders = get_basic_segment_folders()
    algo_json_files = get_algorithm_json_files()
    
    default_single = single_folders[0] if single_folders else None
    default_basic = basic_folders[0] if basic_folders else None
    default_algo_selection = algo_json_files[0] if algo_json_files else "-- 선택 안함 --"

    active_temp_files = set()

    def create_tracked_temp_wav():
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        active_temp_files.add(temp_path)
        return temp_path

    def cleanup_old_temp_files():
        for p in list(active_temp_files):
            if not os.path.exists(p):
                active_temp_files.discard(p)

    custom_css = """
    .wrap.svelte-1p90v75, .options {
        max-height: 200px !important;
        overflow-y: auto !important;
    }
    .page-info, .page-info-bottom, .page-info * {
        text-align: center !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
    }
    .segment-checkbox {
        width: 28px !important; height: 28px !important;
        min-width: 28px !important; min-height: 28px !important;
        max-width: 28px !important; max-height: 28px !important;
        padding: 0 !important; margin: 0 auto !important;
        background-color: #ffffff !important; border: 1px solid #e0e0e0 !important; border-radius: 6px !important;
        display: flex !important; align-items: center !important; justify-content: center !important;
        box-shadow: none !important;
    }
    .segment-checkbox input[type="checkbox"] {
        width: 18px !important; height: 18px !important;
        min-width: 18px !important; min-height: 18px !important;
        margin: 0 auto !important; cursor: pointer !important; accent-color: #ff7f50 !important;
        pointer-events: auto !important;
    }
    .align-stretch-row {
        align-items: stretch !important;
    }
    .tall-action-btn, .tall-action-btn button {
        height: 100% !important;
        min-height: 100px !important;
        font-size: 16px !important;
        font-weight: bold !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    """

    with gr.Blocks(title="데이터 정제 스튜디오", css=custom_css) as demo:
        state_items = gr.State([])
        state_history = gr.State([])
        state_redo = gr.State([])
        state_page = gr.State(1)

        with gr.Row():
            gr.Markdown("## 🛠️ 데이터 정제 및 화자 알고리즘 등록 스튜디오")
            close_ui_btn = gr.Button("🚪 Web UI 종료 ", variant="stop")

        gr.Markdown("💡 **안내:** 종료버튼 클릭 시 터미널에서 엔터를 누르면 메뉴로 돌아가게 됩니다.")

        with gr.Row(variant="panel"):
            with gr.Column():
                gr.Markdown("### 🎤 단일 화자 분석 폴더 선택")
                single_dropdown = gr.Dropdown(choices=single_folders, value=default_single, label="단일 화자 세그먼트", interactive=True)
                load_single_btn = gr.Button("📁 단일 화자 불러오기", variant="primary")
            with gr.Column():
                gr.Markdown("### 📁 기본 분석 폴더 선택")
                basic_dropdown = gr.Dropdown(choices=basic_folders, value=default_basic, label="기본 분석 세그먼트", interactive=True)
                load_basic_btn = gr.Button("📁 기본 분석 불러오기", variant="primary")

        with gr.Row():
            refresh_btn = gr.Button("🔄 전체 목록 새로고침")

        with gr.Row(variant="panel", elem_classes=["align-stretch-row"]):
            with gr.Column(scale=2):
                speaker_name_input = gr.Textbox(label="등록할 화자 이름", placeholder="예: 화자_a", lines=1, interactive=True)
                algo_json_dropdown = gr.Dropdown(choices=algo_json_files, value=default_algo_selection, label="기존 알고리즘 JSON 선택", interactive=True)
            with gr.Column(scale=1):
                save_all_btn = gr.Button("💾 저장", variant="secondary", elem_classes=["tall-action-btn"])
            with gr.Column(scale=2):
                register_algo_btn = gr.Button("알고리즘 등록/업데이트", variant="primary", elem_classes=["tall-action-btn"])

        info_box = gr.Textbox(label="현재 진행 상황", value="작업할 폴더를 선택하고 '불러오기' 버튼을 누르세요.", interactive=False)

        with gr.Row(elem_classes=["pagination-toolbar"]):
            prev_btn = gr.Button("⬅️ 이전 페이지 ", interactive=False)
            page_info_md = gr.Markdown("### 페이지: 0 / 0 (총 0개 항목)", elem_classes=["page-info"])
            merge_selected_btn = gr.Button("🔗 선택 합병", variant="secondary", scale=0, min_width=130)
            undo_global_btn = gr.Button("↩️", elem_classes=["icon-btn"], interactive=False, scale=0, min_width=55)
            redo_global_btn = gr.Button("🔁", elem_classes=["icon-btn"], interactive=False, scale=0, min_width=55)
            next_btn = gr.Button("다음 페이지 ➡️ ", interactive=False)

        row_components = []
        audio_components = []
        text_components = []
        select_checkbox_components = []
        txt_path_components = []
        wav_path_components = []
        delete_btn_components = []
        split_btn_components = []

        for i in range(ITEMS_PER_PAGE):
            with gr.Row(visible=False) as r_box:
                with gr.Column(scale=2):
                    a_comp = gr.Audio(label=f"세그먼트 {i+1}", type="filepath", interactive=False, playback_position=0)
                with gr.Column(scale=3):
                    t_comp = gr.Textbox(label=f"세그먼트 {i+1}", lines=2, interactive=True)
                with gr.Column(scale=1, min_width=110):
                    with gr.Row():
                        del_btn = gr.Button("🗑️ 삭제", variant="stop", scale=1)
                        chk_comp = gr.Checkbox(value=False, show_label=False, elem_classes=["segment-checkbox"], scale=0, min_width=30)
                        split_btn = gr.Button("✂️ 분할", variant="secondary", scale=1)
                tp_comp = gr.Textbox(visible=False)
                wp_comp = gr.Textbox(visible=False)
                
            row_components.append(r_box)
            audio_components.append(a_comp)
            text_components.append(t_comp)
            select_checkbox_components.append(chk_comp)
            delete_btn_components.append(del_btn)
            split_btn_components.append(split_btn)
            txt_path_components.append(tp_comp)
            wav_path_components.append(wp_comp)

        with gr.Row():
            with gr.Column(scale=1): prev_btn_bottom = gr.Button("⬅️ 이전 페이지 ", interactive=False)
            with gr.Column(scale=1): page_info_md_bottom = gr.Markdown("### 페이지: 0 / 0", elem_classes=["page-info-bottom"])
            with gr.Column(scale=1): next_btn_bottom = gr.Button("다음 페이지 ➡️ ", interactive=False)

        def on_textbox_change(val):
            if val and val.strip():
                return gr.Dropdown(value="-- 선택 안함 --")
            return gr.Dropdown()

        def on_dropdown_change(val):
            if val and val != "-- 선택 안함 --":
                return gr.Textbox(value="")
            return gr.Textbox()

        speaker_name_input.change(fn=on_textbox_change, inputs=[speaker_name_input], outputs=[algo_json_dropdown])
        algo_json_dropdown.change(fn=on_dropdown_change, inputs=[algo_json_dropdown], outputs=[speaker_name_input])

        def handle_refresh_click():
            new_choices = get_algorithm_json_files()
            default_val = new_choices[0] if new_choices else "-- 선택 안함 --"
            return (
                gr.Dropdown(choices=get_single_speaker_folders()),
                gr.Dropdown(choices=get_basic_segment_folders()),
                gr.Dropdown(choices=new_choices, value=default_val)
            )

        def get_pagination_states(page_num, total_pages):
            has_prev = page_num > 1
            has_next = page_num < total_pages
            return (
                gr.update(interactive=has_prev),
                gr.update(interactive=has_next),
                gr.update(interactive=has_prev),
                gr.update(interactive=has_next)
            )

        def load_folder_data(folder_name, is_single_mode=False):
            cleanup_old_temp_files()
            empty_pagination = (gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False))
            if not folder_name:
                msg = "폴더가 선택되지 않았습니다."
                log_error(MODULE_NAME, msg, ValueError(msg))
                return [[], [], [], 1, msg, "### 페이지: 0 / 0 (총 0개 항목)", "### 페이지: 0 / 0"] + list(empty_pagination) + [gr.update(visible=False), gr.update(value=None, label="세그먼트 1"), gr.update(value="", label="세그먼트 1"), False, "", ""] * ITEMS_PER_PAGE
            
            target_dir = os.path.join(SEGMENTS_BASE_DIR, folder_name)
            is_valid, inspect_msg = inspect_all_files(target_dir)
            if not is_valid:
                return [[], [], [], 1, inspect_msg, "### 페이지: 0 / 0 (총 0개 항목)", "### 페이지: 0 / 0"] + list(empty_pagination) + [gr.update(visible=False), gr.update(value=None, label="세그먼트 1"), gr.update(value="", label="세그먼트 1"), False, "", ""] * ITEMS_PER_PAGE

            if not os.path.exists(target_dir):
                msg = "존재하지 않는 폴더입니다."
                log_error(MODULE_NAME, f"{msg}: {target_dir}", FileNotFoundError(target_dir))
                return [[], [], [], 1, msg, "### 페이지: 0 / 0 (총 0개 항목)", "### 페이지: 0 / 0"] + list(empty_pagination) + [gr.update(visible=False), gr.update(value=None, label="세그먼트 1"), gr.update(value="", label="세그먼트 1"), False, "", ""] * ITEMS_PER_PAGE
                
            try:
                wav_files = sorted([f for f in os.listdir(target_dir) if f.lower().endswith('.wav')])
            except Exception as e:
                log_error(MODULE_NAME, f"디렉토리 읽기 오류: {folder_name}", e)
                return [[], [], [], 1, f"디렉토리 읽기 오류: {e}", "### 페이지: 0 / 0 (총 0개 항목)", "### 페이지: 0 / 0"] + list(empty_pagination) + [gr.update(visible=False), gr.update(value=None, label="세그먼트 1"), gr.update(value="", label="세그먼트 1"), False, "", ""] * ITEMS_PER_PAGE

            if not wav_files:
                msg = "해당 폴더에 WAV 파일이 없습니다."
                log_error(MODULE_NAME, f"{msg} ({folder_name})", ValueError(msg))
                return [[], [], [], 1, msg, "### 페이지: 0 / 0 (총 0개 항목)", "### 페이지: 0 / 0"] + list(empty_pagination) + [gr.update(visible=False), gr.update(value=None, label="세그먼트 1"), gr.update(value="", label="세그먼트 1"), False, "", ""] * ITEMS_PER_PAGE
                
            _, folder_is_mixed = check_folder_speaker_type(target_dir)

            items = []
            single_count = 0
            multi_count = 0

            for idx, w_f in enumerate(wav_files):
                base_name = os.path.splitext(w_f)[0]
                w_path = os.path.join(target_dir, w_f)
                t_path = os.path.join(target_dir, f"{base_name}.txt")
                
                t_content = ""
                if os.path.exists(t_path):
                    try:
                        with open(t_path, "r", encoding="utf-8") as tf:
                            t_content = tf.read().strip()
                    except Exception as e:
                        log_error(MODULE_NAME, f"텍스트 파일 읽기 오류: {t_path}", e)
                        t_content = ""
                
                if is_single_mode:
                    speaker_num, is_mixed = "1", False
                    single_count += 1
                else:
                    speaker_num, _ = extract_speaker_from_filename(w_f)
                    is_mixed = folder_is_mixed
                    if is_mixed:
                        multi_count += 1
                    else:
                        single_count += 1

                items.append({
                    "wav": w_path, "txt": t_path, "content": t_content, "original_content": t_content, 
                    "deleted": False, "folder_name": folder_name, "wav_filename": w_f,
                    "speaker_num": speaker_num, "is_mixed": is_mixed, "selected": False,
                    "audio_segment": None, "is_new": False
                })

            surviving = [it for it in items if not it["deleted"]]
            total_pages = int(np.ceil(len(surviving) / ITEMS_PER_PAGE)) or 1
            status_msg = f"총 {len(items)}개의 세그먼트 로드 완료 (총 {total_pages}페이지)."
            
            surviving_page_items = surviving[:ITEMS_PER_PAGE]
            updates = []
            for i in range(ITEMS_PER_PAGE):
                if i < len(surviving_page_items):
                    item = surviving_page_items[i]
                    temp_path = create_tracked_temp_wav()
                    if item.get("audio_segment") is not None:
                        item["audio_segment"].export(temp_path, format="wav")
                    elif os.path.exists(item["wav"]):
                        import shutil
                        shutil.copyfile(item["wav"], temp_path)

                    label_str = get_segment_label(item, i + 1)
                    
                    updates.extend([gr.update(visible=True),gr.update(value=temp_path, label=label_str),gr.update(value=item["content"], label=label_str),item.get("selected", False),item["txt"],item["wav"]])
                                    
                else:
                    updates.extend([gr.update(visible=False), gr.update(value=None, label=f"세그먼트 {i+1}"), gr.update(value="", label=f"세그먼트 {i+1}"), False, "", ""])
            
            p_prev, p_next, p_prev_b, p_next_b = get_pagination_states(1, total_pages)
            return [items, [], [], 1, status_msg, f"### 페이지: 1 / {total_pages} (총 {len(surviving)}개 유효 항목)", f"### 페이지: 1 / {total_pages}", p_prev, p_next, p_prev_b, p_next_b] + updates

        def render_page(items, page_num):
            cleanup_old_temp_files()
            surviving = [it for it in items if not it["deleted"]]
            if not surviving:
                return [1, "### 페이지: 0 / 0 (총 0개 항목)", "### 페이지: 0 / 0", gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)] + [gr.update(visible=False), gr.update(value=None), gr.update(value=""), False, "", ""] * ITEMS_PER_PAGE
            
            total_pages = int(np.ceil(len(surviving) / ITEMS_PER_PAGE))
            if page_num < 1: page_num = 1
            if page_num > total_pages: page_num = total_pages
            
            start_idx = (page_num - 1) * ITEMS_PER_PAGE
            page_surviving = surviving[start_idx:start_idx + ITEMS_PER_PAGE]
            
            updates = []
            for i in range(ITEMS_PER_PAGE):
                if i < len(page_surviving):
                    item = page_surviving[i]
                    global_idx_display = start_idx + i + 1
                    label_str = get_segment_label(item, global_idx_display)
                    
                    temp_path = create_tracked_temp_wav()
                    if item.get("audio_segment") is not None:
                        item["audio_segment"].export(temp_path, format="wav")
                    elif os.path.exists(item["wav"]):
                        import shutil
                        shutil.copyfile(item["wav"], temp_path)

                    updates.extend([
                        gr.update(visible=True), 
                        gr.update(value=temp_path, label=label_str), 
                        gr.update(value=item["content"], label=label_str), 
                        item.get("selected", False),
                        item["txt"], 
                        temp_path
                    ])
                else:
                    updates.extend([gr.update(visible=False), gr.update(value=None, label=f"세그먼트 {i+1}"), gr.update(value="", label=f"세그먼트 {i+1}"), False, "", ""])
            
            p_prev, p_next, p_prev_b, p_next_b = get_pagination_states(page_num, total_pages)
            return [page_num, f"### 페이지: {page_num} / {total_pages} (총 {len(surviving)}개 유효 항목)", f"### 페이지: {page_num} / {total_pages}", p_prev, p_next, p_prev_b, p_next_b] + updates

        def sync_current_data(items, page_num, current_texts, current_checkboxes):
            if not items:
                return items
            updated = clone_items(items)
            surviving = [it for it in updated if not it["deleted"]]
            start_idx = (page_num - 1) * ITEMS_PER_PAGE
            page_surviving = surviving[start_idx:start_idx + ITEMS_PER_PAGE]
            
            for i, target_item in enumerate(page_surviving):
                if i < len(current_texts):
                    target_item["content"] = current_texts[i]
                if i < len(current_checkboxes):
                    target_item["selected"] = bool(current_checkboxes[i])
            return updated

        def change_page_wrapper(items, current_page, direction, *args):
            half = len(text_components)
            current_texts = args[:half]
            current_checkboxes = args[half:]
            
            updated = sync_current_data(items, current_page, current_texts, current_checkboxes)
            surviving = [it for it in updated if not it["deleted"]]
            total_pages = int(np.ceil(len(surviving) / ITEMS_PER_PAGE)) or 1
            new_page = max(1, min(current_page + direction, total_pages))
            
            res = render_page(updated, new_page)
            return [updated] + res

        def handle_text_commit(items, history, redo, page_num, *args):
            half = len(text_components)
            current_texts = args[:half]
            current_checkboxes = args[half:]
            if not items:
                return [items, history, redo, gr.update(interactive=bool(history)), gr.update(interactive=bool(redo))]
            
            current_state = sync_current_data(items, page_num, current_texts, current_checkboxes)
            if current_state != items:
                new_history = history + [clone_items(items)]
                new_redo = []
                return [current_state, new_history, new_redo, gr.update(interactive=True), gr.update(interactive=False)]
            
            return [items, history, redo, gr.update(interactive=bool(history)), gr.update(interactive=bool(redo))]

        def handle_single_delete(items, history, redo, page_num, item_index, *args):
            half = len(text_components)
            current_texts = args[:half]
            current_checkboxes = args[half:]
            if not items:
                msg = "항목이 없습니다."
                log_error(MODULE_NAME, f"삭제 실패: {msg}", ValueError(msg))
                empty_pagination = (gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False))
                return [items, history, redo, msg, gr.update(interactive=False), gr.update(interactive=False), page_num, "### 페이지: 0 / 0", "### 페이지: 0 / 0"] + list(empty_pagination) + [gr.update(visible=False), gr.update(value=None), gr.update(value=""), False, "", ""] * ITEMS_PER_PAGE
            
            new_items = sync_current_data(items, page_num, current_texts, current_checkboxes)
            surviving = [it for it in new_items if not it["deleted"]]
            start_idx = (page_num - 1) * ITEMS_PER_PAGE
            target_surviving_idx = start_idx + item_index
            
            if target_surviving_idx < len(surviving):
                target_item = surviving[target_surviving_idx]
                target_item["deleted"] = True
                target_item["selected"] = False
            else:
                msg = "잘못된 삭제 대상 인덱스입니다."
                log_error(MODULE_NAME, msg, IndexError(msg))

            new_history = history + [clone_items(items)]
            new_redo = []
            
            surviving_after = [it for it in new_items if not it["deleted"]]
            total_pages = int(np.ceil(len(surviving_after) / ITEMS_PER_PAGE)) or 1
            target_page = min(page_num, total_pages)
            
            res = render_page(new_items, target_page)
            return [new_items, new_history, new_redo, f"세그먼트 삭제 대기됨 (저장 시 반영)", gr.update(interactive=True), gr.update(interactive=False)] + res

        def handle_single_split(items, history, redo, page_num, item_index, playback: gr.Audio, *args):
            half = len(text_components)
            current_texts = args[:half]
            current_checkboxes = args[half:]

            if not items:
                msg = "항목이 없습니다."
                log_error(MODULE_NAME, f"분할 실패: {msg}", ValueError(msg))
                empty_pagination = (gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False))
                return [items, history, redo, msg, gr.update(interactive=bool(history)), gr.update(interactive=bool(redo)), page_num, "### 페이지: 0 / 0", "### 페이지: 0 / 0"] + list(empty_pagination) + [gr.update(visible=False), gr.update(value=None), gr.update(value=""), False, "", ""] * ITEMS_PER_PAGE
            
            new_items = sync_current_data(items, page_num, current_texts, current_checkboxes)
            surviving = [it for it in new_items if not it["deleted"]]
            start_idx = (page_num - 1) * ITEMS_PER_PAGE
            target_surviving_idx = start_idx + item_index
            
            if target_surviving_idx >= len(surviving):
                msg = "잘못된 인덱스입니다."
                log_error(MODULE_NAME, f"분할 실패: {msg}", IndexError(msg))
                res = render_page(new_items, page_num)
                return [new_items, history, redo, msg, gr.update(interactive=bool(history)), gr.update(interactive=bool(redo))] + res
            
            target_item = surviving[target_surviving_idx]

            try:
                audio = target_item.get("audio_segment")
                if audio is None:
                    source_wav = target_item["wav"]
                    if os.path.exists(source_wav):
                        audio = AudioSegment.from_wav(source_wav)
                    else:
                        msg = "오디오 소스를 찾을 수 없습니다."
                        log_error(MODULE_NAME, f"분할 실패: {msg} ({source_wav})", FileNotFoundError(source_wav))
                        res = render_page(new_items, page_num)
                        return [new_items, history, redo, msg, gr.update(interactive=bool(history)), gr.update(interactive=bool(redo))] + res
                
                if len(audio) < 1000:
                    msg = "오디오가 너무 짧아서 분할할 수 없습니다."
                    log_error(MODULE_NAME, f"분할 실패: {msg} (길이: {len(audio)}ms)", ValueError(msg))
                    res = render_page(new_items, page_num)
                    return [new_items, history, redo, msg, gr.update(interactive=bool(history)), gr.update(interactive=bool(redo))] + res

                position = float(playback.playback_position or 0.0)
                split_ms = int(position * 1000)

                part1_audio = audio[:split_ms]
                part2_audio = audio[split_ms:]

                target_item["deleted"] = True
                target_item["selected"] = False

                base_dir = os.path.dirname(target_item["wav"])
                base_name = os.path.splitext(target_item["wav_filename"])[0]
                current_content = target_item["content"]

                speaker_num = target_item.get("speaker_num", "1")
                is_mixed = target_item.get("is_mixed", False)

                split_items = [
                    {
                        "wav": os.path.join(base_dir, f"{base_name}_part1.wav"), "txt": os.path.join(base_dir, f"{base_name}_part1.txt"),
                        "content": current_content, "original_content": "", "deleted": False, "folder_name": target_item["folder_name"],
                        "wav_filename": f"{base_name}_part1.wav", "speaker_num": speaker_num, "is_mixed": is_mixed, "selected": False,
                        "audio_segment": part1_audio, "is_new": True
                    },
                    {
                        "wav": os.path.join(base_dir, f"{base_name}_part2.wav"), "txt": os.path.join(base_dir, f"{base_name}_part2.txt"),
                        "content": current_content, "original_content": "", "deleted": False, "folder_name": target_item["folder_name"],
                        "wav_filename": f"{base_name}_part2.wav", "speaker_num": speaker_num, "is_mixed": is_mixed, "selected": False,
                        "audio_segment": part2_audio, "is_new": True
                    }
                ]

                orig_idx = new_items.index(target_item)
                for ins_item in reversed(split_items):
                    new_items.insert(orig_idx + 1, ins_item)

                new_history = history + [clone_items(items)]
                new_redo = []
                
                surviving_after = [it for it in new_items if not it["deleted"]]
                total_pages = int(np.ceil(len(surviving_after) / ITEMS_PER_PAGE)) or 1
                target_page = min(page_num, total_pages)

                res = render_page(new_items, target_page)
                return [new_items, new_history, new_redo, f"세그먼트 분할 완료 (저장 시 적용)", gr.update(interactive=True), gr.update(interactive=False)] + res
            except Exception as e:
                log_error(MODULE_NAME, "세그먼트 분할 중 예외 발생", e)
                res = render_page(new_items, page_num)
                return [new_items, history, redo, f"분할 오류: {e}", gr.update(interactive=bool(history)), gr.update(interactive=bool(redo))] + res

        def handle_selected_merge(items, history, redo, page_num, *args):
            if not items:
                msg = "항목이 없습니다."
                log_error(MODULE_NAME, f"합병 실패: {msg}", ValueError(msg))
                empty_pagination = (gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False))
                return [items, history, redo, msg, gr.update(interactive=False), gr.update(interactive=False), page_num, "### 페이지: 0 / 0", "### 페이지: 0 / 0"] + list(empty_pagination) + [gr.update(visible=False), gr.update(value=None), gr.update(value=""), False, "", ""] * ITEMS_PER_PAGE
            
            half_len = len(text_components)
            current_texts = args[:half_len]
            checkbox_values = args[half_len:]

            new_items = sync_current_data(items, page_num, current_texts, checkbox_values)
            selected_items = [it for it in new_items if it.get("selected", False) and not it["deleted"]]

            if len(selected_items) < 2:
                msg = "합병하려면 최소 2개 이상의 세그먼트를 체크해주세요."
                log_error(MODULE_NAME, f"합병 실패: {msg}", ValueError(msg))
                res = render_page(new_items, page_num)
                return [new_items, history, redo, msg, gr.update(interactive=bool(history)), gr.update(interactive=bool(redo))] + res

            try:
                cur_audio = None
                combined_texts = []
                for target_it in selected_items:
                    t_audio = target_it.get("audio_segment")
                    if t_audio is None:
                        if os.path.exists(target_it["wav"]):
                            t_audio = AudioSegment.from_wav(target_it["wav"])
                        else:
                            msg = "오디오 소스를 찾을 수 없습니다."
                            log_error(MODULE_NAME, f"합병 실패: {msg}", FileNotFoundError(target_it['wav']))
                            res = render_page(new_items, page_num)
                            return [new_items, history, redo, msg, gr.update(interactive=bool(history)), gr.update(interactive=bool(redo))] + res
                    
                    if cur_audio is None:
                        cur_audio = t_audio
                    else:
                        cur_audio += t_audio
                    combined_texts.append(target_it["content"])

                if len(cur_audio) > 60000:
                    msg = "합병된 오디오의 길이가 60초를 초과하여 합병할 수 없습니다."
                    log_error(MODULE_NAME, f"합병 실패: {msg}", ValueError(msg))
                    res = render_page(new_items, page_num)
                    return [new_items, history, redo, msg, gr.update(interactive=bool(history)), gr.update(interactive=bool(redo))] + res

                new_history = history + [clone_items(new_items)]
                new_redo = []

                first_item = selected_items[0]
                for target_it in selected_items:
                    target_it["deleted"] = True
                    target_it["selected"] = False

                final_combined_text = " ".join([t.strip() for t in combined_texts if t.strip()]).strip()
                base_dir = os.path.dirname(first_item["wav"])
                base_name = os.path.splitext(first_item["wav_filename"])[0]

                merged_item = {
                    "wav": os.path.join(base_dir, f"{base_name}_merged.wav"),
                    "txt": os.path.join(base_dir, f"{base_name}_merged.txt"),
                    "content": final_combined_text,
                    "original_content": "",
                    "deleted": False,
                    "folder_name": first_item["folder_name"],
                    "wav_filename": f"{base_name}_merged.wav",
                    "speaker_num": first_item.get("speaker_num", "1"),
                    "is_mixed": first_item.get("is_mixed", False),
                    "selected": False,
                    "audio_segment": cur_audio,
                    "is_new": True
                }

                first_orig_idx = new_items.index(first_item)
                new_items.insert(first_orig_idx + 1, merged_item)
                
                surviving = [it for it in new_items if not it["deleted"]]
                target_page = (surviving.index(merged_item) // ITEMS_PER_PAGE) + 1 if merged_item in surviving else 1
                res = render_page(new_items, target_page)
                return [new_items, new_history, new_redo, f"선택된 {len(selected_items)}개 세그먼트 합병 완료 (저장 시 적용)", gr.update(interactive=True), gr.update(interactive=False)] + res
            except Exception as e:
                log_error(MODULE_NAME, "세그먼트 합병 중 예외 발생", e)
                res = render_page(new_items, page_num)
                return [new_items, history, redo, f"합병 오류: {e}", gr.update(interactive=bool(history)), gr.update(interactive=bool(redo))] + res

        def handle_undo(items, history, redo, page_num, *args):
            half = len(text_components)
            current_texts = args[:half]
            current_checkboxes = args[half:]
            if not history:
                msg = "더 이상 되돌릴 수 없습니다."
                log_error(MODULE_NAME, f"되돌리기 실패: {msg}", ValueError(msg))
                res = render_page(items, page_num)
                return [items, history, redo, msg, gr.update(interactive=False), gr.update(interactive=bool(redo))] + res
            
            current_state = sync_current_data(items, page_num, current_texts, current_checkboxes)
            prev_state = history[-1]
            new_history = history[:-1]
            new_redo = redo + [clone_items(current_state)]
            
            surviving_prev = [it for it in prev_state if not it["deleted"]]
            total_pages = int(np.ceil(len(surviving_prev) / ITEMS_PER_PAGE)) or 1
            target_page = min(page_num, total_pages)
            
            res = render_page(prev_state, target_page)
            return [prev_state, new_history, new_redo, "[안내] 이전 상태로 되돌렸습니다.", gr.update(interactive=bool(new_history)), gr.update(interactive=bool(new_redo))] + res

        def handle_redo(items, history, redo, page_num, *args):
            half = len(text_components)
            current_texts = args[:half]
            current_checkboxes = args[half:]
            if not redo:
                msg = "더 이상 앞으로 돌릴 수 없습니다."
                log_error(MODULE_NAME, f"다시 실행 실패: {msg}", ValueError(msg))
                res = render_page(items, page_num)
                return [items, history, redo, msg, gr.update(interactive=bool(history)), gr.update(interactive=False)] + res
            
            current_state = sync_current_data(items, page_num, current_texts, current_checkboxes)
            next_state = redo[-1]
            new_redo = redo[:-1]
            new_history = history + [clone_items(current_state)]
            
            surviving_next = [it for it in next_state if not it["deleted"]]
            total_pages = int(np.ceil(len(surviving_next) / ITEMS_PER_PAGE)) or 1
            target_page = min(page_num, total_pages)
            
            res = render_page(next_state, target_page)
            return [next_state, new_history, new_redo, "[안내] 작업을 앞으로 돌렸습니다.", gr.update(interactive=bool(new_history)), gr.update(interactive=bool(new_redo))] + res

        def save_all_changes(items, history, redo, page_num, *args):
            half = len(text_components)
            current_texts = args[:half]
            current_checkboxes = args[half:]

            if not items:
                msg = "저장할 데이터가 없습니다."
                log_error(MODULE_NAME, f"저장 실패: {msg}", ValueError(msg))
                empty_pagination = (gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False))
                return [items, history, redo, msg, gr.update(interactive=False), gr.update(interactive=False)] + list(empty_pagination) + [gr.update(visible=False), gr.update(value=None), gr.update(value=""), False, "", ""] * ITEMS_PER_PAGE
            try:
                current_state = sync_current_data(items, page_num, current_texts, current_checkboxes)
                surviving_items = []
                
                for item in current_state:
                    if item["deleted"]:
                        if os.path.exists(item["wav"]): 
                            try: os.remove(item["wav"])
                            except Exception as ex: log_error(MODULE_NAME, f"삭제된 WAV 파일 제거 실패: {item['wav']}", ex)
                        if os.path.exists(item["txt"]): 
                            try: os.remove(item["txt"])
                            except Exception as ex: log_error(MODULE_NAME, f"삭제된 TXT 파일 제거 실패: {item['txt']}", ex)
                    else:
                        if item.get("audio_segment") is not None:
                            item["audio_segment"].export(item["wav"], format="wav")
                            item["audio_segment"] = None

                        with open(item["txt"], "w", encoding="utf-8") as tf:
                            tf.write(item["content"])
                        item["original_content"] = item["content"]
                        item["is_new"] = False
                        surviving_items.append(item)
                        
                res = render_page(surviving_items, page_num)
                return [surviving_items, [], [], "[성공] 저장 버튼을 통해 변경사항이 디스크에 반영되었습니다.", gr.update(interactive=False), gr.update(interactive=False)] + res
            except Exception as e:
                log_error(MODULE_NAME, "변경사항 저장 중 예외 발생", e)
                res = render_page(items, page_num)
                return [items, history, redo, f"[오류] 저장 실패: {e}", gr.update(interactive=bool(history)), gr.update(interactive=bool(redo))] + res

        def register_algo_action(speaker_name, algo_json_val, items, history, redo, page_num, *args):
            speaker_name = speaker_name.strip() if speaker_name else ""
            
            selected_json = ""
            if algo_json_val and algo_json_val != "-- 선택 안함 --":
                selected_json = algo_json_val

            if speaker_name and selected_json:
                msg = "[오류] 화자 이름 직접 입력과 알고리즘 JSON 선택 중 하나만 선택해주세요."
                log_error(MODULE_NAME, f"알고리즘 등록 실패: {msg}", ValueError(msg))
                return msg

            target_name = speaker_name if speaker_name else os.path.splitext(selected_json)[0]
            if not target_name:
                msg = "[오류] 등록할 화자 이름을 입력하거나 알고리즘 JSON을 선택해주세요."
                log_error(MODULE_NAME, f"알고리즘 등록 실패: {msg}", ValueError(msg))
                return msg

            if not items:
                msg = "[오류] 로드된 데이터가 없습니다."
                log_error(MODULE_NAME, f"알고리즘 등록 실패: {msg}", ValueError(msg))
                return msg
            
            # 1. 먼저 현재 상태를 동기화하여 검사 준비
            half = len(text_components)
            current_texts = args[:half]
            current_checkboxes = args[half:]
            current_state = sync_current_data(items, page_num, current_texts, current_checkboxes)
            
            # 2. 살아남은(deleted가 아닌) 세그먼트들의 화자 종류(speaker_num) 추출
            surviving_items = [it for it in current_state if not it["deleted"]]
            if not surviving_items:
                msg = "[오류] 등록할 유효한 세그먼트가 존재하지 않습니다."
                log_error(MODULE_NAME, f"알고리즘 등록 실패: {msg}", ValueError(msg))
                return msg

            speakers_found = set()
            for it in surviving_items:
                # 파일명 재검사 혹은 item에 저장된 speaker_num 확인
                spk, _ = extract_speaker_from_filename(it["wav_filename"])
                if not spk:
                    spk = it.get("speaker_num", "1")
                speakers_found.add(spk)

            # 3. 다중 화자 검증 (남은 화자가 2개 이상이거나 기존 폴더가 다중 화자인 경우 차단)
            # 만약 다중 화자 상태가 남아있다면 등록 불가능하도록 제한
            folder_path = os.path.dirname(surviving_items[0]["txt"]) if surviving_items else ""
            _, folder_is_mixed = check_folder_speaker_type(folder_path)

            if len(speakers_found) > 1 or folder_is_mixed:
                # 단일 화자만 남았는지 체크 (사용자가 세그먼트 삭제 등을 통해 한 화자만 남겼는지 확인)
                # 만약 surviving_items 전체의 화자 번호가 모두 동일하다면 다중화자 여부 해제 가능 판단
                if len(speakers_found) > 1:
                    msg = "[경고] 다중 화자 세그먼트가 감지되었습니다. 단일 화자 데이터로 정제한 후에만 등록할 수 있습니다."
                    log_error(MODULE_NAME, "알고리즘 등록 실패 (다중 화자 잔존)", ValueError(msg))
                    return msg

            save_result = save_all_changes(items, history, redo, page_num, *args)
            new_items = save_result[0]
            save_msg = save_result[3]
            
            if "[오류]" in save_msg:
                log_error(MODULE_NAME, f"알고리즘 등록 전 저장 실패: {save_msg}", ValueError(save_msg))
                return f"등록 실패: {save_msg}"
            try:
                first_txt_path = new_items[0]["txt"] if new_items else None
                if not first_txt_path:
                    msg = "[오류] 폴더 경로를 찾을 수 없습니다."
                    log_error(MODULE_NAME, f"알고리즘 등록 실패: {msg}", FileNotFoundError(msg))
                    return msg
                success = ah.register_dataset_from_refined_folder(target_name, os.path.dirname(first_txt_path))
                if success: return f"[🎉 성공] '{target_name}' 화자 알고리즘 등록 완료!"
                msg = "[경고] 기존 알고리즘 화자와 일치하지 않습니다."
                log_error(MODULE_NAME, f"알고리즘 등록 실패: {target_name} - {msg}")
                return msg
            except Exception as e:
                log_error(MODULE_NAME, f"화자 알고리즘 등록 중 예외 발생: {target_name}", e)
                return f"[오류] 예외 발생: {e}"

        close_ui_btn.click(
            fn=None, 
            inputs=[], 
            outputs=[], 
            js="() => { window.close(); }"
        ).then(
            fn=lambda: demo.close(), 
            outputs=[]
        )

        refresh_btn.click(fn=handle_refresh_click, outputs=[single_dropdown, basic_dropdown, algo_json_dropdown])
        
        load_outputs = [state_items, state_history, state_redo, state_page, info_box, page_info_md, page_info_md_bottom, prev_btn, next_btn, prev_btn_bottom, next_btn_bottom]
        for i in range(ITEMS_PER_PAGE): 
            load_outputs.extend([row_components[i], audio_components[i], text_components[i], select_checkbox_components[i], txt_path_components[i], wav_path_components[i]])
            
        load_single_btn.click(fn=lambda folder: load_folder_data(folder, is_single_mode=True), inputs=[single_dropdown], outputs=load_outputs)
        load_basic_btn.click(fn=lambda folder: load_folder_data(folder, is_single_mode=False), inputs=[basic_dropdown], outputs=load_outputs)

        pagination_outputs = [state_items, state_page, page_info_md, page_info_md_bottom, prev_btn, next_btn, prev_btn_bottom, next_btn_bottom]
        for i in range(ITEMS_PER_PAGE): 
            pagination_outputs.extend([row_components[i], audio_components[i], text_components[i], select_checkbox_components[i], txt_path_components[i], wav_path_components[i]])

        def on_prev_page(items, p, *args):
            return change_page_wrapper(items, p, -1, *args)

        def on_next_page(items, p, *args):
            return change_page_wrapper(items, p, 1, *args)

        prev_btn.click(fn=on_prev_page, inputs=[state_items, state_page] + text_components + select_checkbox_components, outputs=pagination_outputs)
        prev_btn_bottom.click(fn=on_prev_page, inputs=[state_items, state_page] + text_components + select_checkbox_components, outputs=pagination_outputs)
        next_btn.click(fn=on_next_page, inputs=[state_items, state_page] + text_components + select_checkbox_components, outputs=pagination_outputs)
        next_btn_bottom.click(fn=on_next_page, inputs=[state_items, state_page] + text_components + select_checkbox_components, outputs=pagination_outputs)

        for t_comp in text_components:
            for trigger_event in [t_comp.blur, t_comp.submit]:
                trigger_event(fn=handle_text_commit, inputs=[state_items, state_history, state_redo, state_page] + text_components + select_checkbox_components, outputs=[state_items, state_history, state_redo, undo_global_btn, redo_global_btn])

        undo_redo_outputs = [state_items, state_history, state_redo, info_box, undo_global_btn, redo_global_btn, state_page, page_info_md, page_info_md_bottom, prev_btn, next_btn, prev_btn_bottom, next_btn_bottom]
        for i in range(ITEMS_PER_PAGE): 
            undo_redo_outputs.extend([row_components[i], audio_components[i], text_components[i], select_checkbox_components[i], txt_path_components[i], wav_path_components[i]])

        undo_global_btn.click(fn=handle_undo, inputs=[state_items, state_history, state_redo, state_page] + text_components + select_checkbox_components, outputs=undo_redo_outputs)
        redo_global_btn.click(fn=handle_redo, inputs=[state_items, state_history, state_redo, state_page] + text_components + select_checkbox_components, outputs=undo_redo_outputs)

        common_outputs = [state_items, state_history, state_redo, info_box, undo_global_btn, redo_global_btn, state_page, page_info_md, page_info_md_bottom, prev_btn, next_btn, prev_btn_bottom, next_btn_bottom]
        for i in range(ITEMS_PER_PAGE): 
            common_outputs.extend([row_components[i], audio_components[i], text_components[i], select_checkbox_components[i], txt_path_components[i], wav_path_components[i]])

        merge_selected_btn.click(fn=handle_selected_merge, inputs=[state_items, state_history, state_redo, state_page] + text_components + select_checkbox_components, outputs=common_outputs)

        def make_delete_handler(idx):
            def delete_handler(items, history, redo, p_num, *args):
                return handle_single_delete(items, history, redo, p_num, idx, *args)
            return delete_handler

        def make_split_handler(idx):
            def split_handler(items, history, redo, p_num, playback: gr.Audio, *args):
                return handle_single_split(items, history, redo, p_num, idx, playback, *args)
            return split_handler

        for i in range(ITEMS_PER_PAGE):
            delete_btn_components[i].click(
                fn=make_delete_handler(i), 
                inputs=[state_items, state_history, state_redo, state_page] + text_components + select_checkbox_components, 
                outputs=common_outputs
            )
            split_btn_components[i].click(
                fn=make_split_handler(i), 
                inputs=[state_items, state_history, state_redo, state_page, audio_components[i]] + text_components + select_checkbox_components, 
                outputs=common_outputs
            )

        save_outputs = [state_items, state_history, state_redo, info_box, undo_global_btn, redo_global_btn, state_page, page_info_md, page_info_md_bottom, prev_btn, next_btn, prev_btn_bottom, next_btn_bottom]
        for i in range(ITEMS_PER_PAGE): 
            save_outputs.extend([row_components[i], audio_components[i], text_components[i], select_checkbox_components[i], txt_path_components[i], wav_path_components[i]])

        save_all_btn.click(fn=save_all_changes, inputs=[state_items, state_history, state_redo, state_page] + text_components + select_checkbox_components, outputs=save_outputs)

        register_algo_inputs = [speaker_name_input, algo_json_dropdown, state_items, state_history, state_redo, state_page] + text_components + select_checkbox_components
        register_algo_btn.click(fn=register_algo_action, inputs=register_algo_inputs, outputs=[info_box])

    demo.launch(inbrowser=True, server_name="127.0.0.1", server_port=None, prevent_thread_lock=True)
    
    input("[*] Web UI가 실행되었습니다. 창을 닫거나 엔터를 누르면 종료됩니다.\n")
    try:
        demo.close()
    except:
        pass
        
    log_info(MODULE_NAME, "Web UI 서버가 정상적으로 종료되었습니다.")
import os
import re
import numpy as np
import gradio as gr
from pydub import AudioSegment
import algorithm_handler as ah
from error_logger import log_error, log_info

MODULE_NAME = "RefinementStudioUI"

SEGMENTS_BASE_DIR = "segments_base"
ITEMS_PER_PAGE = 10

# --- 오디오 분할 및 합병 기본 백엔드 함수 정의 (algorithm_handler에 없는 경우를 대비한 보완) ---
if not hasattr(ah, "split_audio_segment"):
    def _fallback_split_audio_segment(wav_path, txt_path):
        try:
            if not os.path.exists(wav_path):
                return False, "원본 WAV 파일이 존재하지 않습니다.", []
            audio = AudioSegment.from_wav(wav_path)
            duration_ms = len(audio)
            if duration_ms < 1000:
                return False, "오디오 길이가 너무 짧아 분할할 수 없습니다.", []
            mid_ms = duration_ms // 2
            part1 = audio[:mid_ms]
            part2 = audio[mid_ms:]
            
            base_dir = os.path.dirname(wav_path)
            base_name = os.path.splitext(os.path.basename(wav_path))[0]
            
            new_wav_1 = os.path.join(base_dir, f"{base_name}_part1.wav")
            new_txt_1 = os.path.join(base_dir, f"{base_name}_part1.txt")
            new_wav_2 = os.path.join(base_dir, f"{base_name}_part2.wav")
            new_txt_2 = os.path.join(base_dir, f"{base_name}_part2.txt")
            
            part1.export(new_wav_1, format="wav")
            part2.export(new_wav_2, format="wav")
            
            txt_content = ""
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    txt_content = f.read().strip()
            
            with open(new_txt_1, "w", encoding="utf-8") as f:
                f.write(txt_content)
            with open(new_txt_2, "w", encoding="utf-8") as f:
                f.write(txt_content)
                
            return True, "성공적으로 분할되었습니다.", [new_wav_1, new_wav_2]
        except Exception as e:
            log_error(MODULE_NAME, "오디오 분할 백엔드 예외 발생", e, debug=True)
            return False, f"오디오 분할 중 오류 발생: {str(e)}", []
    ah.split_audio_segment = _fallback_split_audio_segment

if not hasattr(ah, "merge_audio_segments"):
    def _fallback_merge_audio_segments(wav_path1, wav_path2, txt_path1, txt_path2):
        try:
            if not os.path.exists(wav_path1) or not os.path.exists(wav_path2):
                return False, "합병할 대상 WAV 파일 중 일부가 존재하지 않습니다.", None
            audio1 = AudioSegment.from_wav(wav_path1)
            audio2 = AudioSegment.from_wav(wav_path2)
            merged_audio = audio1 + audio2
            
            base_dir = os.path.dirname(wav_path1)
            base_name1 = os.path.splitext(os.path.basename(wav_path1))[0]
            
            merged_wav_path = os.path.join(base_dir, f"{base_name1}_merged.wav")
            merged_txt_path = os.path.join(base_dir, f"{base_name1}_merged.txt")
            
            merged_audio.export(merged_wav_path, format="wav")
            
            text1, text2 = "", ""
            if os.path.exists(txt_path1):
                with open(txt_path1, "r", encoding="utf-8") as f:
                    text1 = f.read().strip()
            if os.path.exists(txt_path2):
                with open(txt_path2, "r", encoding="utf-8") as f:
                    text2 = f.read().strip()
            
            combined_text = f"{text1} {text2}".strip()
            with open(merged_txt_path, "w", encoding="utf-8") as f:
                f.write(combined_text)
                
            merged_info = {
                "wav": merged_wav_path,
                "txt": merged_txt_path,
                "content": combined_text
            }
            return True, "성공적으로 합병되었습니다.", merged_info
        except Exception as e:
            log_error(MODULE_NAME, "오디오 합병 백엔드 예외 발생", e, debug=True)
            return False, f"오디오 합병 중 오류 발생: {str(e)}", None
    ah.merge_audio_segments = _fallback_merge_audio_segments
# --------------------------------------------------------------------------

def get_single_speaker_folders():
    if not os.path.exists(SEGMENTS_BASE_DIR):
        return []
    folders = []
    try:
        for root, dirs, files in os.walk(SEGMENTS_BASE_DIR):
            for d in dirs:
                d_lower = d.lower()
                if d_lower.startswith("seg_single") or "single" in d_lower or "단일" in d or check_custom_single_format(d_lower)[0]:
                    rel_path = os.path.relpath(os.path.join(root, d), SEGMENTS_BASE_DIR)
                    folders.append(rel_path)
    except Exception as e:
        log_error(MODULE_NAME, "단일 화자 폴더 탐색 중 예외 발생", e, debug=True)
    return folders

def get_basic_segment_folders():
    if not os.path.exists(SEGMENTS_BASE_DIR):
        return []
    folders = []
    try:
        for root, dirs, files in os.walk(SEGMENTS_BASE_DIR):
            for d in dirs:
                if not d.startswith("seg_multi") and ("세그먼트" in d or "seg" in d.lower() or "basic" in d.lower()):
                    rel_path = os.path.relpath(os.path.join(root, d), SEGMENTS_BASE_DIR)
                    folders.append(rel_path)
    except Exception as e:
        log_error(MODULE_NAME, "기본 분석 폴더 탐색 중 예외 발생", e, debug=True)
    return folders

def clone_items(items):
    return [item.copy() for item in items]

def check_custom_single_format(filename_or_foldername):
    if not filename_or_foldername:
        return False, "1"
        
    cleaned = os.path.splitext(filename_or_foldername)[0].lower()
    parts = cleaned.split("_")
    
    if len(parts) >= 5 and parts[0] == "seg" and parts[1] == "sub":
        speaker_slot = parts[3]
        if speaker_slot == "a":
            speaker_num = "1"
            return True, speaker_num
            
    if len(parts) > 0 and parts[0] == "a":
        return True, "1"
        
    return False, "1"

def inspect_all_files(target_dir):
    if not os.path.exists(target_dir):
        return False, "존재하지 않는 폴더입니다."
    
    try:
        files = os.listdir(target_dir)
        wav_files = [f for f in files if f.lower().endswith('.wav')]
        
        if not wav_files:
            return False, "해당 폴더에 WAV 파일이 없습니다."
            
        issues = []
        for w_f in wav_files:
            base_name = os.path.splitext(w_f)[0]
            t_path = os.path.join(target_dir, f"{base_name}.txt")
            
            if not os.path.exists(t_path):
                issues.append(f"누락된 텍스트 파일: {base_name}.txt")
                
            lower_wf = w_f.lower()
            is_custom_single, _ = check_custom_single_format(lower_wf)
            
            pattern = r"^seg_sub_\d+_(speaker_\d+|화자_\d+)_\d+(?:\.\d+)?s-\d+(?:\.\d+)?s\.wav$"
            is_match_pattern = bool(re.match(pattern, lower_wf))
            
            is_processed_file = "_merged" in lower_wf or "_part1" in lower_wf or "_part2" in lower_wf
            
            if not is_custom_single and not is_match_pattern and not is_processed_file:
                issues.append(f"명명 규칙 및 구조 위치 불일치 파일: {w_f}")
                
        if issues:
            return False, f"전체 검사 실패 (총 {len(issues)}개 문제 발견):\n" + "\n".join(issues[:3])
        
        return True, ""
    except Exception as e:
        log_error(MODULE_NAME, "전체 파일 검사 중 예외 발생", e, debug=True)
        return False, f"검사 중 오류 발생: {e}"

def run_data_refinement_webui():
    log_info(MODULE_NAME, "Web UI 스튜디오를 실행합니다...")
    
    single_folders = get_single_speaker_folders()
    basic_folders = get_basic_segment_folders()
    
    default_single = single_folders[0] if single_folders else None
    default_basic = basic_folders[0] if basic_folders else None

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
        width: 28px !important;
        height: 28px !important;
        min-width: 28px !important;
        min-height: 28px !important;
        max-width: 28px !important;
        max-height: 28px !important;

        padding: 0 !important;
        margin: 0 auto !important;

        background-color: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 6px !important;

        display: flex !important;
        align-items: center !important;
        justify-content: center !important;

        box-shadow: none !important;
    }

    .segment-checkbox input[type="checkbox"] {
        width: 18px !important;
        height: 18px !important;
        min-width: 18px !important;
        min-height: 18px !important;
        margin: 0 auto !important;
        cursor: pointer !important;
        accent-color: #ff7f50 !important;
        pointer-events: auto !important;
    }
    """

    with gr.Blocks(title="데이터 정제 스튜디오", css=custom_css) as demo:
        state_items = gr.State([])
        state_history = gr.State([])
        state_redo = gr.State([])
        state_page = gr.State(1)
        state_checkboxes = gr.State([])

        with gr.Row():
            gr.Markdown("## 🛠️ 데이터 정제 및 화자 알고리즘 등록 스튜디오")
            close_ui_btn = gr.Button("🚪 Web UI 종료 (메뉴로 돌아가기)", variant="stop")

        gr.Markdown("💡 **안내:** 단일 화자 세그먼트 또는 기본 분석 세그먼트 중 작업할 폴더를 선택하여 불러오세요.")

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

        with gr.Row(variant="panel"):
            with gr.Column(scale=2):
                speaker_name_input = gr.Textbox(label="등록할 화자 이름", placeholder="예: 화자_a", lines=1)
            with gr.Column(scale=1):
                save_all_btn = gr.Button("💾 저장", variant="secondary")
            with gr.Column(scale=1):
                register_algo_btn = gr.Button("알고리즘 등록/업데이트", variant="primary")

        info_box = gr.Textbox(label="현재 진행 상황", value="작업할 폴더를 선택하고 '불러오기' 버튼을 누르세요.", interactive=False)

        with gr.Row(elem_classes=["pagination-toolbar"]):
            prev_btn = gr.Button("⬅️ 이전 페이지 ")

            page_info_md = gr.Markdown(
                "### 페이지: 0 / 0 (총 0개 항목)",
                elem_classes=["page-info"]
            )

            merge_selected_btn = gr.Button(
                "🔗 선택 합병",
                variant="secondary",
                scale=0,
                min_width=130
            )

            undo_global_btn = gr.Button(
                "↩️",
                elem_classes=["icon-btn"],
                interactive=False,
                scale=0,
                min_width=55
            )

            redo_global_btn = gr.Button(
                "🔁",
                elem_classes=["icon-btn"],
                interactive=False,
                scale=0,
                min_width=55
            )

            next_btn = gr.Button("다음 페이지 ➡️ ")

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
                    a_comp = gr.Audio(label=f"세그먼트 {i+1}", type="filepath", interactive=False)
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
            with gr.Column(scale=1):
                prev_btn_bottom = gr.Button("⬅️ 이전 페이지 ")
            with gr.Column(scale=1):
                page_info_md_bottom = gr.Markdown("### 페이지: 0 / 0", elem_classes=["page-info-bottom"])
            with gr.Column(scale=1):
                next_btn_bottom = gr.Button("다음 페이지 ➡️ ")

        close_ui_btn.click(
            fn=None,
            inputs=[],
            outputs=[],
            js="() => { window.close(); }"
        ).then(
            fn=lambda: demo.close(),
            outputs=[]
        )

        def load_folder_data(current_items, folder_name):
            if not folder_name:
                return [[], [], [], 1, [], "폴더가 선택되지 않았습니다.", "### 페이지: 0 / 0 (총 0개 항목)", "### 페이지: 0 / 0", gr.update(interactive=False), gr.update(interactive=False)] + [gr.update(visible=False), gr.update(value=None, label="세그먼트 1"), gr.update(value="", label="세그먼트 1"), False, "", ""] * ITEMS_PER_PAGE
            
            if current_items:
                for it in current_items:
                    if it.get("is_temp", False) and not it.get("saved", False):
                        temp_w = it.get("wav")
                        temp_t = it.get("txt")
                        if temp_w and os.path.exists(temp_w):
                            try: os.remove(temp_w)
                            except: pass
                        if temp_t and os.path.exists(temp_t):
                            try: os.remove(temp_t)
                            except: pass

            target_dir = os.path.join(SEGMENTS_BASE_DIR, folder_name)
            
            is_valid, inspect_msg = inspect_all_files(target_dir)
            if not is_valid:
                log_error(MODULE_NAME, f"전체 파일 검사 경고: {inspect_msg}")

            if not os.path.exists(target_dir):
                return [[], [], [], 1, [], "존재하지 않는 폴더입니다.", "### 페이지: 0 / 0 (총 0개 항목)", "### 페이지: 0 / 0", gr.update(interactive=False), gr.update(interactive=False)] + [gr.update(visible=False), gr.update(value=None, label="세그먼트 1"), gr.update(value="", label="세그먼트 1"), False, "", ""] * ITEMS_PER_PAGE
                
            try:
                wav_files = sorted([f for f in os.listdir(target_dir) if f.lower().endswith('.wav')])
            except Exception as e:
                log_error(MODULE_NAME, f"디렉토리 읽기 오류 ({target_dir})", e, debug=True)
                return [[], [], [], 1, [], f"디렉토리 읽기 오류: {e}", "### 페이지: 0 / 0 (총 0개 항목)", "### 페이지: 0 / 0", gr.update(interactive=False), gr.update(interactive=False)] + [gr.update(visible=False), gr.update(value=None, label="세그먼트 1"), gr.update(value="", label="세그먼트 1"), False, "", ""] * ITEMS_PER_PAGE

            if not wav_files:
                return [[], [], [], 1, [], "해당 폴더에 WAV 파일이 없습니다.", "### 페이지: 0 / 0 (총 0개 항목)", "### 페이지: 0 / 0", gr.update(interactive=False), gr.update(interactive=False)] + [gr.update(visible=False), gr.update(value=None, label="세그먼트 1"), gr.update(value="", label="세그먼트 1"), False, "", ""] * ITEMS_PER_PAGE
                
            items = []
            for w_f in wav_files:
                base_name = os.path.splitext(w_f)[0]
                w_path = os.path.join(target_dir, w_f)
                t_path = os.path.join(target_dir, f"{base_name}.txt")
                
                t_content = ""
                if os.path.exists(t_path):
                    try:
                        with open(t_path, "r", encoding="utf-8") as tf:
                            t_content = tf.read().strip()
                    except Exception as e:
                        log_error(MODULE_NAME, f"텍스트 파일 읽기 실패 ({t_path})", e, debug=True)
                        t_content = ""
                
                lower_wf = w_f.lower()
                pattern = r"^seg_sub_\d+_(speaker_\d+|화자_\d+)_\d+(?:\.\d+)?s-\d+(?:\.\d+)?s\.wav$"
                match = re.match(pattern, lower_wf)
                
                speaker_num = "1"
                is_mixed = True
                
                is_processed_file = "_merged" in lower_wf or "_part1" in lower_wf or "_part2" in lower_wf
                
                if match:
                    speaker_group = match.group(1)
                    num_match = re.search(r"\d+", speaker_group)
                    if num_match:
                        speaker_num = num_match.group(0)
                    
                    if "speaker_0" in speaker_group or "화자_0" in speaker_group:
                        is_mixed = False
                    else:
                        is_mixed = True
                elif is_processed_file:
                    is_mixed = False
                    speaker_num = "1"
                else:
                    is_mixed = True

                is_folder_custom_single, _ = check_custom_single_format(folder_name)
                is_file_custom_single, _ = check_custom_single_format(lower_wf)

                if is_folder_custom_single or is_file_custom_single or is_processed_file:
                    is_mixed = False

                items.append({
                    "wav": w_path, 
                    "txt": t_path, 
                    "content": t_content, 
                    "original_content": t_content, 
                    "deleted": False, 
                    "folder_name": folder_name, 
                    "wav_filename": w_f,
                    "speaker_num": speaker_num,
                    "is_mixed": is_mixed,
                    "is_temp": False,
                    "saved": True,
                    "selected": False,
                    "action_type": None, 
                    "virtual_sources": [] 
                })
            
            total_pages = int(np.ceil(len(items) / ITEMS_PER_PAGE))
            
            if inspect_msg:
                status_msg = f"총 {len(items)}개의 세그먼트 로드 완료 (총 {total_pages}페이지). [{inspect_msg}]"
            else:
                status_msg = f"총 {len(items)}개의 세그먼트 로드 완료 (총 {total_pages}페이지)."
            
            page_items = items[:ITEMS_PER_PAGE]
            updates = []
            for i in range(ITEMS_PER_PAGE):
                if i < len(page_items):
                    item = page_items[i]
                    seg_idx = i + 1
                    label_str = f"세그먼트{seg_idx}-화자{item['speaker_num']}" if item["is_mixed"] else f"세그먼트 {seg_idx}"
                    updates.extend([
                        gr.update(visible=not item["deleted"]), 
                        gr.update(value=item["wav"], label=label_str), 
                        gr.update(value=item["content"], label=label_str), 
                        item.get("selected", False),
                        item["txt"], 
                        item["wav"]
                    ])
                else:
                    updates.extend([gr.update(visible=False), gr.update(value=None, label=f"세그먼트 {i+1}"), gr.update(value="", label=f"세그먼트 {i+1}"), False, "", ""])
            
            return [items, [], [], 1, [], status_msg, f"### 페이지: 1 / {total_pages} (총 {len(items)}개 항목)", f"### 페이지: 1 / {total_pages}", gr.update(interactive=False), gr.update(interactive=False)] + updates

        def render_page(items, page_num):
            if not items:
                return [1, [], "### 페이지: 0 / 0 (총 0개 항목)", "### 페이지: 0 / 0"] + [gr.update(visible=False), gr.update(value=None), gr.update(value=""), False, "", ""] * ITEMS_PER_PAGE
            
            total_pages = int(np.ceil(len(items) / ITEMS_PER_PAGE))
            if page_num < 1: page_num = 1
            if page_num > total_pages: page_num = total_pages
            
            start_idx = (page_num - 1) * ITEMS_PER_PAGE
            page_items = items[start_idx:start_idx + ITEMS_PER_PAGE]
            
            updates = []
            for i in range(ITEMS_PER_PAGE):
                if i < len(page_items):
                    item = page_items[i]
                    global_idx = start_idx + i + 1
                    label_str = f"세그먼트{global_idx}-화자{item.get('speaker_num', '1')}" if item.get("is_mixed", False) else f"세그먼트 {global_idx}"
                    updates.extend([
                        gr.update(visible=not item["deleted"]), 
                        gr.update(value=item["wav"], label=label_str), 
                        gr.update(value=item["content"], label=label_str), 
                        item.get("selected", False),
                        item["txt"], 
                        item["wav"]
                    ])
                else:
                    updates.extend([gr.update(visible=False), gr.update(value=None, label=f"세그먼트 {i+1}"), gr.update(value="", label=f"세그먼트 {i+1}"), False, "", ""])
            
            return [page_num, [], f"### 페이지: {page_num} / {total_pages} (총 {len(items)}개 항목)", f"### 페이지: {page_num} / {total_pages}"] + updates

        def change_page(items, current_page, direction, *current_checkboxes):
            updated = clone_items(items)
            start_idx = (current_page - 1) * ITEMS_PER_PAGE
            for i, chk in enumerate(current_checkboxes):
                g_idx = start_idx + i
                if g_idx < len(updated):
                    updated[g_idx]["selected"] = bool(chk)

            total_pages = int(np.ceil(len(updated) / ITEMS_PER_PAGE)) if updated else 1
            new_page = max(1, min(current_page + direction, total_pages))
            
            res = render_page(updated, new_page)
            return [updated] + res

        def sync_current_data(items, page_num, current_texts, current_checkboxes):
            updated = clone_items(items)
            start_idx = (page_num - 1) * ITEMS_PER_PAGE
            for i in range(ITEMS_PER_PAGE):
                global_idx = start_idx + i
                if global_idx < len(updated):
                    if i < len(current_texts):
                        updated[global_idx]["content"] = current_texts[i]
                    if i < len(current_checkboxes):
                        updated[global_idx]["selected"] = bool(current_checkboxes[i])
            return updated

        def find_diff_page(old_items, new_items):
            for idx, (o, n) in enumerate(zip(old_items, new_items)):
                if o["content"] != n["content"] or o["deleted"] != n["deleted"]:
                    return (idx // ITEMS_PER_PAGE) + 1
            return None

        def handle_text_commit(items, history, redo, page_num, *args):
            half = len(text_components)
            current_texts = args[:half]
            current_checkboxes = args[half:]
            
            if not items:
                return [items, history, redo, gr.update(interactive=bool(history)), gr.update(interactive=bool(redo))]
            current_state = sync_current_data(items, page_num, current_texts, current_checkboxes)
            has_changed = False
            for old_it, new_it in zip(items, current_state):
                if old_it["content"] != new_it["content"] or old_it["deleted"] != new_it["deleted"] or old_it.get("selected") != new_it.get("selected"):
                    has_changed = True
                    break
            if not has_changed:
                return [items, history, redo, gr.update(interactive=bool(history)), gr.update(interactive=bool(redo))]
            new_history = history + [clone_items(items)]
            new_redo = [] 
            return [current_state, new_history, new_redo, gr.update(interactive=True), gr.update(interactive=False)]

        def handle_single_delete(items, history, redo, page_num, item_index, *args):
            half = len(text_components)
            current_texts = args[:half]
            current_checkboxes = args[half:]

            total_pages_fallback = int(np.ceil(len(items) / ITEMS_PER_PAGE)) if items else 1
            if not items:
                return [items, history, redo, "항목이 없습니다.", gr.update(interactive=bool(history)), gr.update(interactive=bool(redo)), page_num, [], f"### 페이지: {page_num} / {total_pages_fallback} (총 0개 항목)", f"### 페이지: {page_num} / {total_pages_fallback}"] + [gr.update(visible=False), gr.update(value=None), gr.update(value=""), False, "", ""] * ITEMS_PER_PAGE
            
            new_items = sync_current_data(items, page_num, current_texts, current_checkboxes)
            start_idx = (page_num - 1) * ITEMS_PER_PAGE
            global_idx = start_idx + item_index
            if global_idx < len(new_items):
                new_items[global_idx]["deleted"] = True
                new_items[global_idx]["selected"] = False
            new_history = history + [clone_items(items)]
            new_redo = []
            
            total_pages = int(np.ceil(len(new_items) / ITEMS_PER_PAGE))
            page_items = new_items[start_idx:start_idx+ITEMS_PER_PAGE]
            updates = []
            for i in range(ITEMS_PER_PAGE):
                if i < len(page_items):
                    item = page_items[i]
                    g_idx = start_idx + i + 1
                    label_str = f"세그먼트{g_idx}-화자{item.get('speaker_num', '1')}" if item.get("is_mixed", False) else f"세그먼트 {g_idx}"
                    updates.extend([
                        gr.update(visible=not item["deleted"]), 
                        gr.update(value=item["wav"], label=label_str), 
                        gr.update(value=item["content"], label=label_str), 
                        item.get("selected", False),
                        item["txt"], 
                        item["wav"]
                    ])
                else:
                    updates.extend([gr.update(visible=False), gr.update(value=None, label=f"세그먼트 {i+1}"), gr.update(value="", label=f"세그먼트 {i+1}"), False, "", ""])
            return [new_items, new_history, new_redo, f"세그먼트 #{global_idx+1} 삭제 대기됨 (저장 버튼 시 반영)", gr.update(interactive=True), gr.update(interactive=False), page_num, [], f"### 페이지: {page_num} / {total_pages} (총 {len(new_items)}개 항목)", f"### 페이지: {page_num} / {total_pages}"] + updates

        def handle_single_split(items, history, redo, page_num, item_index, *args):
            half = len(text_components)
            current_texts = args[:half]
            current_checkboxes = args[half:]

            total_pages_fallback = int(np.ceil(len(items) / ITEMS_PER_PAGE)) if items else 1
            if not items:
                return [items, history, redo, "항목이 없습니다.", gr.update(interactive=bool(history)), gr.update(interactive=bool(redo)), page_num, [], f"### 페이지: {page_num} / {total_pages_fallback} (총 0개 항목)", f"### 페이지: {page_num} / {total_pages_fallback}"] + [gr.update(visible=False), gr.update(value=None), gr.update(value=""), False, "", ""] * ITEMS_PER_PAGE
            
            new_items = sync_current_data(items, page_num, current_texts, current_checkboxes)
            start_idx = (page_num - 1) * ITEMS_PER_PAGE
            global_idx = start_idx + item_index
            
            if global_idx >= len(new_items):
                total_pages = int(np.ceil(len(new_items) / ITEMS_PER_PAGE))
                return [new_items, history, redo, "잘못된 항목 인덱스입니다.", gr.update(interactive=bool(history)), gr.update(interactive=bool(redo)), page_num, [], f"### 페이지: {page_num} / {total_pages} (총 {len(new_items)}개 항목)", f"### 페이지: {page_num} / {total_pages}"] + [gr.update(visible=False), gr.update(value=None), gr.update(value=""), False, "", ""] * ITEMS_PER_PAGE
            
            target_item = new_items[global_idx]
            if target_item["deleted"]:
                total_pages = int(np.ceil(len(new_items) / ITEMS_PER_PAGE))
                return [new_items, history, redo, "삭제된 세그먼트는 분할할 수 없습니다.", gr.update(interactive=bool(history)), gr.update(interactive=bool(redo)), page_num, [], f"### 페이지: {page_num} / {total_pages} (총 {len(new_items)}개 항목)", f"### 페이지: {page_num} / {total_pages}"] + [gr.update(visible=False), gr.update(value=None), gr.update(value=""), False, "", ""] * ITEMS_PER_PAGE

            try:
                base_dir = os.path.dirname(target_item["wav"])
                base_name = os.path.splitext(target_item["wav_filename"])[0]
                
                temp_wav_1 = os.path.join(base_dir, f"{base_name}_part1.wav")
                temp_txt_1 = os.path.join(base_dir, f"{base_name}_part1.txt")
                temp_wav_2 = os.path.join(base_dir, f"{base_name}_part2.wav")
                temp_txt_2 = os.path.join(base_dir, f"{base_name}_part2.txt")
                
                speaker_num = target_item["speaker_num"]
                is_mixed = target_item["is_mixed"]
                folder_name = target_item["folder_name"]
                
                content = target_item["content"]
                
                ins1 = {
                    "wav": temp_wav_1,
                    "txt": temp_txt_1,
                    "content": content,
                    "original_content": content,
                    "deleted": False,
                    "folder_name": folder_name,
                    "wav_filename": f"{base_name}_part1.wav",
                    "speaker_num": speaker_num,
                    "is_mixed": is_mixed,
                    "is_temp": True,
                    "saved": False,
                    "selected": False,
                    "action_type": "split_source",
                    "virtual_sources": [target_item["wav"]]
                }
                ins2 = {
                    "wav": temp_wav_2,
                    "txt": temp_txt_2,
                    "content": content,
                    "original_content": content,
                    "deleted": False,
                    "folder_name": folder_name,
                    "wav_filename": f"{base_name}_part2.wav",
                    "speaker_num": speaker_num,
                    "is_mixed": is_mixed,
                    "is_temp": True,
                    "saved": False,
                    "selected": False,
                    "action_type": "split_source",
                    "virtual_sources": []
                }
                
                new_items[global_idx]["deleted"] = True 
                new_items[global_idx]["selected"] = False

                new_items.insert(global_idx + 1, ins2)
                new_items.insert(global_idx + 1, ins1)

                new_history = history + [clone_items(items)]
                new_redo = []
                
                total_pages = int(np.ceil(len(new_items) / ITEMS_PER_PAGE))
                page_items = new_items[start_idx:start_idx+ITEMS_PER_PAGE]
                updates = []
                for i in range(ITEMS_PER_PAGE):
                    if i < len(page_items):
                        item = page_items[i]
                        g_idx = start_idx + i + 1
                        label_str = f"세그먼트{g_idx}-화자{item.get('speaker_num', '1')}" if item.get("is_mixed", False) else f"세그먼트 {g_idx}"
                        updates.extend([
                            gr.update(visible=not item["deleted"]), 
                            gr.update(value=item["wav"], label=label_str), 
                            gr.update(value=item["content"], label=label_str), 
                            item.get("selected", False),
                            item["txt"], 
                            item["wav"]
                        ])
                    else:
                        updates.extend([gr.update(visible=False), gr.update(value=None, label=f"세그먼트 {i+1}"), gr.update(value="", label=f"세그먼트 {i+1}"), False, "", ""])

                return [new_items, new_history, new_redo, f"세그먼트 #{global_idx+1} 분할 완료", gr.update(interactive=True), gr.update(interactive=False), page_num, [], f"### 페이지: {page_num} / {total_pages} (총 {len(new_items)}개 항목)", f"### 페이지: {page_num} / {total_pages}"] + updates
            except Exception as e:
                log_error(MODULE_NAME, "세그먼트 분할 중 예외 발생", e, debug=True)
                total_pages = int(np.ceil(len(new_items) / ITEMS_PER_PAGE))
                return [new_items, history, redo, f"분할 중 오류 발생: {e}", gr.update(interactive=bool(history)), gr.update(interactive=bool(redo)), page_num, [], f"### 페이지: {page_num} / {total_pages} (총 {len(new_items)}개 항목)", f"### 페이지: {page_num} / {total_pages}"] + [gr.update(visible=False), gr.update(value=None), gr.update(value=""), False, "", ""] * ITEMS_PER_PAGE

        def handle_selected_merge(items, history, redo, page_num, *args):
            total_pages_fallback = int(np.ceil(len(items) / ITEMS_PER_PAGE)) if items else 1
            if not items:
                return [items, history, redo, "항목이 없습니다.", gr.update(interactive=False), gr.update(interactive=False), page_num, [], f"### 페이지: {page_num} / {total_pages_fallback} (총 0개 항목)", f"### 페이지: {page_num} / {total_pages_fallback}"] + [gr.update(visible=False), gr.update(value=None), gr.update(value=""), False, "", ""] * ITEMS_PER_PAGE
            
            half_len = len(text_components)
            current_texts = args[:half_len]
            checkbox_values = args[half_len:]

            new_items = sync_current_data(items, page_num, current_texts, checkbox_values)
            start_idx = (page_num - 1) * ITEMS_PER_PAGE

            selected_global_indices = []
            for g_idx, item in enumerate(new_items):
                if item.get("selected", False) and not item["deleted"]:
                    selected_global_indices.append(g_idx)

            if len(selected_global_indices) < 2:
                total_pages = int(np.ceil(len(new_items) / ITEMS_PER_PAGE))
                return [new_items, history, redo, "합병하려면 최소 2개 이상의 세그먼트를 체크박스로 선택해주세요.", gr.update(interactive=bool(history)), gr.update(interactive=bool(redo)), page_num, [], f"### 페이지: {page_num} / {total_pages} (총 {len(new_items)}개 항목)", f"### 페이지: {page_num} / {total_pages}"] + [gr.update(visible=False), gr.update(value=None), gr.update(value=""), False, "", ""] * ITEMS_PER_PAGE

            try:
                first_idx = selected_global_indices[0]
                item1 = new_items[first_idx]
                
                base_dir = os.path.dirname(item1["wav"])
                base_name1 = os.path.splitext(item1["wav_filename"])[0]
                merged_wav_path = os.path.join(base_dir, f"{base_name1}_merged.wav")
                merged_txt_path = os.path.join(base_dir, f"{base_name1}_merged.txt")
                
                combined_texts = []
                merged_sources = []
                for idx in selected_global_indices:
                    it = new_items[idx]
                    combined_texts.append(it["content"])
                    merged_sources.append(it["wav"])
                    if idx != first_idx:
                        new_items[idx]["deleted"] = True 
                        new_items[idx]["selected"] = False

                final_combined_text = " ".join([t for t in combined_texts if t]).strip()

                new_items[first_idx]["wav"] = merged_wav_path
                new_items[first_idx]["txt"] = merged_txt_path
                new_items[first_idx]["content"] = final_combined_text
                new_items[first_idx]["original_content"] = final_combined_text
                new_items[first_idx]["wav_filename"] = os.path.basename(merged_wav_path)
                new_items[first_idx]["is_temp"] = True
                new_items[first_idx]["saved"] = False
                new_items[first_idx]["selected"] = False
                new_items[first_idx]["action_type"] = "merge_source"
                new_items[first_idx]["virtual_sources"] = merged_sources
                new_items[first_idx]["is_mixed"] = False

                new_history = history + [clone_items(items)]
                new_redo = []
                
                target_page = (first_idx // ITEMS_PER_PAGE) + 1
                total_pages = int(np.ceil(len(new_items) / ITEMS_PER_PAGE))
                if target_page > total_pages: target_page = total_pages
                
                page_items = new_items[(target_page-1)*ITEMS_PER_PAGE : target_page*ITEMS_PER_PAGE]
                updates = []
                for i in range(ITEMS_PER_PAGE):
                    if i < len(page_items):
                        item = page_items[i]
                        g_idx = (target_page - 1) * ITEMS_PER_PAGE + i + 1
                        label_str = f"세그먼트{g_idx}-화자{item.get('speaker_num', '1')}" if item.get("is_mixed", False) else f"세그먼트 {g_idx}"
                        updates.extend([
                            gr.update(visible=not item["deleted"]), 
                            gr.update(value=item["wav"], label=label_str), 
                            gr.update(value=item["content"], label=label_str), 
                            item.get("selected", False),
                            item["txt"], 
                            item["wav"]
                        ])
                    else:
                        updates.extend([gr.update(visible=False), gr.update(value=None, label=f"세그먼트 {i+1}"), gr.update(value="", label=f"세그먼트 {i+1}"), False, "", ""])

                return [new_items, new_history, new_redo, f"선택된 {len(selected_global_indices)}개 세그먼트 합병 대기 완료 (저장 시 실제 파일 생성 및 확정)", gr.update(interactive=True), gr.update(interactive=False), target_page, [], f"### 페이지: {target_page} / {total_pages} (총 {len(new_items)}개 항목)", f"### 페이지: {target_page} / {total_pages}"] + updates
            except Exception as e:
                log_error(MODULE_NAME, "선택 세그먼트 합병 중 예외 발생", e, debug=True)
                total_pages = int(np.ceil(len(new_items) / ITEMS_PER_PAGE))
                return [new_items, history, redo, f"합병 중 오류 발생: {e}", gr.update(interactive=bool(history)), gr.update(interactive=bool(redo)), page_num, [], f"### 페이지: {page_num} / {total_pages} (총 {len(new_items)}개 항목)", f"### 페이지: {page_num} / {total_pages}"] + [gr.update(visible=False), gr.update(value=None), gr.update(value=""), False, "", ""] * ITEMS_PER_PAGE

        def handle_undo(items, history, redo, page_num, *args):
            half = len(text_components)
            current_texts = args[:half]
            current_checkboxes = args[half:]

            if not history:
                return [items, history, redo, "더 이상 되돌릴 수 없습니다.", gr.update(interactive=False), gr.update(interactive=bool(redo)), page_num, [], "### 페이지: 0 / 0", "### 페이지: 0 / 0"] + [gr.update(), gr.update(), gr.update(), False, "", ""] * ITEMS_PER_PAGE
            current_state = sync_current_data(items, page_num, current_texts, current_checkboxes)
            prev_state = history[-1]
            new_history = history[:-1]
            new_redo = redo + [clone_items(current_state)]
            target_page = find_diff_page(prev_state, current_state)
            if target_page is None: target_page = page_num
            total_pages = int(np.ceil(len(prev_state) / ITEMS_PER_PAGE)) if prev_state else 1
            if target_page > total_pages: target_page = total_pages
            if target_page < 1: target_page = 1
            start_idx = (target_page - 1) * ITEMS_PER_PAGE
            page_items = prev_state[start_idx:start_idx+ITEMS_PER_PAGE]
            updates = []
            for i in range(ITEMS_PER_PAGE):
                if i < len(page_items):
                    item = page_items[i]
                    g_idx = start_idx + i + 1
                    label_str = f"세그먼트{g_idx}-화자{item.get('speaker_num', '1')}" if item.get("is_mixed", False) else f"세그먼트 {g_idx}"
                    updates.extend([
                        gr.update(visible=not item["deleted"]), 
                        gr.update(value=item["wav"], label=label_str), 
                        gr.update(value=item["content"], label=label_str), 
                        item.get("selected", False),
                        item["txt"], 
                        item["wav"]
                    ])
                else:
                    updates.extend([gr.update(visible=False), gr.update(value=None, label=f"세그먼트 {i+1}"), gr.update(value="", label=f"세그먼트 {i+1}"), False, "", ""])
            return [prev_state, new_history, new_redo, "[안내] 이전 상태로 일괄 되돌렸습니다.", target_page, [], f"### 페이지: {target_page} / {total_pages} (총 {len(prev_state)}개 항목)", f"### 페이지: {target_page} / {total_pages}", gr.update(interactive=bool(new_history)), gr.update(interactive=bool(new_redo))] + updates

        def handle_redo(items, history, redo, page_num, *args):
            half = len(text_components)
            current_texts = args[:half]
            current_checkboxes = args[half:]

            if not redo:
                return [items, history, redo, "더 이상 앞으로 돌릴 수 없습니다.", gr.update(interactive=bool(history)), gr.update(interactive=False), page_num, [], "### 페이지: 0 / 0", "### 페이지: 0 / 0"] + [gr.update(), gr.update(), gr.update(), False, "", ""] * ITEMS_PER_PAGE
            current_state = sync_current_data(items, page_num, current_texts, current_checkboxes)
            next_state = redo[-1]
            new_redo = redo[:-1]
            new_history = history + [clone_items(current_state)]
            target_page = find_diff_page(current_state, next_state)
            if target_page is None: target_page = page_num
            total_pages = int(np.ceil(len(next_state) / ITEMS_PER_PAGE)) if next_state else 1
            if target_page > total_pages: target_page = total_pages
            if target_page < 1: target_page = 1
            start_idx = (target_page - 1) * ITEMS_PER_PAGE
            page_items = next_state[start_idx:start_idx+ITEMS_PER_PAGE]
            updates = []
            for i in range(ITEMS_PER_PAGE):
                if i < len(page_items):
                    item = page_items[i]
                    g_idx = start_idx + i + 1
                    label_str = f"세그먼트{g_idx}-화자{item.get('speaker_num', '1')}" if item.get("is_mixed", False) else f"세그먼트 {g_idx}"
                    updates.extend([
                        gr.update(visible=not item["deleted"]), 
                        gr.update(value=item["wav"], label=label_str), 
                        gr.update(value=item["content"], label=label_str), 
                        item.get("selected", False),
                        item["txt"], 
                        item["wav"]
                    ])
                else:
                    updates.extend([gr.update(visible=False), gr.update(value=None, label=f"세그먼트 {i+1}"), gr.update(value="", label=f"세그먼트 {i+1}"), False, "", ""])
            return [next_state, new_history, new_redo, "[안내] 작업을 앞으로 일괄 돌렸습니다.", target_page, [], f"### 페이지: {target_page} / {total_pages} (총 {len(next_state)}개 항목)", f"### 페이지: {target_page} / {total_pages}", gr.update(interactive=bool(new_history)), gr.update(interactive=bool(new_redo))] + updates

        def save_all_changes(items, history, redo, page_num, *args):
            half = len(text_components)
            current_texts = args[:half]
            current_checkboxes = args[half:]

            if not items:
                return [items, history, redo, "저장할 데이터가 없습니다.", gr.update(interactive=False), gr.update(interactive=False)]
            try:
                current_state = sync_current_data(items, page_num, current_texts, current_checkboxes)
                
                # 저장 시점에 실제 분할/합병 백엔드 실행 및 파일 생성
                for item in current_state:
                    if item.get("action_type") == "split_source":
                        v_sources = item.get("virtual_sources", [])
                        if v_sources and os.path.exists(v_sources[0]):
                            parent_wav = v_sources[0]
                            parent_base = os.path.splitext(parent_wav)[0]
                            parent_txt = f"{parent_base}.txt"
                            ah.split_audio_segment(parent_wav, parent_txt)
                    elif item.get("action_type") == "merge_source":
                        v_sources = item.get("virtual_sources", [])
                        if len(v_sources) >= 2:
                            cur_w = v_sources[0]
                            cur_t = f"{os.path.splitext(cur_w)[0]}.txt"
                            for sw in v_sources[1:]:
                                st = f"{os.path.splitext(sw)[0]}.txt"
                                _, _, res_info = ah.merge_audio_segments(cur_w, sw, cur_t, st)
                                if res_info:
                                    cur_w = res_info["wav"]
                                    cur_t = res_info["txt"]

                surviving_items = []
                for item in current_state:
                    if item["deleted"]:
                        if os.path.exists(item["wav"]): 
                            try: os.remove(item["wav"])
                            except: pass
                        if os.path.exists(item["txt"]): 
                            try: os.remove(item["txt"])
                            except: pass
                    else:
                        # 🔥 합병된 항목의 경우, 참여했던 원본 소스 파일들(virtual_sources) 물리 삭제 처리 보완
                        if item.get("action_type") == "merge_source":
                            for sw in item.get("virtual_sources", []):
                                st = f"{os.path.splitext(sw)[0]}.txt"
                                if sw != item["wav"]:
                                    if os.path.exists(sw):
                                        try: os.remove(sw)
                                        except: pass
                                    if os.path.exists(st):
                                        try: os.remove(st)
                                        except: pass

                        with open(item["txt"], "w", encoding="utf-8") as tf:
                            tf.write(item["content"])
                        item["original_content"] = item["content"]
                        item["is_temp"] = False
                        item["saved"] = True
                        item["action_type"] = None
                        item["virtual_sources"] = []
                        surviving_items.append(item)
                        
                return [surviving_items, [], [], "[성공] 변경사항이 저장되었습니다.", gr.update(interactive=False), gr.update(interactive=False)]
            except Exception as e:
                log_error(MODULE_NAME, "변경사항 저장 실패", e, debug=True)
                return [items, history, redo, f"[오류] 저장 실패: {e}", gr.update(interactive=bool(history)), gr.update(interactive=bool(redo))]

        refresh_btn.click(
            fn=lambda: (gr.Dropdown(choices=get_single_speaker_folders()), gr.Dropdown(choices=get_basic_segment_folders())),
            outputs=[single_dropdown, basic_dropdown]
        )
        
        load_outputs = [state_items, state_history, state_redo, state_page, state_checkboxes, info_box, page_info_md, page_info_md_bottom, undo_global_btn, redo_global_btn]
        for i in range(ITEMS_PER_PAGE):
            load_outputs.extend([row_components[i], audio_components[i], text_components[i], select_checkbox_components[i], txt_path_components[i], wav_path_components[i]])
            
        load_single_btn.click(fn=load_folder_data, inputs=[state_items, single_dropdown], outputs=load_outputs)
        load_basic_btn.click(fn=load_folder_data, inputs=[state_items, basic_dropdown], outputs=load_outputs)

        pagination_outputs = [state_items, state_page, state_checkboxes, page_info_md, page_info_md_bottom]
        for i in range(ITEMS_PER_PAGE):
            pagination_outputs.extend([row_components[i], audio_components[i], text_components[i], select_checkbox_components[i], txt_path_components[i], wav_path_components[i]])

        prev_btn.click(
            fn=lambda items, p, *chk: change_page(items, p, -1, *chk),
            inputs=[state_items, state_page] + select_checkbox_components,
            outputs=pagination_outputs
        )
        prev_btn_bottom.click(
            fn=lambda items, p, *chk: change_page(items, p, -1, *chk),
            inputs=[state_items, state_page] + select_checkbox_components,
            outputs=pagination_outputs
        )
        
        next_btn.click(
            fn=lambda items, p, *chk: change_page(items, p, 1, *chk),
            inputs=[state_items, state_page] + select_checkbox_components,
            outputs=pagination_outputs
        )
        next_btn_bottom.click(
            fn=lambda items, p, *chk: change_page(items, p, 1, *chk),
            inputs=[state_items, state_page] + select_checkbox_components,
            outputs=pagination_outputs
        )

        for t_comp in text_components:
            for trigger_event in [t_comp.blur, t_comp.submit]:
                trigger_event(
                    fn=handle_text_commit,
                    inputs=[state_items, state_history, state_redo, state_page] + text_components + select_checkbox_components,
                    outputs=[state_items, state_history, state_redo, undo_global_btn, redo_global_btn]
                )

        undo_outputs = [state_items, state_history, state_redo, info_box, state_page, state_checkboxes, page_info_md, page_info_md_bottom, undo_global_btn, redo_global_btn]
        for i in range(ITEMS_PER_PAGE):
            undo_outputs.extend([row_components[i], audio_components[i], text_components[i], select_checkbox_components[i], txt_path_components[i], wav_path_components[i]])

        undo_global_btn.click(fn=handle_undo, inputs=[state_items, state_history, state_redo, state_page] + text_components + select_checkbox_components, outputs=undo_outputs)
        redo_global_btn.click(fn=handle_redo, inputs=[state_items, state_history, state_redo, state_page] + text_components + select_checkbox_components, outputs=undo_outputs)

        merge_selected_outputs = [state_items, state_history, state_redo, info_box, undo_global_btn, redo_global_btn, state_page, state_checkboxes, page_info_md, page_info_md_bottom]
        for i in range(ITEMS_PER_PAGE):
            merge_selected_outputs.extend([row_components[i], audio_components[i], text_components[i], select_checkbox_components[i], txt_path_components[i], wav_path_components[i]])

        merge_selected_btn.click(
            fn=handle_selected_merge,
            inputs=[state_items, state_history, state_redo, state_page] + text_components + select_checkbox_components,
            outputs=merge_selected_outputs
        )

        for i in range(ITEMS_PER_PAGE):
            single_outputs = [state_items, state_history, state_redo, info_box, undo_global_btn, redo_global_btn, state_page, state_checkboxes, page_info_md, page_info_md_bottom]
            for j in range(ITEMS_PER_PAGE):
                single_outputs.extend([row_components[j], audio_components[j], text_components[j], select_checkbox_components[j], txt_path_components[j], wav_path_components[j]])
            
            delete_btn_components[i].click(
                fn=lambda items, history, redo, p_num, *args, idx=i: handle_single_delete(items, history, redo, p_num, idx, *args),
                inputs=[state_items, state_history, state_redo, state_page] + text_components + select_checkbox_components,
                outputs=single_outputs
            )
            split_btn_components[i].click(
                fn=lambda items, history, redo, p_num, *args, idx=i: handle_single_split(items, history, redo, p_num, idx, *args),
                inputs=[state_items, state_history, state_redo, state_page] + text_components + select_checkbox_components,
                outputs=single_outputs
            )

        save_all_btn.click(
            fn=save_all_changes,
            inputs=[state_items, state_history, state_redo, state_page] + text_components + select_checkbox_components,
            outputs=[state_items, state_history, state_redo, info_box, undo_global_btn, redo_global_btn]
        )

        def register_algo_action(speaker_name, items, history, redo, page_num, *args):
            speaker_name = speaker_name.strip()
            if not speaker_name: return "[오류] 등록할 화자 이름을 입력해주세요."
            if not items: return "[오류] 로드된 세그먼트 데이터가 없습니다."
            
            for item in items:
                if not item["deleted"]:
                    if item.get("is_mixed", False) and item.get("speaker_num") != "1":
                        return "[거부 경고] 다중 화자 또는 혼입된 세그먼트가 감지되었습니다. 단일 화자 데이터로 완전히 정제한 후에만 등록할 수 있습니다."
            
            try:
                existing_speakers = []
                if hasattr(ah, "get_registered_speakers"):
                    existing_speakers = ah.get_registered_speakers()
                elif hasattr(ah, "list_speakers"):
                    existing_speakers = ah.list_speakers()
                
                if existing_speakers and speaker_name in existing_speakers:
                    log_info(MODULE_NAME, f"'{speaker_name}' 화자는 이미 등록되어 있습니다. 기존 알고리즘 업데이트/덮어쓰기를 진행합니다.")
            except Exception as check_err:
                log_error(MODULE_NAME, "기존 화자 목록 확인 중 예외 발생 (무시됨)", check_err, debug=True)

            new_items, _, _, save_msg, _, _ = save_all_changes(items, history, redo, page_num, *args)
            if "[오류]" in save_msg: return f"등록 실패: {save_msg}"
            
            try:
                first_txt_path = new_items[0]["txt"] if new_items else None
                if not first_txt_path: return "[오류] 유효한 폴더 경로를 찾을 수 없습니다."
                success = ah.register_dataset_from_refined_folder(speaker_name, os.path.dirname(first_txt_path))
                if success: return f"[🎉 성공] '{speaker_name}' 화자의 알고리즘 등록 및 텍스트 치환 사전 학습 완료!"
                return f"[오류] 알고리즘 등록 처리 중 문제가 발생했습니다."
            except Exception as e:
                log_error(MODULE_NAME, f"화자 알고리즘 등록 처리 중 예외 발생 ({speaker_name})", e, debug=True)
                return f"[오류] 예외 발생: {e}"

        register_algo_inputs = [speaker_name_input, state_items, state_history, state_redo, state_page] + text_components + select_checkbox_components
        register_algo_btn.click(fn=register_algo_action, inputs=register_algo_inputs, outputs=[info_box])

    demo.launch(inbrowser=True, server_name="127.0.0.1", server_port=None, prevent_thread_lock=True)
    
    input("[*] Web UI가 실행되었습니다. Web UI 종료 버튼 클릭 시 엔터키로 메뉴이동\n")
    try:
        demo.close()
    except Exception as e:
        log_error(MODULE_NAME, "Gradio 데모 종료 중 예외 발생", e, debug=True)
    log_info(MODULE_NAME, "메인 메뉴로 복귀합니다.")
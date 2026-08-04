import os
import numpy as np
import gradio as gr
import algorithm_handler as ah

SEGMENTS_BASE_DIR = "segments_base"
ITEMS_PER_PAGE = 10

def get_single_speaker_folders():
    if not os.path.exists(SEGMENTS_BASE_DIR):
        return []
    folders = []
    for root, dirs, files in os.walk(SEGMENTS_BASE_DIR):
        for d in dirs:
            if d.startswith("seg_single"):
                rel_path = os.path.relpath(os.path.join(root, d), SEGMENTS_BASE_DIR)
                folders.append(rel_path)
    return folders

def get_basic_segment_folders():
    if not os.path.exists(SEGMENTS_BASE_DIR):
        return []
    folders = []
    for root, dirs, files in os.walk(SEGMENTS_BASE_DIR):
        for d in dirs:
            if d.startswith("세그먼트") and not d.startswith("seg_multi"):
                rel_path = os.path.relpath(os.path.join(root, d), SEGMENTS_BASE_DIR)
                folders.append(rel_path)
    return folders

def clone_items(items):
    return [item.copy() for item in items]

def run_data_refinement_webui():
    print("\n[*] 데이터 정제 Web UI 스튜디오를 실행합니다...")
    
    single_folders = get_single_speaker_folders()
    basic_folders = get_basic_segment_folders()
    
    default_single = single_folders[0] if single_folders else None
    default_basic = basic_folders[0] if basic_folders else None

    with gr.Blocks(title="데이터 정제 스튜디오") as demo:
        with gr.Row():
            gr.Markdown("## 🛠️ 데이터 정제 및 화자 알고리즘 등록 스튜디오")
            close_ui_btn = gr.Button("🚪 Web UI 종료 (메뉴로 돌아가기)", variant="stop")

        gr.Markdown("💡 **안내:** 단일 화자 세그먼트 또는 기본 분석 세그먼트 중 작업할 폴더를 선택하여 불러오세요. (다중 화자는 원천 차단됩니다)")

        with gr.Row(variant="panel"):
            with gr.Column():
                gr.Markdown("### 🎤 단일 화자 분석 폴더 선택")
                single_dropdown = gr.Dropdown(choices=single_folders, value=default_single, label="단일 화자 세그먼트", interactive=True)
                load_single_btn = gr.Button("📁 단일 화자 불러오기", variant="primary")
            with gr.Column():
                gr.Markdown("### 📁 기본 분석 폴더 선택 (알고리즘 최초 등록용)")
                basic_dropdown = gr.Dropdown(choices=basic_folders, value=default_basic, label="기본 분석 세그먼트", interactive=True)
                load_basic_btn = gr.Button("📁 기본 분석 불러오기", variant="primary")

        with gr.Row():
            refresh_btn = gr.Button("🔄 전체 목록 새로고침")

        with gr.Row(variant="panel"):
            with gr.Column(scale=2):
                speaker_name_input = gr.Textbox(label="등록할 화자 이름", placeholder="예: 화자_a", lines=1)
            with gr.Column(scale=1):
                save_all_btn = gr.Button("💾 전체 변경사항 저장", variant="secondary")
            with gr.Column(scale=1):
                register_algo_btn = gr.Button("🚀 화자 알고리즘 등록/업데이트", variant="primary")

        state_items = gr.State([])      
        state_history = gr.State([])    
        state_redo = gr.State([])       
        state_page = gr.State(1)        

        info_box = gr.Textbox(label="현재 진행 상황", value="작업할 폴더를 선택하고 '불러오기' 버튼을 누르세요.", interactive=False)

        with gr.Row():
            prev_btn = gr.Button("⬅️ 이전 페이지")
            with gr.Column(scale=2, min_width=250):
                page_info_md = gr.Markdown("### 페이지: 0 / 0 (총 0개 항목)")
            with gr.Column(scale=1, min_width=180):
                with gr.Row():
                    undo_global_btn = gr.Button("↩️", elem_classes=["icon-btn"], interactive=False)
                    redo_global_btn = gr.Button("🔁", elem_classes=["icon-btn"], interactive=False)
            next_btn = gr.Button("다음 페이지 ➡️")

        row_components = []
        audio_components = []
        text_components = []
        txt_path_components = []
        wav_path_components = []
        delete_btn_components = []

        for i in range(ITEMS_PER_PAGE):
            with gr.Row(visible=False) as r_box:
                with gr.Column(scale=2):
                    a_comp = gr.Audio(label=f"세그먼트 #{i+1} 오디오", type="filepath", interactive=False)
                with gr.Column(scale=3):
                    t_comp = gr.Textbox(label=f"세그먼트 #{i+1} 대사 수정", lines=2, interactive=True)
                with gr.Column(scale=1, min_width=80):
                    del_btn = gr.Button("🗑️ 삭제", variant="stop")
                
                tp_comp = gr.Textbox(visible=False)
                wp_comp = gr.Textbox(visible=False)
                
            row_components.append(r_box)
            audio_components.append(a_comp)
            text_components.append(t_comp)
            delete_btn_components.append(del_btn)
            txt_path_components.append(tp_comp)
            wav_path_components.append(wp_comp)

        with gr.Row():
            prev_btn_bottom = gr.Button("⬅️ 이전 페이지 ")
            page_info_md_bottom = gr.Markdown("### 페이지: 0 / 0")
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

        def load_folder_data(folder_name):
            if not folder_name:
                return [[], [], [], 1, "폴더가 선택되지 않았습니다.", "### 페이지: 0 / 0 (총 0개 항목)", "### 페이지: 0 / 0", gr.update(interactive=False), gr.update(interactive=False)] + [gr.update(visible=False)] * (ITEMS_PER_PAGE * 5)
            
            target_dir = os.path.join(SEGMENTS_BASE_DIR, folder_name)
            if not os.path.exists(target_dir):
                return [[], [], [], 1, "존재하지 않는 폴더입니다.", "### 페이지: 0 / 0 (총 0개 항목)", "### 페이지: 0 / 0", gr.update(interactive=False), gr.update(interactive=False)] + [gr.update(visible=False)] * (ITEMS_PER_PAGE * 5)
                
            wav_files = sorted([f for f in os.listdir(target_dir) if f.lower().endswith('.wav')])
            if not wav_files:
                return [[], [], [], 1, "해당 폴더에 WAV 파일이 없습니다.", "### 페이지: 0 / 0 (총 0개 항목)", "### 페이지: 0 / 0", gr.update(interactive=False), gr.update(interactive=False)] + [gr.update(visible=False)] * (ITEMS_PER_PAGE * 5)
                
            items = []
            for w_f in wav_files:
                base_name = os.path.splitext(w_f)[0]
                w_path = os.path.join(target_dir, w_f)
                t_path = os.path.join(target_dir, f"{base_name}.txt")
                
                t_content = ""
                if os.path.exists(t_path):
                    with open(t_path, "r", encoding="utf-8") as tf:
                        t_content = tf.read().strip()
                items.append({"wav": w_path, "txt": t_path, "content": t_content, "original_content": t_content, "deleted": False})
            
            total_pages = int(np.ceil(len(items) / ITEMS_PER_PAGE))
            status_msg = f"총 {len(items)}개의 세그먼트 로드 완료 (총 {total_pages}페이지)."
            
            updates = []
            page_items = items[:ITEMS_PER_PAGE]
            for i in range(ITEMS_PER_PAGE):
                if i < len(page_items):
                    item = page_items[i]
                    updates.extend([gr.update(visible=not item["deleted"]), item["wav"], item["content"], item["txt"], item["wav"]])
                else:
                    updates.extend([gr.update(visible=False), None, "", "", ""])
            
            return [items, [], [], 1, status_msg, f"### 페이지: 1 / {total_pages} (총 {len(items)}개 항목)", f"### 페이지: 1 / {total_pages}", gr.update(interactive=False), gr.update(interactive=False)] + updates

        def render_page(items, page_num):
            if not items:
                return [1, "### 페이지: 0 / 0 (총 0개 항목)", "### 페이지: 0 / 0"] + [gr.update(visible=False)] * (ITEMS_PER_PAGE * 4)
            
            total_pages = int(np.ceil(len(items) / ITEMS_PER_PAGE))
            if page_num < 1: page_num = 1
            if page_num > total_pages: page_num = total_pages
            
            start_idx = (page_num - 1) * ITEMS_PER_PAGE
            page_items = items[start_idx:start_idx + ITEMS_PER_PAGE]
            
            updates = []
            for i in range(ITEMS_PER_PAGE):
                if i < len(page_items):
                    item = page_items[i]
                    updates.extend([gr.update(visible=not item["deleted"]), item["wav"], item["content"], item["txt"], item["wav"]])
                else:
                    updates.extend([gr.update(visible=False), None, "", "", ""])
            
            return [page_num, f"### 페이지: {page_num} / {total_pages} (총 {len(items)}개 항목)", f"### 페이지: {page_num} / {total_pages}"] + updates

        def change_page(items, current_page, direction):
            total_pages = int(np.ceil(len(items) / ITEMS_PER_PAGE)) if items else 1
            new_page = max(1, min(current_page + direction, total_pages))
            return render_page(items, new_page)

        def sync_current_texts(items, page_num, *current_texts):
            updated = clone_items(items)
            start_idx = (page_num - 1) * ITEMS_PER_PAGE
            for i in range(ITEMS_PER_PAGE):
                global_idx = start_idx + i
                if global_idx < len(updated) and i < len(current_texts):
                    updated[global_idx]["content"] = current_texts[i]
            return updated

        def find_diff_page(old_items, new_items):
            for idx, (o, n) in enumerate(zip(old_items, new_items)):
                if o["content"] != n["content"] or o["deleted"] != n["deleted"]:
                    return (idx // ITEMS_PER_PAGE) + 1
            return None

        def handle_text_commit(items, history, redo, page_num, *current_texts):
            if not items:
                return [items, history, redo, gr.update(interactive=bool(history)), gr.update(interactive=bool(redo))]
            current_state = sync_current_texts(items, page_num, *current_texts)
            has_changed = False
            for old_it, new_it in zip(items, current_state):
                if old_it["content"] != new_it["content"] or old_it["deleted"] != new_it["deleted"]:
                    has_changed = True
                    break
            if not has_changed:
                return [items, history, redo, gr.update(interactive=bool(history)), gr.update(interactive=bool(redo))]
            new_history = history + [clone_items(items)]
            new_redo = [] 
            return [current_state, new_history, new_redo, gr.update(interactive=True), gr.update(interactive=False)]

        def handle_single_delete(items, history, redo, page_num, item_index, *current_texts):
            if not items:
                return [items, history, redo, "항목이 없습니다.", gr.update(interactive=False), gr.update(interactive=False)] + [gr.update() for _ in range(ITEMS_PER_PAGE * 4)]
            new_items = sync_current_texts(items, page_num, *current_texts)
            start_idx = (page_num - 1) * ITEMS_PER_PAGE
            global_idx = start_idx + item_index
            if global_idx < len(new_items):
                new_items[global_idx]["deleted"] = True
            new_history = history + [clone_items(items)]
            new_redo = []
            page_items = new_items[start_idx:start_idx+ITEMS_PER_PAGE]
            updates = []
            for i in range(ITEMS_PER_PAGE):
                if i < len(page_items):
                    item = page_items[i]
                    updates.extend([gr.update(visible=not item["deleted"]), item["wav"], item["content"], item["txt"], item["wav"]])
                else:
                    updates.extend([gr.update(visible=False), None, "", "", ""])
            return [new_items, new_history, new_redo, f"세그먼트 #{global_idx+1} 삭제됨", gr.update(interactive=True), gr.update(interactive=False)] + updates

        def handle_undo(items, history, redo, page_num, *current_texts):
            if not history:
                return [items, history, redo, "더 이상 되돌릴 수 없습니다.", gr.update(interactive=False), gr.update(interactive=bool(redo)), page_num, "### 페이지: 0 / 0", "### 페이지: 0 / 0"] + [gr.update() for _ in range(ITEMS_PER_PAGE * 4)]
            current_state = sync_current_texts(items, page_num, *current_texts)
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
                    updates.extend([gr.update(visible=not item["deleted"]), item["wav"], item["content"], item["txt"], item["wav"]])
                else:
                    updates.extend([gr.update(visible=False), None, "", "", ""])
            return [prev_state, new_history, new_redo, "[안내] 이전 상태로 일괄 되돌렸습니다.", target_page, f"### 페이지: {target_page} / {total_pages} (총 {len(prev_state)}개 항목)", f"### 페이지: {target_page} / {total_pages}", gr.update(interactive=bool(new_history)), gr.update(interactive=bool(new_redo))] + updates

        def handle_redo(items, history, redo, page_num, *current_texts):
            if not redo:
                return [items, history, redo, "더 이상 앞으로 돌릴 수 없습니다.", gr.update(interactive=bool(history)), gr.update(interactive=False), page_num, "### 페이지: 0 / 0", "### 페이지: 0 / 0"] + [gr.update() for _ in range(ITEMS_PER_PAGE * 4)]
            current_state = sync_current_texts(items, page_num, *current_texts)
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
                    updates.extend([gr.update(visible=not item["deleted"]), item["wav"], item["content"], item["txt"], item["wav"]])
                else:
                    updates.extend([gr.update(visible=False), None, "", "", ""])
            return [next_state, new_history, new_redo, "[안내] 작업을 앞으로 일괄 돌렸습니다.", target_page, f"### 페이지: {target_page} / {total_pages} (총 {len(next_state)}개 항목)", f"### 페이지: {target_page} / {total_pages}", gr.update(interactive=bool(new_history)), gr.update(interactive=bool(new_redo))] + updates

        def save_all_changes(items, history, redo, page_num, *current_texts):
            if not items:
                return [items, history, redo, "저장할 데이터가 없습니다.", gr.update(interactive=False), gr.update(interactive=False)]
            try:
                current_state = sync_current_texts(items, page_num, *current_texts)
                surviving_items = []
                for item in current_state:
                    if item["deleted"]:
                        if os.path.exists(item["wav"]): os.remove(item["wav"])
                        if os.path.exists(item["txt"]): os.remove(item["txt"])
                    else:
                        with open(item["txt"], "w", encoding="utf-8") as tf:
                            tf.write(item["content"])
                        item["original_content"] = item["content"]
                        surviving_items.append(item)
                return [surviving_items, [], [], "[성공] 변경사항이 저장되었습니다.", gr.update(interactive=False), gr.update(interactive=False)]
            except Exception as e:
                return [items, history, redo, f"[오류] 일괄 저장 실패: {e}", gr.update(interactive=bool(history)), gr.update(interactive=bool(redo))]

        refresh_btn.click(
            fn=lambda: (gr.Dropdown(choices=get_single_speaker_folders()), gr.Dropdown(choices=get_basic_segment_folders())),
            outputs=[single_dropdown, basic_dropdown]
        )
        
        load_outputs = [state_items, state_history, state_redo, state_page, info_box, page_info_md, page_info_md_bottom, undo_global_btn, redo_global_btn]
        for i in range(ITEMS_PER_PAGE):
            load_outputs.extend([row_components[i], audio_components[i], text_components[i], txt_path_components[i], wav_path_components[i]])
            
        load_single_btn.click(fn=load_folder_data, inputs=[single_dropdown], outputs=load_outputs)
        load_basic_btn.click(fn=load_folder_data, inputs=[basic_dropdown], outputs=load_outputs)

        pagination_outputs = [state_page, page_info_md, page_info_md_bottom]
        for i in range(ITEMS_PER_PAGE):
            pagination_outputs.extend([row_components[i], audio_components[i], text_components[i], txt_path_components[i], wav_path_components[i]])

        for btn in [prev_btn, prev_btn_bottom]:
            btn.click(fn=lambda items, p: change_page(items, p, -1), inputs=[state_items, state_page], outputs=pagination_outputs)
        for btn in [next_btn, next_btn_bottom]:
            btn.click(fn=lambda items, p: change_page(items, p, 1), inputs=[state_items, state_page], outputs=pagination_outputs)

        for t_comp in text_components:
            for trigger_event in [t_comp.blur, t_comp.submit]:
                trigger_event(
                    fn=handle_text_commit,
                    inputs=[state_items, state_history, state_redo, state_page] + text_components,
                    outputs=[state_items, state_history, state_redo, undo_global_btn, redo_global_btn]
                )

        undo_outputs = [state_items, state_history, state_redo, info_box, state_page, page_info_md, page_info_md_bottom, undo_global_btn, redo_global_btn]
        for i in range(ITEMS_PER_PAGE):
            undo_outputs.extend([row_components[i], audio_components[i], text_components[i], txt_path_components[i], wav_path_components[i]])

        undo_global_btn.click(fn=handle_undo, inputs=[state_items, state_history, state_redo, state_page] + text_components, outputs=undo_outputs)
        redo_global_btn.click(fn=handle_redo, inputs=[state_items, state_history, state_redo, state_page] + text_components, outputs=undo_outputs)

        for i in range(ITEMS_PER_PAGE):
            single_outputs = [state_items, state_history, state_redo, info_box, undo_global_btn, redo_global_btn]
            for j in range(ITEMS_PER_PAGE):
                single_outputs.extend([row_components[j], audio_components[j], text_components[j], txt_path_components[j], wav_path_components[j]])
            delete_btn_components[i].click(
                fn=lambda items, history, redo, p_num, *texts, idx=i: handle_single_delete(items, history, redo, p_num, idx, *texts),
                inputs=[state_items, state_history, state_redo, state_page] + text_components,
                outputs=single_outputs
            )

        save_all_btn.click(
            fn=save_all_changes,
            inputs=[state_items, state_history, state_redo, state_page] + text_components,
            outputs=[state_items, state_history, state_redo, info_box, undo_global_btn, redo_global_btn]
        )

        def register_algo_action(speaker_name, items, history, redo, page_num, *current_texts):
            speaker_name = speaker_name.strip()
            if not speaker_name: return "[오류] 등록할 화자 이름을 입력해주세요."
            if not items: return "[오류] 로드된 세그먼트 데이터가 없습니다."
            
            new_items, _, _, save_msg, _, _ = save_all_changes(items, history, redo, page_num, *current_texts)
            if "[오류]" in save_msg: return f"등록 실패: {save_msg}"
            
            try:
                first_txt_path = new_items[0]["txt"] if new_items else None
                if not first_txt_path: return "[오류] 유효한 폴더 경로를 찾을 수 없습니다."
                success = ah.register_dataset_from_refined_folder(speaker_name, os.path.dirname(first_txt_path))
                if success: return f"[🎉 성공] '{speaker_name}' 화자의 알고리즘 등록 및 텍스트 치환 사전 학습 완료!"
                return f"[오류] 알고리즘 등록 처리 중 문제가 발생했습니다."
            except Exception as e:
                return f"[오류] 예외 발생: {e}"

        register_algo_inputs = [speaker_name_input, state_items, state_history, state_redo, state_page] + text_components
        register_algo_btn.click(fn=register_algo_action, inputs=register_algo_inputs, outputs=[info_box])

    demo.launch(inbrowser=True, server_name="127.0.0.1", server_port=None, prevent_thread_lock=True)
    
    input("[*] Web UI가 실행되었습니다. Web UI 종료 버튼을 눌러 창을 닫은 후 엔터키를 누르면 메인 메뉴로 돌아갑니다...\n")
    try:
        demo.close()
    except:
        pass
    print("[*] 메인 메뉴로 복귀합니다.")
import os
import json
import torch
import shutil
import requests
import sys
import time
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset

# 공통 에러 로거 연동 (터미널 출력 대신 내부 기록용으로만 활용)
from error_logger import log_error, log_info

# 경로 정의
STORAGE_DIR = "saved_algorithms"       # 알고리즘 JSON 폴더
TRAINED_DIR = "trained_personalities"  # LoRA 학습 완료된 어댑터 폴더
SEGMENTS_BASE_DIR = Path("segments_base")
MODULE_NAME = "chatbot_lora"

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

def ensure_directories():
    """필요한 모든 디렉토리 보장"""
    try:
        os.makedirs(STORAGE_DIR, exist_ok=True)
        os.makedirs(TRAINED_DIR, exist_ok=True)
        SEGMENTS_BASE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        log_error(MODULE_NAME, "필요한 디렉토리 생성 실패", e, debug=True)

def load_existing_profiles():
    """saved_algorithms 폴더에서 등록된 화자 프로파일 목록 로드"""
    ensure_directories()
    try:
        if not os.path.exists(STORAGE_DIR):
            return []
        profiles = [f[:-5] for f in os.listdir(STORAGE_DIR) if f.endswith(".json")]
        return profiles
    except Exception as e:
        log_error(MODULE_NAME, "기존 프로파일 목록 로드 중 예외 발생", e, debug=True)
        return []

def get_persona_status(persona_name):
    """화자의 학습 상태 반환 (어댑터 존재 여부 기준)"""
    try:
        algo_exists = os.path.join(STORAGE_DIR, f"{persona_name}.json")
        adapter_config_exists = os.path.join(TRAINED_DIR, persona_name, "adapter_config.json")
        
        if not os.path.exists(algo_exists):
            return "등록되지 않음"
            
        if not os.path.exists(adapter_config_exists):
            return "학습 대기 중 (최초 1회 학습 필요)"
            
        return "학습 완료됨"
    except Exception as e:
        log_error(MODULE_NAME, f"화자 상태 확인 중 예외 발생 ({persona_name})", e)
        return "상태 확인 불가"

def chatbot_persona_menu():
    """
    main.py 메뉴에서 호출되는 챗봇 인물 설정 메뉴
    """
    while True:
        print("\n================================================")
        print("             챗봇 인물 설정 메뉴 (LoRA)")
        print("================================================")
        
        profiles = load_existing_profiles()
        
        if not profiles:
            print("[알림] 등록된 알고리즘 프로파일이 없습니다.")
            print("------------------------------------------------")
            print(" 1. 메인 메뉴로 돌아가기")
            print("------------------------------------------------")
            try:
                choice = input("선택: ").strip()
            except Exception:
                return None
            if choice == "1":
                return None
            else:
                print("오류가 나서 진행할 수 없어. 올바른 번호를 입력해줘.")
                continue

        for idx, persona in enumerate(profiles, 1):
            status = get_persona_status(persona)
            print(f" {idx:2d}. {persona} ({status})")
        
        back_idx = len(profiles) + 1
        print(f" {back_idx:2d}. 메인 메뉴로 돌아가기")
        print("------------------------------------------------")
        
        try:
            choice = input("선택: ").strip()
        except Exception as e:
            log_error(MODULE_NAME, "메뉴 선택 입력 수신 중 예외 발생", e)
            continue
            
        if not choice.isdigit():
            print("오류가 나서 진행할 수 없어. 올바른 번호를 입력해줘.")
            continue
            
        choice_val = int(choice)
        if choice_val == back_idx:
            return None
            
        adjusted_idx = choice_val - 1
        if 0 <= adjusted_idx < len(profiles):
            selected_persona = profiles[adjusted_idx]
            print(f"'{selected_persona}' 인물이 선택되었어.")
            return selected_persona
        else:
            print("오류가 나서 진행할 수 없어. 존재하지 않는 번호야.")


# 제공해주신 분석 파이프라인의 시간 포맷 및 게이지바 코드 (수정 없이 그대로 활용)[cite: 2]
def format_time(seconds):
    try:
        if seconds < 60:
            return f"{seconds:.1f}초"
        elif seconds < 3600:
            m = int(seconds // 60)
            s = seconds % 60
            return f"{m}분 {s:.1f}초"
        else:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = seconds % 60
            return f"{h}시간 {m}분 {s:.1f}초"
    except Exception as e:
        return f"{seconds}초"

def format_mmss(seconds):
    try:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"
    except Exception:
        return "00:00"

def print_clean_stage_progress(current_item, total_items, start_time):
    if total_items <= 0:
        percent = 100
        sub_progress = 1.0
    else:
        percent = int((current_item / total_items) * 100)
        percent = min(100, max(0, percent))
        sub_progress = current_item / total_items

    now = time.time()
    elapsed = max(0.0, now - start_time)
    
    if sub_progress > 0.0 and current_item < total_items:
        estimated_total = elapsed / sub_progress
        eta = max(0.0, estimated_total - elapsed)
    else:
        eta = 0.0

    elapsed_str = format_mmss(elapsed)
    eta_str = format_mmss(eta)

    terminal_width = shutil.get_terminal_size(fallback=(120, 24)).columns
    prefix = f"{percent:3d}% |"
    suffix = f"| {current_item}/{total_items} {elapsed_str}<{eta_str}"
    
    fixed_width = len(prefix) + len(suffix)
    bar_len = max(10, terminal_width - fixed_width - 2)
    
    filled_len = int(bar_len * sub_progress)
    filled_len = min(bar_len, max(0, filled_len))
    empty_len = bar_len - filled_len
    
    bar_str = "█" * filled_len + " " * empty_len

    if current_item >= total_items:
        sys.stdout.write("\r" + prefix + bar_str + suffix + "\n")
        sys.stdout.flush()
    else:
        sys.stdout.write("\r" + prefix + bar_str + suffix)
        sys.stdout.flush()


class ProgressCallback(TrainerCallback):
    """Trainer 진행 상황을 제공해주신 게이지바 형태로 출력하기 위한 콜백"""
    def __init__(self, start_time, total_steps):
        self.start_time = start_time
        self.total_steps = total_steps

    def on_step_end(self, args, state, control, **kwargs):
        current_step = state.global_step
        if self.total_steps > 0:
            print_clean_stage_progress(current_step, self.total_steps, self.start_time)


class LocalPersonaChatbot:
    def __init__(self, target_persona):
        self.target_persona = target_persona
        self.adapter_dir = os.path.join(TRAINED_DIR, target_persona)
        
        # 기억 상태 변수 초기화
        self.web_search_allowed = None  
        self.custom_rules = []          # 사용자 지정 규칙들
        self.relationships = {}         # 인물 관계 설정들
        
        print("🤖 로컬 모델 및 토크나이저 로딩 중...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.base_model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
        except Exception as e:
            log_error(MODULE_NAME, "기본 모델 토크나이저 및 파이프라인 로딩 실패", e, debug=True)
            print("오류가 나서 모델을 불러올 수 없어.")
            raise

        # 어댑터 존재 여부 확인 및 재학습 여부 질의 분기 처리
        adapter_config_path = os.path.join(self.adapter_dir, "adapter_config.json")
        should_train = False

        if not os.path.exists(adapter_config_path):
            print(f"\n[시작] '{target_persona}'의 최초 LoRA 학습을 진행할게요.")
            should_train = True
        else:
            print(f"\n[알림] '{target_persona}'의 기존 학습된 어댑터가 존재합니다.")
            try:
                re_train = input("재학습하시겠습니까? (y/n): ").strip().lower()
                if re_train == 'y':
                    print(f"'{target_persona}'의 LoRA 재학습을 진행합니다.")
                    should_train = True
                else:
                    print(f"'{target_persona}'의 기존 어댑터를 불러옵니다.")
            except Exception:
                print(f"'{target_persona}'의 기존 어댑터를 불러옵니다.")

        if should_train:
            success = self.train_lora_from_segments()
            if not success:
                print("학습에 실패하여 기본 모델로 동작합니다.")

        # 모델에 LoRA 어댑터 장착
        try:
            if os.path.exists(os.path.join(self.adapter_dir, "adapter_config.json")):
                self.model = PeftModel.from_pretrained(self.base_model, self.adapter_dir)
            else:
                self.model = self.base_model
        except Exception as e:
            log_error(MODULE_NAME, f"어댑터 로딩 실패 ({target_persona})", e, debug=True)
            self.model = self.base_model

        # 저장된 권한 및 대화 기억 상태 복원
        self.load_learned_state()

    def train_lora_from_segments(self):
        algo_file = os.path.join(STORAGE_DIR, f"{self.target_persona}.json")
        texts = []

        if os.path.exists(algo_file):
            try:
                with open(algo_file, "r", encoding="utf-8") as f:
                    algo_data = json.load(f)
                    samples = algo_data.get("samples", [])
                    for sample in samples:
                        txt_path = sample.get("txt_path")
                        if txt_path and os.path.exists(txt_path):
                            content = Path(txt_path).read_text(encoding="utf-8").strip()
                            if content:
                                texts.append(content)
            except Exception as e:
                log_error(MODULE_NAME, f"세그먼트 읽기 실패 ({self.target_persona})", e, debug=True)

        if not texts:
            texts = [f"내 이름은 {self.target_persona}이고, 내 말투대로 말해."]

        formatted_data = []
        for text in texts:
            prompt = f"인물: {self.target_persona}\n대화 내용: {text}"
            tokenized = self.tokenizer(prompt, truncation=True, max_length=512, padding="max_length")
            
            labels = tokenized["input_ids"].copy()
            labels = [
                -100 if token == self.tokenizer.pad_token_id else token
                for token in labels
            ]
            tokenized["labels"] = labels
            formatted_data.append(tokenized)

        dataset = Dataset.from_list(formatted_data)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"]
        )

        model_peft = get_peft_model(self.base_model, peft_config)

        temp_adapter_dir = os.path.join(TRAINED_DIR, f"{self.target_persona}_temp")
        backup_adapter_dir = os.path.join(TRAINED_DIR, f"{self.target_persona}_backup")
        
        if os.path.exists(temp_adapter_dir):
            shutil.rmtree(temp_adapter_dir)
        if os.path.exists(backup_adapter_dir):
            shutil.rmtree(backup_adapter_dir)

        batch_size = 2
        num_epochs = 3
        total_steps = (len(dataset) // batch_size) * num_epochs
        if total_steps == 0:
            total_steps = 1

        training_args = TrainingArguments(
            output_dir=temp_adapter_dir,
            per_device_train_batch_size=batch_size,
            num_train_epochs=num_epochs,
            logging_steps=1,
            save_strategy="no",
            learning_rate=2e-4,
            fp16=torch.cuda.is_available(),
            report_to="none"
        )

        stage_start_time = time.time()
        print(f"\nLoRA 파인튜닝 학습 시작 (총 {len(dataset)}개 샘플, {num_epochs} 에폭)")
        
        trainer = Trainer(
            model=model_peft,
            args=training_args,
            train_dataset=dataset,
            callbacks=[ProgressCallback(stage_start_time, total_steps)]
        )

        try:
            trainer.train()
            model_peft.save_pretrained(temp_adapter_dir)
            
            if os.path.exists(self.adapter_dir):
                shutil.move(self.adapter_dir, backup_adapter_dir)
            
            shutil.move(temp_adapter_dir, self.adapter_dir)
            
            if not os.path.exists(os.path.join(self.adapter_dir, "adapter_config.json")):
                raise RuntimeError("새 어댑터 검증 실패")

            if os.path.exists(backup_adapter_dir):
                shutil.rmtree(backup_adapter_dir)
                
            self.save_learned_state()
            
            print(f"\n[완료] '{self.target_persona}' 세그먼트 기반 LoRA 학습 완료 (소요 시간: {format_time(time.time() - stage_start_time)})")
            log_info(MODULE_NAME, f"'{self.target_persona}' 세그먼트 기반 LoRA 학습 완료.")
            return True
        except Exception as e:
            log_error(MODULE_NAME, f"LoRA 학습 중 예외 발생 ({self.target_persona})", e, debug=True)
            
            if os.path.exists(temp_adapter_dir):
                shutil.rmtree(temp_adapter_dir)
                
            if os.path.exists(backup_adapter_dir):
                if os.path.exists(self.adapter_dir):
                    shutil.rmtree(self.adapter_dir)
                shutil.move(backup_adapter_dir, self.adapter_dir)
                
            print("\n오류가 나서 학습을 진행할 수 없어. 기존 설정을 유지할게.")
            return False

    def save_learned_state(self):
        """웹 검색 권한뿐만 아니라 규칙 및 인물 관계 설정도 영구 저장"""
        ensure_directories()
        meta_path = os.path.join(self.adapter_dir, "meta_state.json")
        try:
            meta_data = {
                "web_search_allowed": self.web_search_allowed,
                "custom_rules": self.custom_rules,
                "relationships": self.relationships
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            log_error(MODULE_NAME, f"상태 저장 실패 ({meta_path})", e, debug=True)

    def load_learned_state(self):
        """저장된 메타 파일에서 권한, 규칙, 인물 관계 복원"""
        meta_path = os.path.join(self.adapter_dir, "meta_state.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.web_search_allowed = data.get("web_search_allowed", None)
                    self.custom_rules = data.get("custom_rules", [])
                    self.relationships = data.get("relationships", {})
            except Exception as e:
                log_error(MODULE_NAME, f"상태 로드 실패 ({meta_path})", e)

    def process_user_intent_for_memory(self, user_message):
        """사용자의 대화에서 인물 관계 변경, 규칙 추가/삭제 등의 요청을 감지하고 메모리에 반영합니다."""
        msg = user_message.strip()
        updated = False

        if "불러줘" in msg or "관계야" in msg or "기억해" in msg or "규칙" in msg:
            if msg not in self.custom_rules:
                self.custom_rules.append(msg)
                updated = True
                
        if updated:
            self.save_learned_state()
            return True
        return False

    def search_web(self, query):
        try:
            results = DDGS().text(query, max_results=2)
            scraped_info = ""
            for r in results:
                url = r.get("href")
                resp = requests.get(url, timeout=5)
                soup = BeautifulSoup(resp.text, "html.parser")
                text = " ".join([p.get_text() for p in soup.find_all("p")])
                scraped_info += text[:1000] + "\n"
            return scraped_info if scraped_info else "검색 결과 내용을 가져오지 못했습니다."
        except Exception as e:
            log_error(MODULE_NAME, f"웹 검색 중 예외 발생 (query: {query})", e, debug=True)
            return "오류가 나서 검색을 진행할 수 없어."

    def generate_response(self, user_message):
        try:
            self.load_learned_state()

            memory_updated = self.process_user_intent_for_memory(user_message)

            search_context = ""
            if any(keyword in user_message for keyword in ["검색", "찾아", "기사", "최근", "뉴스", "조사"]):
                if self.web_search_allowed is None:
                    prompt_ask = f"인물: {self.target_persona}\n상황: 사용자가 인터넷 검색을 요구했으나 아직 허락을 받지 않았다. 사용자에게 직접 인터넷 검색을 허락해 줄 것인지 네 말투로 물어봐라.\n답변:"
                    inputs_ask = self.tokenizer(prompt_ask, return_tensors="pt").to(self.model.device)
                    with torch.no_grad():
                        out_ask = self.model.generate(**inputs_ask, max_new_tokens=100, do_sample=True, temperature=0.7)
                    ask_msg = self.tokenizer.decode(out_ask[0], skip_special_tokens=True).replace(prompt_ask, "").strip()
                    
                    confirm = input(f"\n{self.target_persona}: {ask_msg} (y/n): ").strip().lower()
                    self.web_search_allowed = (confirm == "y")
                    self.save_learned_state()

                if self.web_search_allowed is True:
                    search_result = self.search_web(user_message)
                    search_context = f"\n[인터넷 검색 참고 자료]\n{search_result}\n"

            permission_hint = ""
            if self.web_search_allowed is True and any(w in user_message for w in ["그만", "회수", "취소", "안 해도", "지워", "꺼줘"]):
                self.web_search_allowed = False
                self.save_learned_state()
                permission_hint = "(시스템 안내: 사용자가 검색 권한을 거두거나 취소함)"

            rules_text = "\n".join([f"- {rule}" for rule in self.custom_rules]) if self.custom_rules else "없음"
            relations_text = str(self.relationships) if self.relationships else "기본 관계"
            
            memory_prompt_context = f"\n[기억된 설정 및 규칙]\n{rules_text}\n[인물 관계]\n{relations_text}\n"

            if memory_updated:
                permission_hint += " (시스템 안내: 새로운 규칙이나 설정이 기억되었습니다.)"

            prompt = f"인물: {self.target_persona}\n대화 기록 및 상황: 검색권한({self.web_search_allowed}) {permission_hint}\n{memory_prompt_context}\n사용자: {user_message}{search_context}\n답변:"

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=250,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id if hasattr(self.tokenizer, "eos_token_id") else self.tokenizer.pad_token_id
                )

            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_message = decoded.replace(prompt, "").strip()
            
            if not response_message:
                response_message = "뭐라고?"

            return response_message
        except Exception as e:
            log_error(MODULE_NAME, "응답 생성 중 예외 발생", e, debug=True)
            return "오류가 나서 진행할 수 없어."
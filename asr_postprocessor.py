import re
import torch

# 표준 공통 에러 로거 연동
from error_logger import log_error, log_info

MODULE_NAME = "ASRPostProcessor"

def convert_korean_numbers(text):
    """
    1. 일상어 보호 블랙리스트 마스킹 처리 (멤버십, 세어보다 등 오인 방지 포함)
    2. 한국어 숫자(고유어 및 한자어 수사 조합) 아라비아 숫자 변환     
    3. 일상어 원본 복원 및 공백 정리
    """
    if not text:
        return ""

    protected_patterns = [
        # 0 관련
        r"영\s*아니", r"영\s*딴",
        # 1 관련
        r"일단", r"일부", r"일반", r"일이",
        # 2 관련 (핵심)
        r"이번(?:\s*에|\s*주|\s*달)?", r"이것", r"이거", r"이렇게", r"이미",
        # 3 관련 (동음이의어 '세' / '삼' 오인 방지 보호 추가)
        r"삼키", r"삼다", r"세(?:\s*어|\s*봐|\s*보|\s*상)",
        # 4 관련
        r"사실(?:\s*은)?", r"사이(?:\s*에|\s*를)?", r"사람(?:\s*들)?", r"사뭇",
        # 5 관련
        r"오히려", r"오랫동안", r"오랜만", r"오호",
        # 6 관련
        r"육교", r"육성",
        # 7 관련
        r"칠칠\s*맞", r"칠칠\s*치",
        # 8 관련
        r"팔팔\s*하", r"팔자",
        # 9 관련
        r"구하", r"구해", r"구경", r"구분", r"구조",
        # 일반 명사 속 '십' 및 '만' 오인 방지 보호 추가
        r"멤버\s*십", r"만\s*약", r"만\s*일", r"만\s*족", r"만\s*들"
    ]

    protected_storage = []
    def shield_match(match):
        protected_storage.append(match.group(0))
        return f"__PROTECTED_{len(protected_storage) - 1}__"

    for pattern in protected_patterns:
        flexible_pattern = pattern.replace(r"\s*", r"\s*")
        text = re.sub(flexible_pattern, shield_match, text)

    # 고유어 기본 및 복합 조합 확장 (일흔, 여든, 아흔 및 스물한/서른둘 등 대응)
    native_full_map = {
        '하나': '1', '한': '1',
        '둘': '2', '두': '2',
        '셋': '3', '세': '3',
        '넷': '4', '네': '4',
        '다섯': '5', '여섯': '6', '일곱': '7', '여덟': '8', '아홉': '9',
        '열': '10', '열한': '11', '열두': '12', '열세': '13', '열네': '14',
        '열다섯': '15', '열여섯': '16', '열일곱': '17', '열여덟': '18', '열아홉': '19',
        '스물': '20', '스물한': '21', '스물두': '22', '스물셋': '23', '스물넷': '24', '스물다섯': '25', '스물여섯': '26', '스물일곱': '27', '스물여덟': '28', '스물아홉': '29',
        '서른': '30', '서른한': '31', '서른두': '32', '서른셋': '33', '서른넷': '34', '서른다섯': '35', '서른여섯': '36', '서른일곱': '37', '서른여덟': '38', '서른아홉': '39',
        '마흔': '40', '마흔한': '41', '마흔두': '42', '마흔셋': '43', '마흔넷': '44', '마흔다섯': '45', '마흔여섯': '46', '마흔일곱': '47', '마흔여덟': '48', '마흔아홉': '49',
        '쉰': '50', '쉰한': '51', '쉰두': '52', '쉰셋': '53', '쉰넷': '54', '쉰다섯': '55', '쉰여섯': '56', '쉰일곱': '57', '쉰여덟': '58', '쉰아홉': '59',
        '예순': '60', '예순한': '61', '예순두': '62', '예순셋': '63', '예순넷': '64', '예순다섯': '65', '예순여섯': '66', '예순일곱': '67', '예순여덟': '68', '예순아홉': '69',
        '일흔': '70', '일흔한': '71', '일흔두': '72', '일흔셋': '73', '일흔넷': '74', '일흔다섯': '75', '일흔여섯': '76', '일흔일곱': '77', '일흔여덟': '78', '일흔아홉': '79',
        '여든': '80', '여든한': '81', '여든두': '82', '여든셋': '83', '여든넷': '84', '여든다섯': '85', '여든여섯': '86', '여든일곱': '87', '여든여덟': '88', '여든아홉': '89',
        '아흔': '90', '아흔한': '91', '아흔두': '92', '아흔셋': '93', '아흔넷': '94', '아흔다섯': '95', '아흔여섯': '96', '아흔일곱': '97', '아흔여덟': '98', '아흔아홉': '99'
    }
    
    sorted_natives = sorted(native_full_map.keys(), key=len, reverse=True)
    for k in sorted_natives:
        text = re.sub(rf'\b{k}\b', native_full_map[k], text)

    sino_nums = {'일': 1, '이': 2, '삼': 3, '사': 4, '오': 5, '육': 6, '칠': 7, '팔': 8, '구': 9}
    multipliers = {'십': 10, '백': 100, '천': 1000, '만': 10000, '억': 100000000, '조': 1000000000000}
    
    def korean_to_number(word):
        total, current_val = 0, 0
        for char in word:
            if char in sino_nums:
                current_val = sino_nums[char]
            elif char in multipliers:
                if current_val == 0:
                    current_val = 1
                total += current_val * multipliers[char]
                current_val = 0
        total += current_val
        return total

    def replace_sino(m):
        full_str = m.group(0)
        exception_words = ["억지", "조상", "조절", "조차", "조용", "조각", "만일", "만약"]
        if any(ex in full_str for ex in exception_words):
            return full_str
            
        word = m.group(1).replace(' ', '')
        unit = m.group(2)
        
        if word == '만' and unit == '일': 
            return full_str
        if not word: 
            return full_str
            
        # '만육십'처럼 '만', '억', '조' 같은 대단위 뒤에 '십', '백', '천' 같은 작은 자릿수가 
        # 곧바로 이어지는 부자연스러운 조합은 숫자로 변환하지 않고 원문 유지
        if re.search(r'[만억조].*[십백천]', word):
            return full_str
            
        num = korean_to_number(word)
        return f"{num}{unit}"
        
    pattern_sino = r'([일일이삼사오육칠팔구십백천만억조][일일이삼사오육칠팔구십백천만억조\s]*)(원|억|조|만|월|일|년|분|초|프로|퍼센트|개|명|번|살|시)'
    text = re.sub(pattern_sino, replace_sino, text)

    # 60~99 한자어 확장 복합 매핑
    complex_sino_map = {
        '십': '10', '십일': '11', '십이': '12', '십삼': '13', '십사': '14', '십오': '15',
        '십육': '16', '십칠': '17', '십팔': '18', '십구': '19', '이십': '20',
        '이십일': '21', '이십이': '22', '이십삼': '23', '이십사': '24', '이십오': '25',
        '이십육': '26', '이십칠': '27', '이십팔': '28', '이십구': '29', '삼십': '30',
        '삼십일': '31', '삼십이': '32', '삼십삼': '33', '삼십사': '34', '삼십오': '35',
        '삼십육': '36', '삼십칠': '37', '삼십팔': '38', '삼십구': '39', '사십': '40',
        '사십일': '41', '사십이': '42', '사십삼': '43', '사십사': '44', '사십오': '45',
        '사십육': '46', '사십칠': '47', '사십팔': '48', '사십구': '49', '오십': '50',
        '오십일': '51', '오십이': '52', '오십삼': '53', '오십사': '54', '오십오': '55',
        '오십육': '56', '오십칠': '57', '오십팔': '58', '오십구': '59', '육십': '60',
        '육십일': '61', '육십이': '62', '육십삼': '63', '육십사': '64', '육십오': '65',
        '육십육': '66', '육십칠': '67', '육십팔': '68', '육십구': '69', '칠십': '70',
        '칠십일': '71', '칠십이': '72', '칠십삼': '73', '칠십사': '74', '칠십오': '75',
        '칠십육': '76', '칠십칠': '77', '칠십팔': '78', '칠십구': '79', '팔십': '80',
        '팔십일': '81', '팔십이': '82', '팔십삼': '83', '팔십사': '84', '팔십오': '85',
        '팔십육': '86', '팔십칠': '87', '팔십팔': '88', '팔십구': '89', '구십': '90',
        '구십일': '91', '구십이': '92', '구십삼': '93', '구십사': '94', '구십오': '95',
        '구십육': '96', '구십칠': '97', '구십팔': '98', '구십구': '99'    }
    
    for k in sorted(complex_sino_map.keys(), key=len, reverse=True):
        text = re.sub(rf'({k})(?=일부터|일까지|일에|일도|일은|일이|일\b)', complex_sino_map[k], text)
        text = re.sub(rf'\b{k}\b', complex_sino_map[k], text)

    for i, original_text in enumerate(protected_storage):
        text = text.replace(f"__PROTECTED_{i}__", original_text)

    return re.sub(r"\s+", " ", text).strip()


def clean_and_convert_numbers(text):
    """
    외부 통합 정제 파이프라인 확장용 함수
    """
    if not text:
        return ""
    text = clean_hallucination_text(text)
    text = convert_korean_numbers(text)
    return text


def clean_hallucination_text(text):
    """
    1. 한자 및 일본어(히라가나/가타카나) 차단
    2. 인도네시아어/말레이어 환각 패턴 통합 차단
    3. 힌디어(데바나가리 문자 및 주요 환각 패턴) 차단
    4. 일반적인 영어/기타 환각 패턴 차단
    5. 과도한 자음/모음 반복 정제
    """
    if not text:
        return ""
    
    if re.search(r'[\u4e00-\u9fff\u3040-\u30ff\u31f0-\u31ff]', text):
        return ""
    
    indonesian_malay_hallucinations = [
        r'terima\s*kasih', r'selamat\s*pagi', r'selamat\s*siang', r'selamat\s*malam',
        r'silakan', r'mohon', r'perhatian', r'sampai\s*jumpa', r'terima', r'kasih',
        r'selamat', r'pagi', r'siang', r'malam', r'bapak', r'ibu', r'saudara',
        r'sahaja', r'keras\s*sahaja', r'\b(dan|yang|adalah|untuk|dengan|tidak)\b'
    ]
    for pattern in indonesian_malay_hallucinations:
        if re.search(pattern, text, re.IGNORECASE):
            return ""

    hindi_hallucinations = [
        r'[\u0900-\u097F]',  
        r'जो\s*मामे\s*थे\s*ना'     
    ]
    for pattern in hindi_hallucinations:
        if re.search(pattern, text, re.IGNORECASE):
            return ""

    hallucination_patterns = [
        r'subtitles?\s*by', r'amara\.org', r'thank you', r'thanks for watching',
        r'copyright', r'like and subscribe', r'music', r'applause', r'laughter',
        r'b\.g\.m\.', r'oh+', r'ah+', r'yeah+'
    ]
    
    cleaned = text.strip()
    for pattern in hallucination_patterns:
        if re.fullmatch(pattern, cleaned, re.IGNORECASE):
            return ""
        
    cleaned = re.sub(r"([ㄱ-ㅎㅏ-ㅣ])\1{4,}", "", cleaned)
        
    return cleaned


def sanitize_asr_output(raw_text, recent_texts=None, current_start=0.0):
    """
    최근 인식 결과를 바탕으로 한 중복/유사 구간 제거 후처리 필터
    """
    if not raw_text:
        return ""

    hallucination_patterns = [
        r"([ㄱ-ㅎㅏ-ㅣ])\1{4,}"
    ]
    
    for pattern in hallucination_patterns:
        raw_text = re.sub(pattern, "", raw_text, flags=re.IGNORECASE)

    raw_text = re.sub(r"\s+", " ", raw_text).strip()
    if not raw_text:
        return ""

    if recent_texts is not None:
        cleaned_current = re.sub(r'[\s\.,!]', '', raw_text).lower()
        
        for prev_txt, prev_start in recent_texts:
            if abs(current_start - prev_start) < 4.0:
                cleaned_prev = re.sub(r'[\s\.,!]', '', prev_txt).lower()
                if cleaned_prev in cleaned_current or cleaned_current in cleaned_prev:
                    return "" 

        recent_texts.append((raw_text, current_start))
        if len(recent_texts) > 5:
            recent_texts.pop(0)

    return raw_text


def perform_global_asr_pass(model, vocal_audio_data, target_sample_rate):
    """
    전체 오디오 스트림에 대한 1차 ASR 분석 및 정제 수행 
    (30초 단위 오디오 청크 분할 처리 및 메모리/VRAM 누수 방지 적용)
    """
    if model is None or vocal_audio_data is None or vocal_audio_data.size == 0:
        return ""

    full_texts = []
    try:
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        
        # 32GB 시스템 메모리/VRAM 랙 방지를 위해 전체 오디오를 30초 단위 청크로 분할
        chunk_duration = 30.0
        chunk_samples = int(chunk_duration * target_sample_rate)
        total_samples = len(vocal_audio_data)
        
        for start_idx in range(0, total_samples, chunk_samples):
            end_idx = min(start_idx + chunk_samples, total_samples)
            chunk_audio = vocal_audio_data[start_idx:end_idx]
            
            if chunk_audio.size == 0:
                continue
                
            with torch.no_grad():
                res_chunk = model.transcribe((chunk_audio, target_sample_rate))
                
            chunk_text = ""
            if hasattr(res_chunk, "text"):
                chunk_text = res_chunk.text
            elif isinstance(res_chunk, list):
                chunk_text = " ".join([str(item.get("text", item)) if isinstance(item, dict) else str(getattr(item, "text", item)) for item in res_chunk])
            else:
                chunk_text = str(res_chunk)
                
            if chunk_text.strip():
                full_texts.append(chunk_text.strip())
                
            # 각 청크 처리 직후 GPU/시스템 캐시 비우기 (메모리 폭발 방지)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        full_audio_text = " ".join(full_texts)
        
        # 환각 차단 및 숫자 변환 정제 적용
        full_audio_text = clean_hallucination_text(full_audio_text)
        full_audio_text = convert_korean_numbers(full_audio_text)
        
    except Exception as e:
        log_error(MODULE_NAME, "전체 오디오 스트림 1차 ASR 분석 수행 중 예외 발생", e)

    return full_audio_text
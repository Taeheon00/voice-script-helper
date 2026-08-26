import re
import torch
import numpy as np

try:
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
except ImportError as e:
    kiwi = None

# 표준 공통 에러 로거 연동
from error_logger import log_error, log_info

MODULE_NAME = "ASRPostProcessorAdvanced"

if kiwi is None:
    log_error(MODULE_NAME, "Kiwipiepy 패키지 로드 실패 (기본 백업 로직으로 동작합니다)", ImportError("Kiwipiepy not found"))


def normalize_korean_numbers_strict(text):
    if not text:
        return text

    text = re.sub(r'영수\s*증\b', '영수증', text)

    kor_to_num_map = {
        '일': '1', '이': '2', '삼': '3', '사': '4', '오': '5',
        '육': '6', '칠': '7', '팔': '8', '구': '9', '십': '10',
        '백': '100', '천': '1000'
    }

    # 1. '월' 단위 표현 변환 (예: 이월, 삼월 -> 2월, 3월)
    def replace_month(match):
        kor_num = match.group(1)
        val = 0
        try:
            if '십' in kor_num:
                parts = kor_num.split('십')
                tens = int(kor_to_num_map.get(parts[0], 1)) * 10 if parts[0] else 10
                units = int(kor_to_num_map.get(parts[1], 0)) if len(parts) > 1 and parts[1] else 0
                val = tens + units
            else:
                val = int(kor_to_num_map.get(kor_num, kor_num))
            return f"{val}월"
        except Exception as e:
            return match.group(0)

    text = re.sub(r'(?<![가-힣])([일이삼사오육칠팔구십]+)\s*월', replace_month, text)

    def replace_portion(match):
        kor_num = match.group(1)
        try:
            val = int(kor_to_num_map.get(kor_num, kor_num))
            return f"{val}인분"
        except Exception:
            return match.group(0)

    text = re.sub(r'(?<![가-힣])([일이삼사오육칠팔구십]+)\s*인분', replace_portion, text)

    def replace_decade(match):
        kor_part = match.group(1)
        try:
            val = 0
            if '십' in kor_part:
                parts = kor_part.split('십')
                tens_part = parts[0]
                units_part = parts[1] if len(parts) > 1 else ""
                
                tens = int(kor_to_num_map.get(tens_part, 1)) * 10 if tens_part else 10
                units = int(kor_to_num_map.get(units_part, 0)) if units_part else 0
                val = tens + units
            else:
                val = int(kor_to_num_map.get(kor_part, kor_part))
            
            return f"{val}대"
        except Exception:
            return match.group(0)

    text = re.sub(r'(?<![가-힣])([일이삼사오육칠팔구십]+)\s*대(?=[가-힣]|\s|,|\.|$)', replace_decade, text)

    def replace_general_compound_number(match):
        kor_part = match.group(0)
        try:
            if '십' in kor_part:
                parts = kor_part.split('십')
                tens_part = parts[0]
                units_part = parts[1] if len(parts) > 1 else ""
                
                tens = int(kor_to_num_map.get(tens_part, 1)) * 10 if tens_part else 10
                units = int(kor_to_num_map.get(units_part, 0)) if units_part else 0
                return str(tens + units)
            else:
                return kor_to_num_map.get(kor_part, kor_part)
        except Exception:
            return kor_part

    text = re.sub(r'(?<![가-힣])([이삼사오육칠팔구]십[일이삼사오육칠팔구]?)(?![가-힣]|\s*(월|인분|대))', replace_general_compound_number, text)

    def replace_day_or_unit(match):
        kor_part = match.group(1)    
        suffix = match.group(2)    
        
        try:
            val = 0
            if '십' in kor_part:
                parts = kor_part.split('십')
                tens_part = parts[0]
                units_part = parts[1] if len(parts) > 1 else ""
                
                tens = int(kor_to_num_map.get(tens_part, 1)) * 10 if tens_part else 10
                units = int(kor_to_num_map.get(units_part, 0)) if units_part else 0
                val = tens + units
            else:
                val = int(kor_to_num_map.get(kor_part, kor_part))
            
            return f"{val}{suffix}"
        except Exception:
            return match.group(0)

    pattern_unit = r'(?<![가-힣])(?!사\s*인)([일이삼사오육칠팔구십]+)\s*(일부터|일까지|일도|일에|일|원|인|종|개|구매|시|분|초)(?=[가-힣]|\s|,|\.|$)'
    text = re.sub(pattern_unit, replace_day_or_unit, text)

    def replace_large_num(match):
        chunk = match.group(0)
        if not any(unit in chunk for unit in ['조', '억', '만', '천', '백']):
            return chunk
            
        result = chunk
        for kor, num in kor_to_num_map.items():
            result = result.replace(kor, num)
        return result

    text = re.sub(r'(?<![가-힣])(?:[일이삼사오육칠팔구]십[일이삼사오육칠팔구]*|[일이삼사오육칠팔구]*십[일이삼사오육칠팔구]+|[백천조억만]+)', replace_large_num, text)

    return text


def clean_text_advanced(text, apply_number_normalization=True):
    if not text:
        return ""
    
    if re.search(r'[\u4e00-\u9fff\u3040-\u30ff\u31f0-\u31ff]', text):
        return ""
        
    cleaned = text.strip()
    if apply_number_normalization: cleaned = normalize_korean_numbers_strict(cleaned)
    
    if re.search(r"([ㄱ-ㅎㅏ-ㅣ])\1{4,}", cleaned):
        return ""
    
    if kiwi:
        try:
            sentences = kiwi.split_into_sents(cleaned)
            cleaned = " ".join([s.text for s in sentences])
        except Exception as e:
            pass
    
    return cleaned


def apply_algorithm_text_correction(text, mapped_speaker):
    if not text:
        return text
    try:
        import algorithm_handler as ah
        if mapped_speaker and hasattr(ah, "apply_text_corrections"):
            return ah.apply_text_corrections(text, mapped_speaker)
    except Exception:
        pass
    return text


# [핵심] 기존 상위 호출부가 완벽하게 인식하는 원래의 인자 구조 복원
def process_segment_text(chunk_text, recent_texts=None, current_start=0.0, mapped_speaker=None):
    try:
        if not chunk_text:
            return ""

        cleaned = clean_text_advanced(chunk_text, apply_number_normalization=(mapped_speaker is None))
        if not cleaned:
            return ""

        raw_text = re.sub(r"\s+", " ", cleaned).strip()
        if not raw_text:
            return ""

        # 전달된 화자 알고리즘 규칙 적용
        if mapped_speaker:
            raw_text = apply_algorithm_text_correction(raw_text, mapped_speaker)
            raw_text = normalize_korean_numbers_strict(raw_text)

        if len(raw_text) < 2 and not re.search(r'[0-9]', raw_text):
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

    except Exception as e:
        err_msg = f"[ASR-PostProcessor-Error] 세그먼트 후처리 중 예외 발생 (구간: {current_start:.1f}초, 화자: {mapped_speaker})"
        log_error(MODULE_NAME, err_msg, e)
        return ""


def perform_global_asr_pass(model, vocal_audio_data, target_sample_rate):
    if model is None or vocal_audio_data is None or vocal_audio_data.size == 0:
        log_error(MODULE_NAME, "전역 ASR 패스 실패: 모델이 없거나 오디오 데이터가 비어 있습니다.", ValueError("Invalid model or audio data"))
        return ""

    full_texts = []
    try:
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        
        with torch.no_grad():
            try:
                res_segments, info = model.transcribe(
                    (vocal_audio_data, target_sample_rate), 
                    language="ko", 
                    vad_filter=True, 
                    vad_parameters=dict(
                        threshold=0.35,            
                        min_silence_duration_ms=250,  
                        speech_pad_ms=100            
                    ),
                    without_timestamps=False
                )
                
                for segment in res_segments:
                    seg_text = getattr(segment, "text", "")
                    if seg_text and seg_text.strip():
                        refined_seg = clean_text_advanced(seg_text.strip())
                        full_texts.append(refined_seg)
                    else:
                        full_texts.append("")
                        
            except (TypeError, AttributeError) as te:
                log_error(MODULE_NAME, "고급 전역 transcribe 실패로 인해 청크 단위 폴백(Fallback) 처리로 전환합니다.", te)
                chunk_duration = 30.0
                chunk_samples = int(chunk_duration * target_sample_rate)
                total_samples = len(vocal_audio_data)
                
                for start_idx in range(0, total_samples, chunk_samples):
                    end_idx = min(start_idx + chunk_samples, total_samples)
                    chunk_audio = vocal_audio_data[start_idx:end_idx]
                    if chunk_audio.size == 0:
                        continue
                    
                    res_chunk = model.transcribe((chunk_audio, target_sample_rate))
                    chunk_text = res_chunk.text if hasattr(res_chunk, "text") else str(res_chunk)
                    if chunk_text.strip():
                        refined_chunk = clean_text_advanced(chunk_text.strip())
                        full_texts.append(refined_chunk)
                    else:
                        full_texts.append("")

        full_audio_text = " ".join(full_texts)
        if not full_audio_text:
            pass
        
    except Exception as e:
        log_error(MODULE_NAME, "[ASR-Stream-Inference-Error] 오디오 스트림 추론 수행 중 예외 발생", e)
        return ""

    return full_audio_text
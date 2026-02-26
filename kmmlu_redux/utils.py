# solution은 1-indexed (1=A, 2=B, ...)

import os
import re

_DEBUG_SAMPLES = []
_DEBUG_MAX = int(os.environ.get("KMMLU_REDUX_DEBUG", "0"))

def doc_to_text(doc):
    question = doc.get("question", "").strip()
    options = doc.get("options", [])
    n = len(options)
    
    instruction = (
        "다음 문제에 대해 정답을 고르세요. 당신의 최종 정답은 ABCD 중 하나이고, \"정답:\" 뒤에 와야 합니다. 정답을 고르기 전에 차근차근 생각하고 추론하세요.\n\n"
    )
    
    option_str = f"A) {options[0]}\nB) {options[1]}\nC) {options[2]}\nD) {options[3]}"
    if n == 5:
        option_str += f"\nE) {options[4]}"
    
    return (
        instruction
        + f"문제:\n{question}\n\n"
        + f"{option_str}\n\n"
    )

def doc_to_choice(doc):
    """선택지 레이블 (loglikelihood 비교용)."""
    n = len(doc.get("options", []))
    if n == 5:
        return ["A", "B", "C", "D", "E"]
    return ["A", "B", "C", "D"]


def doc_to_target(doc):
    """골드 정답 글자 "A"~"E". lm_eval이 doc_to_target(doc) 1인자로 호출하므로 필수."""
    return "ABCDE"[int(doc.get("solution", 1)) - 1]


def process_results(doc, results):
    """
    모델 생성 텍스트에서 정답 추출. 여러 패턴 시도 후 마지막 매칭 사용.
    - 프롬프트가 "정답:"으로 끝나면 모델은 그 뒤만 생성하므로 "정답:"이 없을 수 있음.
    - 숫자(1~5), 소문자, 전각 콜론도 허용.
    - results: lm_eval은 [ [resp1, resp2, ...] ] 형태로 넘길 수 있음 → 첫 문자열 취함.
    """
    raw = results[0] if results else None
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else None
    full_answer = (raw or "").strip() if isinstance(raw, str) else str(raw or "")
    extracted_answer = ""

    # 1) "정답:" / "Answer:" 뒤 글자(대소문자) 또는 숫자(1~5)
    for pattern in [
        r"(?i)(?:정답|Answer)\s*[:\uFF1A]\s*([A-Ea-e])",
        r"(?i)(?:정답|Answer)\s*[:\uFF1A]?\s*([1-5])\s*(?:번)?",
    ]:
        match = re.findall(pattern, full_answer)
        if match:
            m = match[-1].strip().upper()
            if m in "12345":
                extracted_answer = "ABCDE"[int(m) - 1]
            else:
                extracted_answer = m
            break

    # 2) 없으면 "마지막 줄"이 한 글자만 있을 때 (A. / B 형태)
    if not extracted_answer:
        last_line = full_answer.split("\n")[-1].strip() if full_answer else ""
        if re.match(r"^([A-Ea-e])[\.\)]?$", last_line):
            extracted_answer = last_line[0].upper()

    # 3) 여전히 없으면 응답 끝 120자 안의 마지막 A~E 사용. 단 "through E" 의 E 는 제외
    if not extracted_answer and full_answer:
        tail = full_answer[-120:]
        candidates = [(m.start(), m.group(1).upper()) for m in re.finditer(r"([A-Ea-e])", tail)]
        for i in range(len(candidates) - 1, -1, -1):
            pos, letter = candidates[i]
            # "through E" / "through e" 이면 스킵
            if letter.upper() == "E" and pos >= 8 and tail[pos - 8 : pos].lower() == "through ":
                continue
            extracted_answer = letter
            break

    gold_index = int(doc["solution"], 10) - 1
    gold_letter = "ABCDE"[gold_index]
    predicted_index = "ABCDE".find(extracted_answer)
    acc = 1.0 if (predicted_index >= 0 and gold_index == predicted_index) else 0.0

    # 디버그: KMMLU_REDUX_DEBUG=20 등으로 설정 시 처음 N개 샘플을 kmmlu_redux_debug.jsonl에 저장
    if _DEBUG_MAX > 0 and len(_DEBUG_SAMPLES) < _DEBUG_MAX:
        _DEBUG_SAMPLES.append({
            "gold": gold_letter,
            "extracted": extracted_answer,
            "correct": bool(acc),
            "answer_snippet": full_answer[:800] if full_answer else "",
            "answer_tail": full_answer[-600:] if len(full_answer or "") > 600 else (full_answer or ""),
            "len_chars": len(full_answer or ""),
        })
        if len(_DEBUG_SAMPLES) >= _DEBUG_MAX:
            _write_debug_samples()

    return {"acc": acc}


def _write_debug_samples():
    import json
    path = os.path.join(os.path.dirname(__file__), "kmmlu_redux_debug.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for s in _DEBUG_SAMPLES:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[kmmlu_redux] Wrote {len(_DEBUG_SAMPLES)} debug samples to {path}")
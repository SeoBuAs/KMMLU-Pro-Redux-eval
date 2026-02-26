# LM Eval Tasks: KMMLU-Pro & KMMLU-Redux

한국어 전문/기술 자격 시험 벤치마크 **KMMLU-Pro**, **KMMLU-Redux**를 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)로 평가하기 위한 태스크 정의입니다.
이 모든 산출물은 [LG Aimers 8기]를 수행하며 만들어졌습니다.

- **Paper**: [KMMLU-Pro / KMMLU-Redux](https://arxiv.org/abs/2507.08924)
- **Datasets**: [HuggingFace — LGAI-EXAONE](https://huggingface.co/LGAI-EXAONE)

---

## 태스크 요약

| 태스크 | 설명 | 문항 수 | 데이터셋 |
|--------|------|--------|----------|
| **kmmlu_pro** | 한국 국가 **전문자격** 시험 기반 다지선다 | 2,822 | [LGAI-EXAONE/KMMLU-Pro](https://huggingface.co/datasets/LGAI-EXAONE/KMMLU-Pro) |
| **kmmlu_redux** | 한국 국가 **기술자격**(KNTQ) 기반 다지선다 | 2,587 | [LGAI-EXAONE/KMMLU-Redux](https://huggingface.co/datasets/LGAI-EXAONE/KMMLU-Redux) |

둘 다 0-shot, 정확도(accuracy)로 평가합니다.

---

## 디렉터리 구조

```
lm_eval_tasks/
├── eval.py               # 평가 실행 스크립트 (vllm/hf 백엔드)
├── kmmlu_pro/
│   ├── kmmlu_pro.yaml    # 태스크 설정
│   └── utils.py          # doc_to_text, doc_to_choice, doc_to_target, process_results
├── kmmlu_redux/
│   ├── kmmlu_redux.yaml
│   └── utils.py
└── README.md
```

---

## 사용 방법

`eval.py`를 사용해 평가합니다. (lm-evaluation-harness 기반, vllm/hf 백엔드 지원)

```bash
# lm_eval_tasks 디렉터리에서 실행 (기본 --include_path "." = 현재 디렉터리)
python eval.py --model_path /path/to/model --tasks kmmlu_pro
python eval.py --model_path /path/to/model --tasks kmmlu_redux

# 두 태스크 한 번에
python eval.py --model_path /path/to/model --tasks kmmlu_pro,kmmlu_redux

# 옵션: 백엔드, GPU, 샘플 제한 등
python eval.py --model_path /path/to/model --tasks kmmlu_pro --backend vllm --gpu 0 --limit 100
```

태스크 YAML은 `--include_path` 기본값(현재 디렉터리)으로 자동 로드됩니다.

---

## 라이선스 / 참고

- 벤치마크 논문 및 데이터셋 라이선스는 각 링크에서 확인하세요.
- KMMLU-Pro, KMMLU-Redux: [arXiv:2507.08924](https://arxiv.org/abs/2507.08924)

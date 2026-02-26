#!/usr/bin/env python3
"""
- lm-evaluation-harness 기반
- vllm 또는 hf 백엔드 지원
- 결과를 JSON 파일로 저장
"""

import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="양자화 모델 평가")
    parser.add_argument("--model_path", type=str, required=True,
                        help="평가할 모델 경로")
    parser.add_argument("--tasks", type=str, default="kmmlu_pro",# "gsm8k",
                        help="평가 태스크 (쉼표 구분, 예: gsm8k,arc_challenge)")
    parser.add_argument("--gpu", type=int, default=3,
                        help="사용할 GPU 번호")
    parser.add_argument("--backend", type=str, default="vllm",
                        choices=["vllm", "hf"],
                        help="추론 백엔드 (vllm 또는 hf)")
    parser.add_argument("--num_fewshot", type=int, default=0,
                        help="Few-shot 예시 개수 (KMMLU 등 0-shot 기본, gsm8k 등은 --num_fewshot 5)")
    parser.add_argument("--batch_size", type=str, default="auto",
                        help="배치 사이즈 (auto 또는 숫자)")
    parser.add_argument("--gpu_mem_util", type=float, default=0.9,
                        help="vllm GPU 메모리 활용률 (0.0~1.0, 대회 스펙: 0.85)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="결과 저장 디렉토리 (미지정 시 모델 경로에 저장)")
    parser.add_argument("--max_gen_toks", type=int, default=16384,
                        help="최대 생성 토큰 수 (대회 스펙: 16384)")
    parser.add_argument("--apply_chat_template", action="store_true", default=True,
                        help="채팅 템플릿 적용 여부")
    parser.add_argument("--no_chat_template", action="store_true",
                        help="채팅 템플릿 비적용")
    parser.add_argument("--include_path", type=str, default=".",
                        help="커스텀 태스크 YAML 디렉터리 (쉼표 구분 가능. 기본값 '.' = 이 스크립트와 같은 디렉터리)")
    parser.add_argument("--exclude_tasks", type=str, default="",
                        help="제외할 태스크 (쉼표 구분). GPQA 사용 시 비워두고 --hf_token 또는 HF_      TOKEN 설정")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face 토큰 (GPQA 등 gated 데이터셋 필수). 미지정 시 HF_TOKEN 환경변수 사용")
    parser.add_argument("--limit", type=int, default=None,
                        help="평가할 샘플 수 제한 (예: 100)")
    return parser.parse_args()


def main():
    args = parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = "./hf_cache"

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # gated 데이터셋(gpqa 등) 접근을 위해 HF 토큰 명시 적용
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token)
        except Exception as e:
            print(f"[WARN] HF login 실패 (gated 데이터셋 불가): {e}")

    import torch
    import json
    from lm_eval import simple_evaluate
    from lm_eval.utils import make_table
    from lm_eval.tasks import TaskManager

    # 태스크 파싱
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    exclude = [x.strip() for x in args.exclude_tasks.split(",") if x.strip()]
    if exclude:
        tasks = [t for t in tasks if t not in exclude]
        print(f"  제외 태스크: {exclude} → 실행 태스크: {tasks}")
    if not tasks:
        print("[ERROR] 실행할 태스크가 없습니다 (전부 제외되었거나 비어 있음).")
        return
    use_chat_template = args.apply_chat_template and not args.no_chat_template

    device = "cuda"

    print("=" * 60)
    print("모델 평가")
    print("=" * 60)
    print(f"  모델 경로:    {args.model_path}")
    print(f"  태스크:       {tasks}")
    print(f"  백엔드:       {args.backend}")
    print(f"  GPU:          {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}")
    print(f"  Few-shot:     {args.num_fewshot}")
    print(f"  배치 사이즈:  {args.batch_size}")
    if args.limit is not None:
        print(f"  샘플 제한:    {args.limit}")
    print(f"  채팅 템플릿:  {use_chat_template}")
    if args.include_path:
        print(f"  커스텀 경로:  {args.include_path}")
    if args.backend == "vllm":
        print(f"  GPU 메모리:   {args.gpu_mem_util:.0%}")
    print("=" * 60)

    if args.model_path.startswith((".", "/")):
        if not os.path.exists(args.model_path):
            print(f"[ERROR] 모델 경로가 존재하지 않습니다: {args.model_path}")
            return

    if args.backend == "vllm":
        model_args = (
            f"pretrained={args.model_path},"
            f"trust_remote_code=True,"
            f"dtype=auto,"
            f"tensor_parallel_size=1,"
            f"gpu_memory_utilization={args.gpu_mem_util},"
            f"enable_thinking=False"
        )
    else:  # hf
        model_args = (
            f"pretrained={args.model_path},"
            f"trust_remote_code=True,"
            f"dtype=auto,"
            f"enable_thinking=False"
        )

    task_manager = None
    if args.include_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        include_paths = [p.strip() for p in args.include_path.split(",") if p.strip()]
        # 상대경로는 스크립트 위치 기준으로 해석 (절대경로는 그대로 사용)
        resolved = [
            p if os.path.isabs(p) else os.path.normpath(os.path.join(script_dir, p))
            for p in include_paths
        ]
        task_manager = TaskManager(verbosity="INFO", include_path=resolved, include_defaults=True)

    try:
        eval_kwargs = dict(
            model=args.backend,
            model_args=model_args,
            tasks=tasks,
            device=device,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            apply_chat_template=use_chat_template,
            gen_kwargs=f"max_gen_toks={args.max_gen_toks}",
            limit=args.limit,
        )
        if task_manager is not None:
            eval_kwargs["task_manager"] = task_manager
        results = simple_evaluate(**eval_kwargs)
    except Exception as e:
        print(f"\n[ERROR] 평가 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return

    if not results:
        print("[ERROR] 평가 결과가 비어있습니다.")
        return

    # 결과 출력
    print("\n" + "=" * 60)
    print("평가 결과")
    print("=" * 60)
    print(make_table(results))

if __name__ == "__main__":
    main()

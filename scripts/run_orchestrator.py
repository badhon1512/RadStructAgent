import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import config
from pipeline import process_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Orchestrator-agent workflow for structured radiology report refinement."
    )
    parser.add_argument("--input_csv", default=config.INPUT_CSV)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--stats_csv", default=None)
    parser.add_argument("--id_column", default="StudyInstanceUid")
    parser.add_argument("--text_column", default="findings")
    parser.add_argument("--output_column", default=None)
    parser.add_argument("--model_name", default="Qwen3-14B")
    parser.add_argument("--provider", choices=["qwen", "gemma", "medgemma", "gpt"], default=None)
    parser.add_argument("--max_tool_calls", type=int, default=10)
    parser.add_argument("--max_orchestrator_steps", dest="max_tool_calls", type=int)
    parser.add_argument("--max_revision_rounds", type=int, default=3)
    parser.add_argument("--select_final", dest="select_final", action="store_true", default=True)
    parser.add_argument("--no-select_final", dest="select_final", action="store_false")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--use_vllm", dest="use_vllm", action="store_true", default=False,
                        help="Route all LLM calls to a running vLLM server instead of loading locally.")
    parser.add_argument("--openai_base_url", default=config.OPENAI_BASE_URL)
    parser.add_argument("--openai_api_key", default=config.OPENAI_API_KEY)
    parser.add_argument("--openai_model_name", default=None)
    parser.add_argument("--openai_timeout", type=float, default=600)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Parallel study workers (only effective with --use_vllm; ignored otherwise).")
    parser.add_argument("--mute_findings_judge", action="store_true", default=False,
                        help="Disable the findings judge (ablation: no clinical-faithfulness feedback).")
    parser.add_argument("--mute_anatomy_judge", action="store_true", default=False,
                        help="Disable the anatomy judge (ablation: no section-placement/duplicate feedback).")
    parser.add_argument("--blind_revision", action="store_true", default=False,
                        help="Ablation: mute both judges and revise using self-review against source only (no judge feedback).")
    parser.add_argument("--single_pass", action="store_true", default=False,
                        help="Skip the orchestrator: structure each report once (no judges, no revision).")
    return parser.parse_args()


if __name__ == "__main__":
    process_csv(parse_args())

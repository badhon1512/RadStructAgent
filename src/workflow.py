import os
import re

from constants import EMPTY_FINDINGS_FEEDBACK, EMPTY_ANATOMY_FEEDBACK
from judges import (
    normalize_findings_feedback,
    normalize_anatomy_feedback,
    has_actionable_feedback,
)

base_agent = None


def safe_slug(value: str) -> str:
    value = value.strip().replace("/", "-")
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def normalize_model_name(model_name: str, provider: str) -> str:
    if "/" in model_name:
        return model_name
    if provider == "qwen":
        return f"Qwen/{model_name}"
    if provider in {"gemma", "medgemma"}:
        return f"google/{model_name}"
    return model_name


def ensure_base_agent():
    global base_agent
    if base_agent is None:
        import agent
        base_agent = agent
    return base_agent


def infer_provider(model_name: str) -> str:
    name = model_name.lower()
    if "medgemma" in name:
        return "medgemma"
    if "gemma" in name:
        return "gemma"
    if "qwen" in name:
        return "qwen"
    if "gpt" in name or "openai" in name:
        return "gpt"
    raise ValueError(
        f"Could not infer provider from model name '{model_name}'. "
        "Use --provider qwen, gemma, medgemma, or gpt."
    )


def initialize_backend(args):
    import httpx
    import torch
    from openai import OpenAI
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

    ensure_base_agent()
    provider = args.provider or ("gpt" if getattr(args, "use_vllm", False) else infer_provider(args.model_name))
    model_name = normalize_model_name(args.model_name, provider)
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_agent.model_name = model_name
    base_agent.device = device

    if getattr(args, "use_vllm", False):
        print(f"[vllm] Backend: {model_name} @ {args.openai_base_url}")
        base_agent.model_name    = args.openai_model_name or model_name
        base_agent.call_llm      = base_agent.call_gpt
        base_agent.call_llm_chat = base_agent.call_gpt_chat
        os.environ.pop("http_proxy",  None)
        os.environ.pop("https_proxy", None)
        os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
        base_agent.client = OpenAI(
            base_url=args.openai_base_url,
            api_key=args.openai_api_key,
            http_client=httpx.Client(trust_env=False, timeout=args.openai_timeout),
        )
        return "vllm", model_name

    if provider == "medgemma":
        print(f"Using MedGemma backend: {model_name}")
        base_agent.call_llm      = base_agent.call_medgemma
        base_agent.call_llm_chat = base_agent.call_medgemma_chat
        base_agent.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            quantization_config=base_agent.bnb_config,
            torch_dtype="auto",
            device_map=device,
            attn_implementation=base_agent.get_attn_impl(),
            token=hf_token,
        )
        base_agent.processor = AutoProcessor.from_pretrained(model_name, token=hf_token)
    elif provider == "gemma":
        print(f"Using Gemma backend: {model_name}")
        base_agent.call_llm      = base_agent.call_gemma
        base_agent.call_llm_chat = base_agent.call_gemma_chat
        base_agent.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=base_agent.bnb_config,
            torch_dtype="auto",
            device_map=device,
            attn_implementation=base_agent.get_attn_impl(),
            token=hf_token,
        )
        base_agent.processor = AutoProcessor.from_pretrained(model_name, token=hf_token)
    elif provider == "qwen":
        print(f"Using Qwen backend: {model_name}")
        base_agent.call_llm      = base_agent.call_qwen3
        base_agent.call_llm_chat = base_agent.call_qwen3_chat
        base_agent.processor = AutoProcessor.from_pretrained(model_name, token=hf_token)
        base_agent.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=base_agent.bnb_config,
            torch_dtype="auto",
            attn_implementation=base_agent.get_attn_impl(),
            device_map="auto",
            token=hf_token,
        )
    elif provider == "gpt":
        print(f"Using OpenAI-compatible backend: {model_name}")
        base_agent.call_llm      = base_agent.call_gpt
        base_agent.call_llm_chat = base_agent.call_gpt_chat
        base_agent.model_name = args.openai_model_name or model_name
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
        os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
        base_agent.client = OpenAI(
            base_url=args.openai_base_url,
            api_key=args.openai_api_key,
            http_client=httpx.Client(trust_env=False, timeout=args.openai_timeout),
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return provider, model_name

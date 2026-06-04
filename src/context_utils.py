MAX_CONTEXT_RETRY_REPORT_WORDS = 1200
MAX_EMERGENCY_USER_MESSAGE_CHARS = 60000


def is_context_length_error(exc):
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "maximum context length",
            "context length",
            "input_tokens",
            "max_model_len",
            "too many tokens",
        )
    )


def compact_text(text: str, max_chars: int) -> str:
    if not isinstance(text, str) or len(text) <= max_chars:
        return text
    keep = max_chars // 2
    return (
        text[:keep].rstrip()
        + "\n\n[... middle omitted to fit model context ...]\n\n"
        + text[-keep:].lstrip()
    )


def trim_report_for_context_retry(report: str, label: str = "report") -> str:
    if not isinstance(report, str):
        return report
    words = report.split()
    if len(words) <= MAX_CONTEXT_RETRY_REPORT_WORDS:
        return report
    print(
        f"[warn] {label} has {len(words)} words after token-limit failure; "
        f"retrying with first {MAX_CONTEXT_RETRY_REPORT_WORDS} words."
    )
    return " ".join(words[:MAX_CONTEXT_RETRY_REPORT_WORDS]).strip()


def trim_candidates_for_context_retry(candidates):
    trimmed = []
    for idx, candidate in enumerate(candidates, start=1):
        report = candidate.get("report", "")
        trimmed.append({
            **candidate,
            "report": trim_report_for_context_retry(report, f"selection candidate {idx}"),
        })
    return trimmed


def call_llm_with_report_context_retry(prompt_builder, report: str, label: str):
    import workflow
    prompt = prompt_builder(report)
    try:
        return workflow.base_agent.call_llm(prompt).strip(), prompt, report
    except Exception as exc:
        if not is_context_length_error(exc):
            raise
        retry_report = trim_report_for_context_retry(report, label)
        retry_prompt = prompt_builder(retry_report)
        return workflow.base_agent.call_llm(retry_prompt).strip(), retry_prompt, retry_report


def call_selection_with_context_retry(free_text: str, candidates):
    import workflow
    import prompts
    prompt = prompts.build_revision_selection_prompt(free_text, candidates)
    try:
        return workflow.base_agent.call_llm(prompt).strip(), prompt, candidates
    except Exception as exc:
        if not is_context_length_error(exc):
            raise
        print("[warn] Selection prompt exceeded model limit; retrying with trimmed candidate reports.")
        retry_candidates = trim_candidates_for_context_retry(candidates)
        retry_prompt = prompts.build_revision_selection_prompt(free_text, retry_candidates)
        return workflow.base_agent.call_llm(retry_prompt).strip(), retry_prompt, retry_candidates

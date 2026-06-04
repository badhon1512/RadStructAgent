from constants import EMPTY_FINDINGS_FEEDBACK, EMPTY_ANATOMY_FEEDBACK
from context_utils import is_context_length_error, trim_report_for_context_retry


def _finding_text(entry) -> str:
    if isinstance(entry, dict):
        return (entry.get("finding") or "").strip().rstrip(".")
    return str(entry).strip().rstrip(".")


def normalize_findings_feedback(feedback):
    if not isinstance(feedback, dict):
        return EMPTY_FINDINGS_FEEDBACK.copy()
    missing     = feedback.get("missing_findings") or []
    unsupported = feedback.get("unsupported_findings") or []
    missing_texts     = {_finding_text(f) for f in missing}
    unsupported_texts = {_finding_text(f) for f in unsupported}
    contradicted = missing_texts & unsupported_texts
    if contradicted:
        missing     = [f for f in missing     if _finding_text(f) not in contradicted]
        unsupported = [f for f in unsupported if _finding_text(f) not in contradicted]
    return {"missing_findings": missing, "unsupported_findings": unsupported}


def normalize_anatomy_feedback(feedback):
    if not isinstance(feedback, dict):
        return EMPTY_ANATOMY_FEEDBACK.copy()
    wrong = feedback.get("wrong_section_findings") or []
    wrong = [
        f for f in wrong
        if not (isinstance(f, dict) and f.get("current_section") == f.get("correct_section"))
    ]
    return {
        "wrong_section_findings": wrong,
        "duplicate_findings": feedback.get("duplicate_findings") or [],
    }


def has_actionable_feedback(findings_feedback, anatomy_feedback):
    return bool(
        findings_feedback.get("missing_findings")
        or findings_feedback.get("unsupported_findings")
        or anatomy_feedback.get("wrong_section_findings")
        or anatomy_feedback.get("duplicate_findings")
    )


def call_json_judge(prompt: str, expected: dict):
    import workflow
    response = workflow.base_agent.call_llm(prompt)
    parsed = workflow.base_agent.extract_json(response)
    if parsed:
        return parsed, response
    return expected.copy(), response


def call_findings_judge_with_context_retry(free_text: str, report: str):
    import workflow
    prompt = workflow.base_agent.build_findings_judge_prompt(free_text, report)
    try:
        feedback, raw = call_json_judge(prompt, EMPTY_FINDINGS_FEEDBACK)
        return feedback, raw, prompt, report
    except Exception as exc:
        if not is_context_length_error(exc):
            raise
        retry_report = trim_report_for_context_retry(report, "findings judge report")
        retry_prompt = workflow.base_agent.build_findings_judge_prompt(free_text, retry_report)
        feedback, raw = call_json_judge(retry_prompt, EMPTY_FINDINGS_FEEDBACK)
        return feedback, raw, retry_prompt, retry_report


def call_anatomy_judge_with_context_retry(report: str):
    import workflow
    prompt = workflow.base_agent.build_anatomy_duplication_judge_prompt(report)
    try:
        feedback, raw = call_json_judge(prompt, EMPTY_ANATOMY_FEEDBACK)
        return feedback, raw, prompt, report
    except Exception as exc:
        if not is_context_length_error(exc):
            raise
        retry_report = trim_report_for_context_retry(report, "anatomy judge report")
        retry_prompt = workflow.base_agent.build_anatomy_duplication_judge_prompt(retry_report)
        feedback, raw = call_json_judge(retry_prompt, EMPTY_ANATOMY_FEEDBACK)
        return feedback, raw, retry_prompt, retry_report

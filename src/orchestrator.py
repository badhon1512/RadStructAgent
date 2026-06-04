import json
import re

import workflow
from prompts import (
    build_orchestrator_system_prompt,
    build_orchestrator_user_message,
)
from context_utils import (
    MAX_EMERGENCY_USER_MESSAGE_CHARS,
    compact_text,
    is_context_length_error,
    call_llm_with_report_context_retry,
    call_selection_with_context_retry,
)
from judges import (
    call_findings_judge_with_context_retry,
    call_anatomy_judge_with_context_retry,
)
from logging_utils import (
    _sep,
    _print_report,
    _print_decision,
    _print_findings_feedback,
    _print_anatomy_feedback,
    _print_raw,
    _print_summary,
)


VALID_ACTIONS = {
    "run_findings_judge",
    "run_anatomy_judge",
    "revise_report",
    "select_best_candidate",
    "finalize_report",
}

REVISION_LABEL_RE = re.compile(r"^orchestrator_revision_(\d+)\.?$", re.IGNORECASE)


def resolve_selected_report(selected: str, candidates) -> str:
    if not isinstance(selected, str):
        return ""
    text = selected.strip()

    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            report = parsed.get("report")
            if isinstance(report, str) and report.strip():
                return report.strip()
            stage = parsed.get("stage")
            if isinstance(stage, str):
                text = stage.strip()
        except json.JSONDecodeError:
            pass

    label = text.strip().strip('`').strip().rstrip(".")
    for candidate in candidates:
        if label == candidate.get("stage"):
            return candidate.get("report", text)

    match = REVISION_LABEL_RE.match(label)
    if match:
        stage = f"orchestrator_revision_{int(match.group(1))}"
        for candidate in candidates:
            if candidate.get("stage") == stage:
                return candidate.get("report", text)

    if label.lower() == "initial" and candidates:
        return candidates[0].get("report", text)

    return text


def parse_orchestrator_action(response):
    parsed = workflow.base_agent.extract_json(response)
    if not isinstance(parsed, dict):
        return "finalize_report", "Invalid action returned; finalizing conservatively.", "uncertain"
    action    = parsed.get("action")
    reason    = parsed.get("reason", "")
    certainty = parsed.get("certainty", "uncertain")
    if action not in VALID_ACTIONS:
        return "finalize_report", "Invalid action returned; finalizing conservatively.", "uncertain"
    return action, reason, certainty


def feedback_has_issues(findings_feedback, anatomy_feedback):
    return workflow.has_actionable_feedback(findings_feedback, anatomy_feedback)


def run_orchestrator_agent_pipeline(
    free_text, max_tool_calls, max_revision_rounds, select_final,
    mute_findings_judge: bool = False, mute_anatomy_judge: bool = False,
    blind_revision: bool = False,
):
    prompt_log = []

    ablation_label = (
        "no_findings_no_anatomy_blind_revision" if (mute_findings_judge and mute_anatomy_judge and blind_revision)
        else "no_findings_no_anatomy" if (mute_findings_judge and mute_anatomy_judge)
        else "no_findings" if mute_findings_judge
        else "no_anatomy"  if mute_anatomy_judge
        else "all_judges"
    )
    if ablation_label != "all_judges":
        print(f"[ablation] Muted judges: {ablation_label}")

    structuring_prompt = workflow.base_agent.build_structuring_prompt(free_text)
    current_report = workflow.base_agent.call_llm(structuring_prompt).strip()
    prompt_log.append({
        "step": 0, "agent": "structuring", "tool_call": 0,
        "prompt": structuring_prompt, "response": current_report,
    })

    candidates = [{"stage": "initial", "report": current_report}]
    findings_feedback = workflow.EMPTY_FINDINGS_FEEDBACK.copy()
    anatomy_feedback  = workflow.EMPTY_ANATOMY_FEEDBACK.copy()
    tool_calls_used   = 0
    revision_rounds_used = 0
    findings_judge_calls = 0
    anatomy_judge_calls  = 0
    selection_calls      = 0
    fallback_count       = 0
    stop_reason          = "max_tool_calls"
    trace_events         = []
    findings_ran_since_revision = False
    anatomy_ran_since_revision  = False

    findings_feedback_history = []
    anatomy_feedback_history  = []
    certainty_sequence        = []
    revision_reports          = []

    _sep("Initial Structured Report")
    _print_report(current_report)

    messages = [{"role": "system", "content": build_orchestrator_system_prompt(
        mute_findings=mute_findings_judge, mute_anatomy=mute_anatomy_judge,
        blind_revision=blind_revision,
    )}]

    for tool_call in range(1, max_tool_calls + 1):
        tool_calls_used = tool_call
        state = {
            "tool_call": tool_call,
            "revision_rounds_used": revision_rounds_used,
            "findings_judge_calls": findings_judge_calls,
            "anatomy_judge_calls": anatomy_judge_calls,
            "selection_calls": selection_calls,
            "candidate_count": len(candidates),
            "has_actionable_feedback": feedback_has_issues(findings_feedback, anatomy_feedback),
            "findings_judge_ran_since_last_revision": findings_ran_since_revision,
            "anatomy_judge_ran_since_last_revision": anatomy_ran_since_revision,
        }
        user_msg = build_orchestrator_user_message(
            free_text=free_text,
            current_report=current_report,
            findings_feedback=findings_feedback,
            anatomy_feedback=anatomy_feedback,
            candidates=candidates,
            state=state,
            max_tool_calls=max_tool_calls,
            max_revision_rounds=max_revision_rounds,
            mute_findings=mute_findings_judge,
            mute_anatomy=mute_anatomy_judge,
        )
        turn_messages = [*messages, {"role": "user", "content": user_msg}]
        try:
            raw_response = workflow.base_agent.call_llm_chat(turn_messages)
        except Exception as exc:
            if not is_context_length_error(exc):
                raise
            print(
                "[warn] Orchestrator context exceeded model limit; "
                "retrying with system prompt and current state only."
            )
            turn_messages = [messages[0], {"role": "user", "content": user_msg}]
            try:
                raw_response = workflow.base_agent.call_llm_chat(turn_messages)
            except Exception as current_exc:
                if not is_context_length_error(current_exc):
                    raise
                print("[warn] Current state still exceeded context; truncating current state message.")
                compact_user_msg = compact_text(user_msg, MAX_EMERGENCY_USER_MESSAGE_CHARS)
                turn_messages = [messages[0], {"role": "user", "content": compact_user_msg}]
                raw_response = workflow.base_agent.call_llm_chat(turn_messages)

        messages = [*turn_messages, {"role": "assistant", "content": raw_response}]

        requested_action, requested_reason, certainty = parse_orchestrator_action(raw_response)
        action, reason = requested_action, requested_reason

        if action == "run_findings_judge" and mute_findings_judge:
            action = "revise_report" if blind_revision else "finalize_report"
            reason = "No judge feedback available; revising against source." if blind_revision else "Findings judge is muted in this ablation run."
        elif action == "run_anatomy_judge" and mute_anatomy_judge:
            action = "revise_report" if blind_revision else "finalize_report"
            reason = "No judge feedback available; revising against source." if blind_revision else "Anatomy judge is muted in this ablation run."

        if action == "revise_report" and revision_rounds_used >= max_revision_rounds:
            action = "select_best_candidate" if select_final and len(candidates) > 1 else "finalize_report"
            reason = "Revision budget exhausted."
        elif action == "select_best_candidate" and (not select_final or len(candidates) <= 1):
            action = "finalize_report"
            reason = "Selection requested without multiple candidates or selection disabled."

        used_fallback = action != requested_action
        if used_fallback:
            fallback_count += 1

        certainty_sequence.append(certainty)

        event = {
            "tool_call": tool_call,
            "requested_action": requested_action,
            "action": action,
            "used_fallback": used_fallback,
            "reason": reason,
            "certainty": certainty,
            "state_before": state,
            "raw_response": raw_response,
        }
        trace_events.append(event)

        prompt_log.append({
            "step": tool_call, "agent": "orchestrator", "tool_call": tool_call,
            "messages_sent": turn_messages,
            "response": raw_response,
            "parsed_action": action,
            "requested_action": requested_action,
            "reason": reason,
            "certainty": certainty,
            "used_fallback": used_fallback,
            "state_before": state,
            "source_free_text": free_text,
            "current_report_at_decision": current_report,
            "findings_feedback_at_decision": findings_feedback,
            "anatomy_feedback_at_decision": anatomy_feedback,
        })

        _print_decision(tool_call, max_tool_calls, requested_action, action, certainty, reason, used_fallback)
        _print_raw("Orchestrator", raw_response)

        if action == "run_findings_judge":
            feedback, findings_raw, judge_prompt, _judge_input_report = call_findings_judge_with_context_retry(
                free_text, current_report,
            )
            findings_feedback = workflow.normalize_findings_feedback(feedback)
            findings_judge_calls += 1
            findings_ran_since_revision = True
            findings_feedback_history.append({
                "tool_call": tool_call,
                "revision_round": revision_rounds_used,
                "feedback": findings_feedback,
                "raw_response": findings_raw,
            })
            prompt_log.append({
                "step": tool_call, "agent": "findings_judge", "tool_call": tool_call,
                "prompt": judge_prompt,
                "response": findings_raw,
                "parsed": findings_feedback,
            })
            _print_findings_feedback(findings_feedback)
            _print_raw("Findings judge", findings_raw)

        elif action == "run_anatomy_judge":
            feedback, anatomy_raw, judge_prompt, _judge_input_report = call_anatomy_judge_with_context_retry(current_report)
            anatomy_feedback = workflow.normalize_anatomy_feedback(feedback)
            anatomy_judge_calls += 1
            anatomy_ran_since_revision = True
            anatomy_feedback_history.append({
                "tool_call": tool_call,
                "revision_round": revision_rounds_used,
                "feedback": anatomy_feedback,
                "raw_response": anatomy_raw,
            })
            prompt_log.append({
                "step": tool_call, "agent": "anatomy_judge", "tool_call": tool_call,
                "prompt": judge_prompt,
                "response": anatomy_raw,
                "parsed": anatomy_feedback,
            })
            _print_anatomy_feedback(anatomy_feedback)
            _print_raw("Anatomy judge", anatomy_raw)

        elif action == "revise_report":
            # In blind_revision mode both revisors always run with empty feedback, so the
            # model self-reviews against the source text using the same revision prompts.
            has_findings = blind_revision or bool(
                findings_feedback.get("missing_findings") or findings_feedback.get("unsupported_findings")
            )
            has_anatomy = blind_revision or bool(
                anatomy_feedback.get("wrong_section_findings") or anatomy_feedback.get("duplicate_findings")
            )
            intermediate = current_report
            if has_findings:
                intermediate, findings_rev_prompt, _findings_input_report = call_llm_with_report_context_retry(
                    lambda report: workflow.base_agent.build_findings_revision_prompt(
                        free_text, report, findings_feedback,
                    ),
                    current_report,
                    "findings revision input report",
                )
                prompt_log.append({
                    "step": tool_call, "agent": "revision_findings", "tool_call": tool_call,
                    "prompt": findings_rev_prompt,
                    "response": intermediate,
                })
                _sep("After Findings Revision")
                _print_report(intermediate)
            if has_anatomy:
                intermediate, anatomy_rev_prompt, _anatomy_input_report = call_llm_with_report_context_retry(
                    lambda report: workflow.base_agent.build_anatomy_revision_prompt(
                        report, anatomy_feedback,
                    ),
                    intermediate,
                    "anatomy revision input report",
                )
                prompt_log.append({
                    "step": tool_call, "agent": "revision_anatomy", "tool_call": tool_call,
                    "prompt": anatomy_rev_prompt,
                    "response": intermediate,
                })
                _sep("After Anatomy Revision")
                _print_report(intermediate)
            revised_report = intermediate
            if not has_findings and not has_anatomy:
                revised_report, fallback_prompt, _revision_input_report = call_llm_with_report_context_retry(
                    lambda report: workflow.base_agent.build_revision_prompt(
                        free_text, report, findings_feedback, anatomy_feedback,
                    ),
                    current_report,
                    "fallback revision input report",
                )
                prompt_log.append({
                    "step": tool_call, "agent": "revision", "tool_call": tool_call,
                    "prompt": fallback_prompt,
                    "response": revised_report,
                })
            if not revised_report or revised_report == current_report:
                print("  [warn] Revision produced no change -- stopping.")
                stop_reason = "revision_no_change"
                break
            revision_rounds_used += 1
            current_report = revised_report
            candidates.append({"stage": f"orchestrator_revision_{revision_rounds_used}", "report": current_report})
            revision_reports.append({"round": revision_rounds_used, "report": current_report})
            findings_feedback = workflow.EMPTY_FINDINGS_FEEDBACK.copy()
            anatomy_feedback  = workflow.EMPTY_ANATOMY_FEEDBACK.copy()
            findings_ran_since_revision = False
            anatomy_ran_since_revision  = False
            _sep(f"Revised Report (round {revision_rounds_used})")
            _print_report(current_report)

        elif action == "select_best_candidate":
            selected, selection_prompt, selection_candidates = call_selection_with_context_retry(
                free_text, candidates,
            )
            prompt_log.append({
                "step": tool_call, "agent": "selection", "tool_call": tool_call,
                "prompt": selection_prompt,
                "response": selected,
                "candidates_count": len(selection_candidates),
            })
            if selected:
                current_report = resolve_selected_report(selected, selection_candidates)
                selection_calls += 1
                stop_reason = "selected_final"
                _sep("Selected Best Candidate")
                _print_report(current_report)
            break

        elif action == "finalize_report":
            stop_reason = "finalize_report"
            break

    if stop_reason == "max_tool_calls" and select_final and len(candidates) > 1:
        print("\n  [info] Max tool calls reached -- selecting best candidate.")
        selected, selection_prompt, selection_candidates = call_selection_with_context_retry(
            free_text, candidates,
        )
        prompt_log.append({
            "step": tool_calls_used, "agent": "selection_budget_exhausted", "tool_call": tool_calls_used,
            "prompt": selection_prompt,
            "response": selected,
            "candidates_count": len(selection_candidates),
        })
        if selected:
            current_report = resolve_selected_report(selected, selection_candidates)
            selection_calls += 1
            stop_reason = "max_tool_calls_selected_final"
            _sep("Selected Best Candidate (budget exhausted)")
            _print_report(current_report)

    metadata = {
        "tool_calls_used":       tool_calls_used,
        "revision_rounds_used":  revision_rounds_used,
        "findings_judge_calls":  findings_judge_calls,
        "anatomy_judge_calls":   anatomy_judge_calls,
        "selection_calls":       selection_calls,
        "candidate_count":       len(candidates),
        "fallback_count":        fallback_count,
        "stop_reason":           stop_reason,
        "ablation_muted_judges": ablation_label,
        "action_sequence":           "|".join(e["action"]            for e in trace_events),
        "requested_action_sequence": "|".join(e["requested_action"]  for e in trace_events),
        "certainty_sequence":        "|".join(certainty_sequence),
        "initial_report":    candidates[0]["report"],
        "final_report":      current_report,
        "candidates_json":   json.dumps(candidates, ensure_ascii=False),
        "revision_reports_json": json.dumps(revision_reports, ensure_ascii=False),
        "findings_feedback_history_json": json.dumps(findings_feedback_history, ensure_ascii=False),
        "anatomy_feedback_history_json":  json.dumps(anatomy_feedback_history,  ensure_ascii=False),
        "trace_json": json.dumps(trace_events, ensure_ascii=False),
    }
    for action_name in sorted(VALID_ACTIONS):
        metadata[f"action_count_{action_name}"] = sum(1 for e in trace_events if e["action"] == action_name)

    _sep("Final Report")
    _print_report(current_report)
    _print_summary(metadata, tool_calls_used, max_tool_calls, revision_rounds_used, max_revision_rounds)
    return current_report, metadata, prompt_log

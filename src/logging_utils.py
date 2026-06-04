_W = 60


def _sep(title=""):
    if title:
        pad = max(0, _W - len(title) - 5)
        print(f"\n{'=' * 4} {title} {'=' * pad}")
    else:
        print("=" * _W)


def _print_report(report: str):
    print(report)
    print("-" * _W)


def _print_decision(tool_call, max_tool_calls, requested, final, certainty, reason, used_fallback):
    tag = "[warn]" if used_fallback else "[call]"
    override = f"  [overridden -> {final}]" if used_fallback else ""
    print(f"\n{tag} [{tool_call}/{max_tool_calls}] {requested}{override}")
    print(f"  Certainty : {certainty}")
    print(f"  Reason    : {reason}")


def _print_findings_feedback(fb):
    missing     = fb.get("missing_findings") or []
    unsupported = fb.get("unsupported_findings") or []
    print("  Findings Judge ->")
    if missing:
        print(f"    [x] Missing ({len(missing)}):")
        for f in missing:
            print(f"      - {f}")
    else:
        print("    [ok] No missing findings")
    if unsupported:
        print(f"    [x] Unsupported ({len(unsupported)}):")
        for f in unsupported:
            print(f"      - {f}")
    else:
        print("    [ok] No unsupported findings")


def _print_anatomy_feedback(fb):
    wrong  = fb.get("wrong_section_findings") or []
    dupes  = fb.get("duplicate_findings") or []
    print("  Anatomy Judge ->")
    if wrong:
        print(f"    [x] Wrong section ({len(wrong)}):")
        for f in wrong:
            print(f"      - {f.get('finding', f)[:100]}")
    else:
        print("    [ok] All findings in correct sections")
    if dupes:
        print(f"    [x] Duplicates ({len(dupes)}):")
        for f in dupes:
            print(f"      - {f.get('finding', f)[:100]}")
    else:
        print("    [ok] No duplicate findings")


def _print_raw(label, text, max_chars=600):
    preview = text[:max_chars] + ("..." if len(text) > max_chars else "")
    print(f"  {label} raw response:")
    for line in preview.splitlines():
        print(f"    {line}")


def _print_summary(metadata, tool_calls_used, max_tool_calls, revision_rounds_used, max_revision_rounds):
    _sep("Summary")
    seq = " -> ".join(metadata["action_sequence"].split("|"))
    print(f"  Stop reason    : {metadata['stop_reason']}")
    print(f"  Tool calls     : {tool_calls_used} / {max_tool_calls}")
    print(f"  Revisions      : {revision_rounds_used} / {max_revision_rounds}")
    print(f"  Findings judge : {metadata['findings_judge_calls']}x  |  Anatomy judge : {metadata['anatomy_judge_calls']}x")
    print(f"  Candidates     : {metadata['candidate_count']}  |  Fallbacks : {metadata['fallback_count']}")
    print(f"  Action sequence:")
    print(f"    {seq}")
    _sep()

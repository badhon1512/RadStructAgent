import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import workflow


def atomic_write_csv(df, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_csv.with_suffix(output_csv.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(output_csv)


def load_existing_output(output_csv: Path, id_column: str):
    import pandas as pd
    if not output_csv.exists():
        return pd.DataFrame(), set()
    df = pd.read_csv(output_csv)
    completed = set()
    if id_column in df.columns and "status" in df.columns:
        completed = set(df.loc[df["status"] == "ok", id_column].astype(str))
    return df, completed


def build_study_json(study_id, status, prompt_log):
    steps = []
    for entry in prompt_log:
        agent = entry["agent"]
        step  = entry["step"]

        if agent == "orchestrator":
            steps.append({
                "step": step,
                "agent": agent,
                "action_requested": entry.get("requested_action"),
                "action_executed":  entry.get("parsed_action"),
                "certainty":        entry.get("certainty"),
                "reason":           entry.get("reason"),
                "used_fallback":    entry.get("used_fallback"),
                "source_free_text":              entry.get("source_free_text", ""),
                "current_report_at_decision":    entry.get("current_report_at_decision", ""),
                "findings_feedback_at_decision": entry.get("findings_feedback_at_decision", {}),
                "anatomy_feedback_at_decision":  entry.get("anatomy_feedback_at_decision", {}),
                "state_before":  entry.get("state_before", {}),
                "conversation":  [{"role": m["role"], "content": m["content"]}
                                  for m in entry.get("messages_sent", [])],
                "response": entry.get("response"),
            })
        elif agent in ("findings_judge", "anatomy_judge"):
            steps.append({
                "step": step,
                "agent": agent,
                "prompt":          entry.get("prompt"),
                "response":        entry.get("response"),
                "parsed_feedback": entry.get("parsed"),
            })
        elif agent == "revision":
            steps.append({
                "step":     step,
                "agent":    agent,
                "prompt":   entry.get("prompt"),
                "response": entry.get("response"),
            })
        elif agent in ("structuring",):
            steps.append({
                "step":     step,
                "agent":    agent,
                "prompt":   entry.get("prompt"),
                "response": entry.get("response"),
            })
        else:
            steps.append({
                "step":             step,
                "agent":            agent,
                "candidates_count": entry.get("candidates_count"),
                "prompt":           entry.get("prompt"),
                "response":         entry.get("response"),
            })

    return {
        "study_id":    study_id,
        "status":      status,
        "total_steps": len(steps),
        "steps":       steps,
    }


def run_with_retries(free_text, args):
    import orchestrator
    last_error = ""
    for attempt in range(1, args.max_retries + 1):
        try:
            if getattr(args, "single_pass", False):
                report = workflow.base_agent.run_pipeline(free_text, is_agent=False).strip()
                output = workflow.base_agent.extract_findings_section(report)
                prompt_log = [{
                    "step": 0, "agent": "structuring", "tool_call": 0,
                    "prompt": workflow.base_agent.build_structuring_prompt(free_text),
                    "response": report,
                }]
                return output, {"stop_reason": "single_pass"}, prompt_log, "", attempt
            output, metadata, prompt_log = orchestrator.run_orchestrator_agent_pipeline(
                free_text=free_text,
                max_tool_calls=args.max_tool_calls,
                max_revision_rounds=args.max_revision_rounds,
                select_final=args.select_final,
                mute_findings_judge=getattr(args, "mute_findings_judge", False),
                mute_anatomy_judge=getattr(args, "mute_anatomy_judge", False),
                blind_revision=getattr(args, "blind_revision", False),
            )
            return workflow.base_agent.extract_findings_section(output), metadata, prompt_log, "", attempt
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"[warn] Attempt {attempt}/{args.max_retries} failed: {last_error}")
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            if attempt < args.max_retries and args.retry_sleep > 0:
                time.sleep(args.retry_sleep)
    return "", {}, [], last_error, args.max_retries


def process_csv(args):
    import pandas as pd

    provider, resolved_model_name = workflow.initialize_backend(args)
    input_csv = Path(args.input_csv)

    single_pass = getattr(args, "single_pass", False)
    workflow_tag = "single_pass" if single_pass else "orchestrator_agent"

    ablation_parts = []
    if getattr(args, "mute_findings_judge", False):
        ablation_parts.append("no_findings")
    if getattr(args, "mute_anatomy_judge", False):
        ablation_parts.append("no_anatomy")
    if getattr(args, "blind_revision", False):
        ablation_parts.append("blind_revision")
    ablation_suffix = ("_ablation_" + "_".join(ablation_parts)) if ablation_parts else ""

    output_csv = Path(args.output_csv) if args.output_csv else Path(
        f"{workflow.safe_slug(args.model_name)}-{workflow_tag}{ablation_suffix}.csv"
    )
    stats_csv = Path(args.stats_csv) if args.stats_csv else output_csv.with_name(
        output_csv.stem + "_stat" + output_csv.suffix
    )
    prompts_jsonl = output_csv.with_name(output_csv.stem + "_prompts.jsonl")
    prompts_json  = output_csv.with_name(output_csv.stem + "_study_prompts.json")
    gen_column = args.output_column or f"{args.model_name}-{workflow_tag}{ablation_suffix}"

    df = pd.read_csv(input_csv)
    required_columns = {args.id_column, args.text_column}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    if args.start:
        df = df.iloc[args.start:]
    if args.limit:
        df = df.head(args.limit)

    if args.resume:
        existing_df, completed_ids = load_existing_output(output_csv, args.id_column)
        existing_stats_df, _       = load_existing_output(stats_csv, args.id_column)

        if not existing_df.empty and "status" in existing_df.columns:
            ok_mask   = existing_df["status"].fillna("").astype(str).str.lower().eq("ok")
            retry_ids = set(existing_df.loc[~ok_mask, args.id_column].astype(str))
            records   = existing_df.loc[ok_mask].to_dict("records")
            completed_ids = set(existing_df.loc[ok_mask, args.id_column].astype(str))
        else:
            retry_ids = set()
            records   = existing_df.to_dict("records") if not existing_df.empty else []
            completed_ids = set(str(r[args.id_column]) for r in records if args.id_column in r)

        if not existing_stats_df.empty and "status" in existing_stats_df.columns:
            stat_ok_mask  = existing_stats_df["status"].fillna("").astype(str).str.lower().eq("ok")
            stats_records = existing_stats_df.loc[stat_ok_mask].to_dict("records")
        else:
            stats_records = existing_stats_df.to_dict("records") if not existing_stats_df.empty else []

        prompts_dict = json.loads(prompts_json.read_text(encoding="utf-8")) if prompts_json.exists() else {}
        for rid in retry_ids:
            prompts_dict.pop(rid, None)

        if retry_ids and prompts_jsonl.exists():
            kept = [
                line for line in prompts_jsonl.read_text(encoding="utf-8").splitlines()
                if line.strip() and str(json.loads(line).get(args.id_column)) not in retry_ids
            ]
            prompts_jsonl.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

        jsonl_mode = "a"
        if retry_ids:
            print(f"[resume] Preserved {len(completed_ids)} ok rows; retrying {len(retry_ids)} error/incomplete rows.")
    else:
        records, stats_records, completed_ids = [], [], set()
        prompts_dict = {}
        jsonl_mode = "w"
        for f in (output_csv, stats_csv, prompts_jsonl, prompts_json):
            if f.exists():
                f.unlink()
                print(f"[info] Cleared existing file: {f}")

    num_workers = args.num_workers if getattr(args, "use_vllm", False) else 1
    if num_workers > 1 and not getattr(args, "use_vllm", False):
        print("[warn] --num_workers > 1 only works with --use_vllm; falling back to 1 worker.")
        num_workers = 1

    total = len(df)
    print(f"Input: {input_csv}")
    print(f"Output: {output_csv}")
    print(f"Stats CSV: {stats_csv}")
    print(f"Prompts JSONL: {prompts_jsonl}")
    print(f"Study prompts JSON: {prompts_json}")
    print(f"Backend: {provider} ({resolved_model_name})")
    print(f"Workers: {num_workers} {'(parallel — vLLM)' if num_workers > 1 else '(sequential)'}")
    print(f"Workflow: {workflow_tag}")
    print(f"Max tool calls: {args.max_tool_calls}")
    print(f"Max revision rounds: {args.max_revision_rounds}")
    print(f"Select final: {args.select_final}")
    print(f"Rows selected: {total}")
    print(f"Resume: {'on — appending to existing files' if args.resume else 'off — starting fresh'}")
    print(f"Ablation — muted judges: {ablation_suffix[len('_ablation_'):] if ablation_suffix else 'none (all active)'}")

    for pos, row in [(p, r) for p, r in enumerate(df.iterrows(), 1) if args.resume and str(r[1][args.id_column]) in completed_ids]:
        print(f"[{pos}/{total}] Skipping completed: {row[1][args.id_column]}")

    save_lock = threading.Lock()
    processed_since_save = 0

    def process_one(position, row):
        nonlocal processed_since_save, jsonl_mode
        study_id = str(row[args.id_column])
        free_text = "" if pd.isna(row[args.text_column]) else str(row[args.text_column])
        print(f"[{position}/{total}] Processing: {study_id}")
        print(f"  Free text preview: {free_text}")
        start_time = time.time()
        output, metadata, prompt_log, error, attempts = run_with_retries(free_text, args)
        elapsed = round(time.time() - start_time, 3)
        status = "ok" if not error else "error"

        record = {
            args.id_column: study_id,
            "ref": free_text,
            gen_column: output,
            "status": status,
            "error": error,
            "attempts": attempts,
            "elapsed_sec": elapsed,
        }
        stats_record = {args.id_column: study_id, "status": status, "error": error,
                        "attempts": attempts, "elapsed_sec": elapsed}
        stats_record.update(metadata)

        study_json = build_study_json(study_id, status, prompt_log)

        with save_lock:
            records.append(record)
            stats_records.append(stats_record)
            prompts_dict[study_id] = study_json
            tmp = prompts_json.with_suffix(".tmp")
            tmp.write_text(json.dumps(prompts_dict, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(prompts_json)
            with prompts_jsonl.open(jsonl_mode, encoding="utf-8") as pf:
                jsonl_mode = "a"
                pf.write(json.dumps({args.id_column: study_id, "status": status,
                                     "prompt_log": prompt_log}, ensure_ascii=False) + "\n")
            print(f"[row {position}/{total}] {study_id} — status={status} "
                  f"calls={metadata.get('tool_calls_used',0)} "
                  f"revisions={metadata.get('revision_rounds_used',0)} "
                  f"stop={metadata.get('stop_reason','')} elapsed={elapsed}s")
            nonlocal processed_since_save
            processed_since_save += 1
            if processed_since_save >= args.save_every:
                atomic_write_csv(pd.DataFrame(records), output_csv)
                atomic_write_csv(pd.DataFrame(stats_records), stats_csv)
                processed_since_save = 0
                print(f"[info] Saved progress -> {output_csv}")

    pending_rows = [
        (i + 1, row)
        for i, (_, row) in enumerate(df.iterrows())
        if not (args.resume and str(row[args.id_column]) in completed_ids)
    ]

    if num_workers == 1:
        for position, row in pending_rows:
            process_one(position, row)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_one, pos, row): pos for pos, row in pending_rows}
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc:
                    print(f"[error] Worker failed: {exc}")

    atomic_write_csv(pd.DataFrame(records), output_csv)
    atomic_write_csv(pd.DataFrame(stats_records), stats_csv)
    print(f"[done] Wrote {len(records)} rows to {output_csv}")
    print(f"[done] Wrote {len(stats_records)} stats rows to {stats_csv}")
    print(f"[done] Prompt log -> {prompts_jsonl}")
    print(f"[done] Study prompts JSON -> {prompts_json}")

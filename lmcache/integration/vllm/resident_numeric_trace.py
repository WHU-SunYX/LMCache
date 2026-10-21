# SPDX-License-Identifier: Apache-2.0
"""Opt-in fingerprints of the exact Q/K/V consumed by sparse attention.

Disabled unless AISSD_RESIDENT_NUMERIC_TRACE_DIR is set. GPU tensors are only
read; physical page IDs are deliberately excluded from numeric comparisons.
This module's comparison CLI does not require torch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

_REQUESTS = {}
_OPENED_PATHS = set()


def _signature(tensor):
    import torch
    cpu = tensor.detach().contiguous().cpu()
    raw = cpu.view(torch.uint8).numpy().tobytes()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(raw).hexdigest()}


def before_attention(context, layer_id, query, kv_cache):
    root = os.environ.get("AISSD_RESIDENT_NUMERIC_TRACE_DIR", "").strip()
    if not root or not context.get("host_sparse_decode_ready", False):
        return None
    import torch
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Resident numeric tracing requires eager attention")
    if int(context.get("host_active_reqs", 0)) != 1:
        raise RuntimeError("Resident numeric tracing requires one active request")
    req_id = str(context["req_ids"][0])
    max_requests = int(os.environ.get("AISSD_RESIDENT_NUMERIC_TRACE_MAX_REQUESTS", "2"))
    max_steps = int(os.environ.get("AISSD_RESIDENT_NUMERIC_TRACE_MAX_STEPS", "2"))
    if min(max_requests, max_steps) < 1:
        raise ValueError("Numeric trace limits must be positive")
    if req_id not in _REQUESTS:
        if len(_REQUESTS) >= max_requests:
            return None
        _REQUESTS[req_id] = {"ordinal": len(_REQUESTS) + 1, "steps": {}}
    request = _REQUESTS[req_id]
    local_tokens = int(context["local_token_lens"][0].item())
    # A request's local tail grows by one per ordinary decode forward. All
    # layers in that forward share this key, independently of context rebuilds.
    if local_tokens not in request["steps"]:
        request["steps"][local_tokens] = len(request["steps"]) + 1
    step = request["steps"][local_tokens]
    if step > max_steps:
        return None
    local_blocks = int(context["local_block_lens"][0].item())
    seq_len = int(context["fa_seq_lens"][0].item())
    block_count = int(context["fa_block_lens"][0].item())
    physical = context["fa_block_table"][0, :block_count].tolist()
    count = int(context["aissd_rpc_selected_chunk_lens"][0].item())
    selected_ids = context["aissd_rpc_selected_chunk_ids"][0, :count].tolist()
    candidate_count = int(context["aissd_candidate_count"][0].item())
    ids = context["aissd_candidate_chunk_ids"][0, :candidate_count].tolist()
    starts = context["aissd_candidate_token_start"][0, :candidate_count].tolist()
    ends = context["aissd_candidate_token_end"][0, :candidate_count].tolist()
    ranges = {int(i): (int(a), int(b)) for i, a, b in zip(ids, starts, ends)}
    selected = [{"candidate_id": int(i), "token_start": ranges[int(i)][0],
                 "token_end": ranges[int(i)][1]} for i in selected_ids]
    remote_tokens = sum(ch["token_end"] - ch["token_start"] for ch in selected)
    if remote_tokens + local_tokens != seq_len:
        raise RuntimeError("Numeric trace: selected/local lengths disagree with FA")
    if kv_cache.ndim != 5:
        raise RuntimeError("Numeric trace requires a five-dimensional paged KV cache")
    paged = kv_cache if kv_cache.shape[1] == 2 else kv_cache.transpose(0, 1)
    if paged.shape[1] != 2 or any(int(b) <= 0 for b in physical):
        raise RuntimeError("Numeric trace: invalid KV layout or null physical page")
    block_size = int(paged.shape[2])
    if block_count != (seq_len + block_size - 1) // block_size:
        raise RuntimeError("Numeric trace: invalid compact FA block count")
    indices = torch.tensor(physical, dtype=torch.long, device=kv_cache.device)
    # Gather in FA's actual order, then trim unused slots of the last block.
    # [blocks,2,BS,H,D] -> [2,tokens,H,D]. No cache or metadata is modified.
    ordered = paged.index_select(0, indices).transpose(0, 1)
    ordered = ordered.reshape(2, block_count * block_size, paged.shape[3], paged.shape[4])
    ordered = ordered[:, :seq_len].detach().contiguous().cpu()
    cursor = 0
    for ch in selected:
        end = cursor + ch["token_end"] - ch["token_start"]
        ch["k"] = _signature(ordered[0, cursor:end])
        ch["v"] = _signature(ordered[1, cursor:end])
        cursor = end
    record = {
        "schema": "aissd-resident-numeric-v1", "pid": os.getpid(),
        "request_ordinal": request["ordinal"], "req_id": req_id,
        "decode_step": step, "layer": int(layer_id),
        "resident_active": bool(context.get("aissd_resident_active", False)),
        "candidate_count": candidate_count, "seq_len": seq_len,
        "local_tokens": local_tokens, "local_blocks": local_blocks,
        "physical_blocks": [int(x) for x in physical],
        "q": _signature(query), "selected": selected,
        "remote_k": _signature(ordered[0, :remote_tokens]),
        "remote_v": _signature(ordered[1, :remote_tokens]),
        "local_k": _signature(ordered[0, remote_tokens:]),
        "local_v": _signature(ordered[1, remote_tokens:]),
    }
    return str(Path(root) / "numeric-trace.jsonl"), record


def after_attention(pending, output):
    if pending is None:
        return
    path_string, record = pending
    record["attention_output"] = _signature(output)
    path = Path(path_string)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path_string in _OPENED_PATHS else "w"
    with path.open(mode, encoding="utf-8") as stream:
        stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    _OPENED_PATHS.add(path_string)


def read_trace(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schema") != "aissd-resident-numeric-v1":
            raise ValueError("Unsupported numeric trace schema")
        key = (int(row["request_ordinal"]), int(row["decode_step"]), int(row["layer"]))
        if key in rows:
            raise ValueError(f"Duplicate trace record {key}; do not mix runs")
        rows[key] = row
    if not rows:
        raise ValueError("The numeric trace contains no attention records")
    return rows


def compare_traces(control, resident):
    a, b = read_trace(control), read_trace(resident)
    if set(a) != set(b):
        return {"status": "INCOMPLETE", "control_records": len(a), "resident_records": len(b),
                "missing_from_resident": sorted(set(a) - set(b)),
                "missing_from_control": sorted(set(b) - set(a))}
    first = None
    counts = {}
    for key in sorted(a):
        x, y = a[key], b[key]
        selected_x = [(z["token_start"], z["token_end"]) for z in x["selected"]]
        selected_y = [(z["token_start"], z["token_end"]) for z in y["selected"]]
        checks = [
            ("q", x["q"], y["q"]),
            ("candidate_count", x["candidate_count"], y["candidate_count"]),
            ("selected_ranges_in_order", selected_x, selected_y),
            ("local_tokens", x["local_tokens"], y["local_tokens"]),
            ("local_blocks", x["local_blocks"], y["local_blocks"]),
            ("seq_len", x["seq_len"], y["seq_len"]),
        ] + [(name, x[name], y[name]) for name in
             ("remote_k", "remote_v", "local_k", "local_v", "attention_output")]
        different = [name for name, left, right in checks if left != right]
        for name in different:
            counts[name] = counts.get(name, 0) + 1
        if different and first is None:
            first = {"request_ordinal": key[0], "decode_step": key[1], "layer": key[2],
                     "different_fields": different,
                     "control_selected_ranges": selected_x, "resident_selected_ranges": selected_y,
                     "control_local_tokens": x["local_tokens"], "resident_local_tokens": y["local_tokens"],
                     "different_remote_chunks": [
                         {"position": i, "control_range": [u["token_start"], u["token_end"]],
                          "resident_range": [v["token_start"], v["token_end"]],
                          "k_equal": u["k"] == v["k"], "v_equal": u["v"] == v["v"]}
                         for i, (u, v) in enumerate(zip(x["selected"], y["selected"]))
                         if u["k"] != v["k"] or u["v"] != v["v"]]}
    return {"status": "MATCH" if first is None else "DIFFERENT", "records_per_run": len(a),
            "first_difference": first, "different_record_counts": counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", nargs=2, type=Path, required=True)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    try:
        expected = {(r, s, layer) for r in range(1, args.requests + 1)
                    for s in range(1, args.steps + 1) for layer in range(args.layers)}
        for path in args.compare:
            present = set(read_trace(path))
            if present != expected:
                raise ValueError(f"Incomplete trace {path}: expected {len(expected)}, got {len(present)} records")
        report = compare_traces(*args.compare)
        print(json.dumps(report, indent=2))
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"Trace comparison failed: {error}") from error


if __name__ == "__main__":
    main()

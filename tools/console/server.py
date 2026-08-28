"""Observer Console — a live dashboard for Runtime Lab experiments.

Single-file server built on the stdlib. Streams token-level telemetry
over Server-Sent Events as runs execute.

    python tools/console/server.py [--port 8899]
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
RUNS_DIR = REPO_ROOT / "runs"
REGISTRY_PATH = REPO_ROOT / "models.json"

MAX_LOGS = 500
EVENT_TAIL_POLL_S = 0.1
RUN_DISCOVERY_POLL_S = 0.2


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def infer_mode(run_name: str) -> str:
    for mode in (
        "observe",
        "stress",
        "branchpoints",
        "basins",
        "propagation",
        "hysteresis",
        "control",
    ):
        if run_name.startswith(f"{mode}_run_"):
            return mode
    if run_name.startswith("sweep_"):
        return "sweep"
    return "unknown"


def load_registry() -> Dict[str, Any]:
    payload = read_json(REGISTRY_PATH) or {}
    models = payload.get("models", {}) or {}
    return {
        "default_model": payload.get("default_model"),
        "models": [
            {"key": k, "hf_id": v.get("hf_id", "")}
            for k, v in models.items()
        ],
    }


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    mode = infer_mode(run_dir.name)

    if mode == "sweep":
        sweep = read_json(run_dir / "sweep.json") or {}
        desc = sweep.get("describe") or {}
        aggregate = sweep.get("aggregate") or {}
        headline: Dict[str, Any] = {
            "sweep_mode": sweep.get("mode"),
            "n_seeds": sweep.get("n_seeds"),
            "n_ok": sweep.get("n_ok"),
            "model": desc.get("model"),
        }
        # Surface mean/std of the key per-mode metric.
        def _pick(key: str) -> None:
            a = aggregate.get(key)
            if a:
                headline[key] = a.get("mean")
                headline[f"{key}_std"] = a.get("stdev")

        _pick("avg_divergence")
        _pick("metrics.recovery")
        _pick("metrics.drift")
        return {
            "id": run_dir.name,
            "mode": "sweep",
            "sweep_mode": sweep.get("mode"),
            "updated_at": int(run_dir.stat().st_mtime),
            "prompt": desc.get("prompt") or "",
            "headline": headline,
        }

    summary = read_json(run_dir / "summary.json") or {}
    config = read_json(run_dir / "config.json") or {}

    headline: Dict[str, Any] = {
        "model": summary.get("model_id") or config.get("config", {}).get("model"),
        "device": (summary.get("runtime", {}) or {}).get("device"),
        "dtype": (summary.get("runtime", {}) or {}).get("resolved_dtype"),
    }
    advisory = summary.get("advisory") or {}
    flags = advisory.get("flags") or []
    if flags:
        headline["advisory_flag"] = flags[0]
        headline["advisory_summary"] = advisory.get("summary_line")

    if mode == "observe":
        headline.update({
            "tokens": summary.get("tokens"),
            "avg_divergence": summary.get("avg_divergence"),
        })
    elif mode == "hysteresis":
        metrics = summary.get("metrics", {}) or {}
        headline.update({
            "regime": metrics.get("regime"),
            "drift": metrics.get("drift"),
            "hysteresis": metrics.get("hysteresis"),
            "recovery": metrics.get("recovery"),
        })
    elif mode == "stress":
        results = read_json(run_dir / "results.json") or {}
        metrics = results.get("metrics", {}) or {}
        headline.update({
            "regime": metrics.get("regime"),
            "recovery_ratio": metrics.get("recovery_ratio"),
            "token_match_rate": metrics.get("token_match_rate"),
        })
    elif mode == "branchpoints":
        metrics = summary.get("metrics", {}) or {}
        headline.update({
            "rows": (summary.get("protocol_validity") or {}).get("rows"),
            "argmax_flip_rate": metrics.get("argmax_flip_rate"),
            "sampled_flip_rate": metrics.get("sampled_flip_rate"),
        })
    elif mode == "propagation":
        metrics = summary.get("metrics", {}) or {}
        layer = str((summary.get("capture_layers") or [""])[0])
        injection = (
            (metrics.get("layer_summary") or {}).get(layer) or {}
        )
        terminal = (
            (metrics.get("terminal_summary") or {}).get("final_norm")
            or {}
        )
        headline.update({
            "rows": (summary.get("protocol_validity") or {}).get("rows"),
            "injection_relative": (
                (injection.get("delta_relative_clean") or {}).get("mean")
            ),
            "final_norm_relative": (
                (terminal.get("delta_relative_clean") or {}).get("mean")
            ),
            "mean_logit_kl": metrics.get("mean_logit_kl"),
            "argmax_flip_rate": metrics.get("argmax_flip_rate"),
        })
    elif mode == "basins":
        metrics = summary.get("metrics", {}) or {}
        outcomes = metrics.get("outcomes", {}) or {}
        headline.update({
            "branchpoint_rows": metrics.get("branchpoint_rows"),
            "selected_episodes": metrics.get("selected_episodes"),
            "selected_sampled_flips": metrics.get(
                "selected_sampled_flips"
            ),
            "improve": outcomes.get("improve"),
            "degrade": outcomes.get("degrade"),
            "tie": outcomes.get("tie"),
            "identity_control_exact": metrics.get(
                "identity_control_exact"
            ),
        })
    elif mode == "control":
        headline.update({
            "avg_raw_div_mean": summary.get("avg_raw_div_mean"),
            "avg_score_mean": summary.get("avg_score_mean"),
            "status_counts": summary.get("status_counts"),
        })

    prompt = config.get("config", {}).get("prompt") or ""
    if not prompt:
        # Stress stores prompt in results.json → config → prompt.
        # Hysteresis stores under config.json → config → prompt (covered above).
        results_any = read_json(run_dir / "results.json") or {}
        prompt = (results_any.get("config") or {}).get("prompt") or prompt

    return {
        "id": run_dir.name,
        "mode": mode,
        "updated_at": int(run_dir.stat().st_mtime),
        "prompt": prompt,
        "headline": headline,
    }


def list_runs(limit: int = 60) -> List[Dict[str, Any]]:
    if not RUNS_DIR.exists():
        return []
    dirs = [p for p in RUNS_DIR.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [summarize_run(p) for p in dirs[:limit]]


def run_detail(run_dir: Path) -> Dict[str, Any]:
    mode = infer_mode(run_dir.name)

    if mode == "sweep":
        sweep = read_json(run_dir / "sweep.json") or {}
        return {
            "id": run_dir.name,
            "mode": "sweep",
            "sweep": sweep,
        }

    detail: Dict[str, Any] = {
        "id": run_dir.name,
        "mode": mode,
        "summary": read_json(run_dir / "summary.json"),
        "config": read_json(run_dir / "config.json"),
        "results": read_json(run_dir / "results.json"),
    }

    events_path = run_dir / "events.jsonl"
    if events_path.exists():
        events: List[Dict[str, Any]] = []
        try:
            with events_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except Exception:
            pass
        detail["events"] = events

    if mode == "basins":
        detail["branchpoints"] = _read_jsonl(
            run_dir / "branchpoints.jsonl"
        )
        detail["episodes"] = _read_jsonl(run_dir / "episodes.jsonl")
        detail["trajectories"] = _read_jsonl(
            run_dir / "trajectories.jsonl"
        )

    if mode == "hysteresis":
        detail["outputs"] = {
            "clean_exposure": _read_text(
                run_dir / "output_clean_exposure.txt"
            ),
            "perturbed_exposure": _read_text(
                run_dir / "output_perturbed_exposure.txt"
            ),
            "clean_recovery": _read_text(
                run_dir / "output_clean_recovery.txt"
            ),
            "perturbed_recovery": _read_text(
                run_dir / "output_perturbed_recovery.txt"
            ),
        }
    else:
        detail["output"] = _read_text(run_dir / "output.txt")

    return detail


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    try:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# SSE bus
# ---------------------------------------------------------------------------


class EventBus:
    """Fan out messages from the job to every active SSE client."""

    def __init__(self) -> None:
        self._clients: List[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=2048)
        with self._lock:
            self._clients.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)

    def publish(self, event: str, data: Any) -> None:
        payload = {"event": event, "data": data}
        with self._lock:
            dead: List[queue.Queue] = []
            for q in self._clients:
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                if q in self._clients:
                    self._clients.remove(q)


BUS = EventBus()


# ---------------------------------------------------------------------------
# Job manager
# ---------------------------------------------------------------------------


def build_cli_command(payload: Dict[str, Any]) -> List[str]:
    """Translate one Console request into the exact current-interpreter CLI."""
    mode = str(payload.get("mode") or "observe")
    if mode not in (
        "observe",
        "stress",
        "branchpoints",
        "basins",
        "propagation",
        "hysteresis",
        "control",
    ):
        raise ValueError(f"Unknown mode: {mode}")

    prompt = str(payload.get("prompt") or "")
    max_tokens_value = payload.get("max_tokens")
    max_tokens = int(64 if max_tokens_value is None else max_tokens_value)
    seed_value = payload.get("seed")
    seed = int(42 if seed_value is None else seed_value)
    model = str(payload.get("model") or "").strip()

    cmd: List[str] = [sys.executable, "-m", "runtime_lab.cli.main", mode]
    if model:
        cmd += ["--model", model]
    if prompt:
        cmd += ["--prompt", prompt]
    cmd += ["--max-tokens", str(max_tokens), "--seed", str(seed)]

    probe_layers = str(payload.get("probe_layers") or "auto")
    if mode in (
        "observe",
        "stress",
        "branchpoints",
        "basins",
        "control",
    ):
        cmd += ["--probe-layers", probe_layers]

    seeds_spec = payload.get("seeds")
    if seeds_spec:
        cmd += ["--seeds", str(seeds_spec)]

    if payload.get("temperature") is not None:
        cmd += ["--temperature", str(float(payload["temperature"]))]
    if payload.get("top_p") is not None:
        cmd += ["--top-p", str(float(payload["top_p"]))]
    if payload.get("top_k") is not None:
        cmd += ["--top-k", str(int(payload["top_k"]))]

    def _pass(key: str, flag: str, cast=lambda x: str(x)) -> None:
        if key in payload and payload[key] is not None:
            cmd.extend([flag, cast(payload[key])])

    if mode in ("stress", "branchpoints", "basins", "propagation"):
        _pass("layer", "--layer")
        _pass("intervention_type", "--type")
        _pass("magnitude", "--magnitude", lambda value: str(float(value)))
        if payload.get("absolute_magnitude"):
            cmd.append("--absolute-magnitude")
        if mode == "stress":
            _pass("start", "--start", lambda value: str(int(value)))
            _pass("duration", "--duration", lambda value: str(int(value)))
        if mode == "basins":
            prompt_key = str(payload.get("prompt_key") or "custom")
            cmd += ["--prompt-key", prompt_key]
            _pass("prompt_class", "--prompt-class")
            _pass(
                "continuation_tokens",
                "--continuation-tokens",
                lambda value: str(int(value)),
            )
            _pass(
                "max_episodes",
                "--max-episodes",
                lambda value: str(int(value)),
            )
            _pass(
                "tie_tolerance",
                "--tie-tolerance",
                lambda value: str(float(value)),
            )
    elif mode == "hysteresis":
        perturbation_mode = str(payload.get("perturbation_mode") or "noise")
        cmd += ["--perturbation-mode", perturbation_mode]
        if perturbation_mode == "noise":
            _pass("noise_layer", "--noise-layer")
            _pass("noise_magnitude", "--noise-magnitude", lambda value: str(float(value)))
            _pass("noise_start", "--noise-start", lambda value: str(int(value)))
            _pass("noise_duration", "--noise-duration", lambda value: str(int(value)))
            _pass("recovery_tokens", "--recovery-tokens", lambda value: str(int(value)))
    elif mode == "control":
        _pass("measure_layer", "--measure-layer", lambda value: str(int(value)))
        _pass("act_layer", "--act-layer", lambda value: str(int(value)))
        _pass("intervention_type", "--type")
        if payload.get("shadow"):
            cmd.append("--shadow")

    return cmd


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._job: Optional[Dict[str, Any]] = None

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            if not self._job:
                return {"status": "idle"}
            return {k: v for k, v in self._job.items() if k not in ("proc",)}

    def start(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if self._job and self._job["status"] in ("running", "starting"):
                raise RuntimeError("A job is already running.")

        mode = str(payload.get("mode") or "observe")
        if mode not in (
            "observe",
            "stress",
            "branchpoints",
            "basins",
            "propagation",
            "hysteresis",
            "control",
        ):
            raise ValueError(f"Unknown mode: {mode}")

        prompt = str(payload.get("prompt") or "")
        model = str(payload.get("model") or "").strip()
        cmd = build_cli_command(payload)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

        started_at = time.time()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        job_id = uuid.uuid4().hex[:8]
        job = {
            "id": job_id,
            "mode": mode,
            "model": model,
            "prompt": prompt,
            "command": cmd,
            "status": "starting",
            "started_at": started_at,
            "finished_at": None,
            "pid": proc.pid,
            "run_id": None,
            "run_dir": None,
            "exit_code": None,
            "proc": proc,
            "logs": [],
            "event_count": 0,
        }
        with self._lock:
            self._job = job

        BUS.publish("job_started", {k: v for k, v in job.items() if k != "proc"})

        threading.Thread(target=self._watch_stdout, args=(job_id,), daemon=True).start()
        threading.Thread(target=self._watch_run_dir, args=(job_id,), daemon=True).start()
        return self.snapshot()

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            job = self._job
            if not job or job["status"] not in ("running", "starting"):
                raise RuntimeError("No job to stop.")
            pid = job.get("pid")
        if pid:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        with self._lock:
            if self._job:
                self._job["status"] = "stopping"
        return self.snapshot()

    # -------- internals

    def _watch_stdout(self, job_id: str) -> None:
        with self._lock:
            job = self._job
        if not job or job["id"] != job_id:
            return
        proc: subprocess.Popen = job["proc"]
        assert proc.stdout is not None

        for line in proc.stdout:
            line = line.rstrip("\n")
            if not line:
                continue
            with self._lock:
                if self._job and self._job["id"] == job_id:
                    self._job["logs"].append(line)
                    self._job["logs"] = self._job["logs"][-MAX_LOGS:]
                    if self._job["status"] == "starting":
                        self._job["status"] = "running"
            BUS.publish("log", {"line": line})

        proc.wait()
        exit_code = proc.returncode
        with self._lock:
            if self._job and self._job["id"] == job_id:
                self._job["exit_code"] = exit_code
                self._job["finished_at"] = time.time()
                self._job["status"] = "completed" if exit_code == 0 else "failed"
                run_dir = self._job.get("run_dir")

        final_summary: Optional[Dict[str, Any]] = None
        if run_dir:
            final_summary = read_json(Path(run_dir) / "summary.json")

        BUS.publish("job_finished", {
            "exit_code": exit_code,
            "status": "completed" if exit_code == 0 else "failed",
            "run_id": Path(run_dir).name if run_dir else None,
            "summary": final_summary,
        })

    def _watch_run_dir(self, job_id: str) -> None:
        """Discover run directory, then tail events.jsonl (if present)."""
        with self._lock:
            job = self._job
        if not job or job["id"] != job_id:
            return

        mode = job["mode"]
        started_at = job["started_at"]
        prefix = f"{mode}_run_"

        run_dir: Optional[Path] = None
        while True:
            with self._lock:
                current = self._job
                if not current or current["id"] != job_id:
                    return
                if current["status"] in ("completed", "failed", "stopping"):
                    # Still try to find the dir briefly before giving up.
                    pass

            if RUNS_DIR.exists():
                candidates = [
                    p for p in RUNS_DIR.iterdir()
                    if p.is_dir() and p.name.startswith(prefix)
                    and p.stat().st_mtime >= started_at - 1.0
                ]
                if candidates:
                    run_dir = max(candidates, key=lambda p: p.stat().st_mtime)
                    with self._lock:
                        if self._job and self._job["id"] == job_id:
                            self._job["run_dir"] = str(run_dir)
                            self._job["run_id"] = run_dir.name
                    BUS.publish("run_detected", {"run_id": run_dir.name, "run_dir": str(run_dir)})
                    break

            with self._lock:
                if self._job and self._job["id"] == job_id and self._job["status"] in ("completed", "failed"):
                    return
            time.sleep(RUN_DISCOVERY_POLL_S)

        if run_dir is None:
            return

        # Stream events.jsonl if it exists (observe/stress/control).
        events_path = (
            run_dir / "branchpoints.jsonl"
            if mode == "basins"
            else run_dir / "events.jsonl"
        )
        offset = 0
        buffer = ""
        while True:
            with self._lock:
                current = self._job
                if not current or current["id"] != job_id:
                    return
                job_status = current["status"]

            if events_path.exists():
                try:
                    with events_path.open("r", encoding="utf-8") as f:
                        f.seek(offset)
                        chunk = f.read()
                        offset = f.tell()
                except Exception:
                    chunk = ""
                if chunk:
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            ev = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        with self._lock:
                            if self._job and self._job["id"] == job_id:
                                self._job["event_count"] += 1
                        BUS.publish("event", ev)

            if job_status in ("completed", "failed"):
                # Drain one more pass, then exit.
                time.sleep(0.15)
                if events_path.exists():
                    try:
                        with events_path.open("r", encoding="utf-8") as f:
                            f.seek(offset)
                            chunk = f.read()
                    except Exception:
                        chunk = ""
                    if chunk:
                        buffer += chunk
                        for line in buffer.split("\n"):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                ev = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            BUS.publish("event", ev)
                return

            time.sleep(EVENT_TAIL_POLL_S)


JOBS = JobManager()


# ---------------------------------------------------------------------------
# Capabilities (machine-readable description of what this rig can do)
# ---------------------------------------------------------------------------

_MODES_DOC: Dict[str, Dict[str, Any]] = {
    "observe": {
        "description": "Passive observability — generate tokens and log full telemetry (divergence, spectral, SVD, layer-stiffness). No interventions.",
        "useful_for": [
            "Baseline trajectories on a prompt × model combination",
            "Checking whether divergence scales with length before drawing conclusions",
            "Gathering reference distributions for comparisons later",
        ],
        "params": {
            "prompt": "(str) what the model generates from",
            "model": "(str) key in models.json",
            "max_tokens": "(int) length of generation",
            "seed": "(int) deterministic seed",
            "temperature": "(float) 0=greedy (default); >0 enables sampling, essential for perturbation effects to flip tokens",
            "seeds": "(str) multi-seed sweep spec like '0-4' or '1,3,7'; produces sweep_observe_<ts>/",
            "probe_layers": "'auto' (mid-stack + final) or comma-separated ints",
        },
    },
    "stress": {
        "description": "Branchpoint A/B comparison — run baseline vs intervention from the same prompt-pass seed cache. Compares token match, logit-KL, recovery.",
        "useful_for": [
            "Measuring sensitivity of a layer to a specific perturbation",
            "Checking if an intervention has *any* effect even when tokens don't flip (use logit_kl_mean_during)",
        ],
        "params": {
            "layer": "(str|int) 'mid' (default, resolves to n//2-1), 'early', 'late', or explicit int",
            "type": "additive | projection | scaling | sae",
            "magnitude": "(float) default 0.15, interpreted as fraction of hidden-state norm unless --absolute-magnitude",
            "start": "(int) token index where intervention begins",
            "duration": "(int) token count during which intervention is active",
        },
        "advisories_emitted": [
            "no-op (token_match=1.0) — recommends mid-stack + larger magnitude + sampling",
            "runaway — suggests checking magnitude / layer choice",
        ],
    },
    "branchpoints": {
        "description": (
            "Independent local one-step counterfactuals from verified matched "
            "contexts. The perturbed fork is discarded; only clean generation "
            "advances."
        ),
        "useful_for": [
            "Mapping whether an intervention flips the next argmax locally",
            "Building causal flippability labels without trajectory cascades",
        ],
        "params": {
            "layer": "(str|int) intervention layer",
            "type": "additive | projection | scaling",
            "magnitude": "(float) relative by default",
            "intervention_seed": "(int) perturbation direction seed",
        },
    },
    "basins": {
        "description": (
            "causal-branch-continuation-v1: map independent same-context "
            "local forks, outcome-blindly select early/middle/late "
            "branchpoints, replay them, then continue clean and perturbed "
            "branches with no further intervention and paired per-position "
            "sampling randomness."
        ),
        "useful_for": [
            "Testing whether a local token flip changes task outcome",
            "Separating local flips, latent divergence, ties, and exact no-effects",
            "Building controller-free basin outcome labels",
        ],
        "params": {
            "prompt_key": (
                "registered M3R prompt key, or custom for an exploratory "
                "generic rubric"
            ),
            "prompt": (
                "required for custom; registered prompts must match exactly"
            ),
            "layer": "(str|int) intervention layer",
            "type": "additive | projection | scaling",
            "magnitude": "(float) relative by default",
            "continuation_tokens": "(int) no-intervention tokens per branch",
            "max_episodes": "(int) outcome-blind selected forks per run",
            "tie_tolerance": "(float) score delta treated as a tie",
        },
    },
    "propagation": {
        "description": (
            "same-context-layer-propagation-v1: independent one-step "
            "clean/perturbed forks that capture the paired delta at every "
            "downstream transformer block and the model's final normalization."
        ),
        "useful_for": [
            "Mapping where an injected hidden-state delta grows or contracts",
            "Separating raw L2 growth from relative and angular disruption",
            "Detecting perturbations absorbed by final normalization",
        ],
        "params": {
            "layer": "(str|int) injection layer",
            "type": "additive | projection | scaling",
            "magnitude": "(float) relative by default",
            "intervention_seed": "(int) perturbation direction seed",
        },
    },
    "hysteresis": {
        "description": (
            "matched-exposure-recovery-v2: clean and perturbed branches "
            "continue under equal instructions, then both recover with no "
            "intervention."
        ),
        "useful_for": [
            "Separating direct effect, propagation, persistence, and recovery",
            "Classifying recovery only when both branches remain matched",
        ],
        "params": {
            "perturbation_mode": "noise",
            "noise_layer": "'mid' (default) or int — recommended to stay mid-stack so effect cascades",
            "noise_magnitude": "float, default 0.15, fraction of hidden norm",
            "noise_start": "(int) step where noise begins",
            "noise_duration": "(int) how many steps noise is active",
            "recovery_tokens": "(int) equal intervention-free continuation length",
        },
    },
    "control": {
        "description": "Closed-loop controller — continuously monitors divergence, applies scaling intervention when threshold crossed. Shadow mode measures without intervening.",
        "useful_for": [
            "Testing whether proportional control actually stabilizes trajectories (run in active mode!)",
            "Calibrating thresholds via shadow mode before going active",
        ],
        "params": {
            "shadow": "(bool) measure only; set to false to let controller intervene",
            "measure_layer": "(int) layer whose divergence drives the controller",
            "act_layer": "(int) layer where scaling is applied",
            "threshold_warn / threshold_crit": "divergence thresholds for WARN / CRIT status",
            "scale_warn / scale_crit": "scaling factor applied at each status level",
        },
    },
}


def _capabilities_payload() -> Dict[str, Any]:
    registry = load_registry()
    from runtime_lab.basins.evaluators import ALL_PROMPT_SPECS

    return {
        "version": "0.2.0-llm-first",
        "description": (
            "Observer — Observe → Perturb → Compare → Prove instrumentation. "
            "This capabilities endpoint is for LLM-driven experiment planning: "
            "fetch it, pick a mode, construct a payload per `params`, POST to /api/launch, "
            "then read /api/runs/<id> to get the summary including `advisory` — a structured "
            "block explaining what happened and what to try next."
        ),
        "endpoints": {
            "POST /api/launch": "Start a run. Body: {mode, model, prompt, ...mode-specific params}. Returns {job: {...}}.",
            "GET /api/status": "Poll for current job status.",
            "POST /api/stop": "Cancel active job.",
            "GET /api/runs": "List recent runs (summary metadata).",
            "GET /api/runs/{id}": "Full detail for one run, including advisory.",
            "GET /api/compare?ids=a,b,c": "Get divergence/entropy/hidden series for multiple runs for overlay comparison.",
            "GET /api/stream": "SSE stream of job events (logs, events, job_finished).",
            "GET /api/capabilities": "This payload.",
        },
        "models": registry.get("models", []),
        "default_model": registry.get("default_model"),
        "modes": _MODES_DOC,
        "registered_basin_prompts": [
            {
                "key": spec.key,
                "prompt_class": spec.prompt_class,
                "prompt": spec.prompt,
                "evaluator_id": spec.evaluator_id,
            }
            for spec in ALL_PROMPT_SPECS
        ],
        "advisories": {
            "description": (
                "Every completed run's summary.json contains `advisory`, a structured block with "
                "{observations, likely_causes, next_actions, flags, confidence, summary_line}. "
                "The next_actions list is designed to be acted on directly: copy the params dict "
                "into your next POST /api/launch body."
            ),
            "known_flags": ["no-op", "nominal", "quiet", "volatile", "degenerate",
                            "persistent-effect", "no-propagation", "protocol-invalid",
                            "shadow", "active", "good-recovery"],
        },
        "recipes": _RECIPES,
    }


_RECIPES: Dict[str, Dict[str, Any]] = {
    "smoke-test": {
        "description": "Smallest possible end-to-end validation — 8 tokens, default model.",
        "mode": "observe",
        "payload": {"max_tokens": 8, "prompt": "The sky is"},
    },
    "hysteresis-noise-validate": {
        "description": "Matched exposure/recovery calibration run; interpret only when propagation and protocol validity pass.",
        "mode": "hysteresis",
        "payload": {
            "perturbation_mode": "noise",
            "noise_layer": "mid",
            "noise_magnitude": 0.3,
            "noise_duration": 8,
            "recovery_tokens": 32,
            "temperature": 0.8,
            "max_tokens": 64,
        },
    },
    "local-branchpoint-map": {
        "description": "Independent same-context one-step counterfactual map.",
        "mode": "branchpoints",
        "payload": {
            "layer": "mid",
            "intervention_type": "additive",
            "magnitude": 0.15,
            "max_tokens": 64,
        },
    },
    "downstream-propagation-map": {
        "description": (
            "Same-context one-step delta curve through all downstream layers "
            "and final normalization."
        ),
        "mode": "propagation",
        "payload": {
            "layer": "mid",
            "intervention_type": "additive",
            "magnitude": 0.2,
            "max_tokens": 32,
        },
    },
    "causal-basin-map": {
        "description": (
            "Controller-free local fork continuation using a registered "
            "task-specific outcome rubric."
        ),
        "mode": "basins",
        "payload": {
            "prompt_key": "water_cycle",
            "layer": "late",
            "intervention_type": "additive",
            "magnitude": 0.3,
            "max_tokens": 48,
            "continuation_tokens": 48,
            "max_episodes": 3,
            "temperature": 0.8,
        },
    },
    "stress-logit-kl": {
        "description": "Stress run that measures logit-KL even if tokens don't flip. Use to confirm an intervention IS affecting the model even when match_rate=1.",
        "mode": "stress",
        "payload": {
            "layer": "mid",
            "type": "additive",
            "magnitude": 0.3,
            "start": 5,
            "duration": 10,
            "max_tokens": 48,
        },
    },
    "control-ab": {
        "description": "Run the same prompt in shadow and active control back-to-back. Compare metrics to verify the controller actually helps.",
        "mode": "control",
        "chain": [
            {"payload": {"shadow": True, "max_tokens": 64}, "tag": "shadow"},
            {"payload": {"shadow": False, "max_tokens": 64}, "tag": "active"},
        ],
    },
    "seed-variance": {
        "description": "Run N seeds to check that a measured effect is not a single-seed fluke.",
        "mode": "observe",
        "payload": {"seeds": "0-4", "max_tokens": 48, "temperature": 0.7},
    },
}


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


STATIC_MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    # ---- helpers

    def _write(self, status: int, body: bytes, content_type: str, extra_headers: Optional[Dict[str, str]] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._write(status, data, "application/json; charset=utf-8")

    def _read_body(self) -> Dict[str, Any]:
        n = int(self.headers.get("Content-Length", "0"))
        if n <= 0:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    # ---- static

    def _serve_static(self, rel: str) -> bool:
        path = (STATIC_DIR / rel).resolve()
        try:
            path.relative_to(STATIC_DIR.resolve())
        except ValueError:
            return False
        if not path.exists() or not path.is_file():
            return False
        mime = STATIC_MIME.get(path.suffix, "application/octet-stream")
        body = path.read_bytes()
        self._write(200, body, mime)
        return True

    # ---- routes

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            if self._serve_static("index.html"):
                return
            self._json({"error": "index.html missing"}, 500)
            return

        if path.startswith("/static/"):
            rel = path[len("/static/"):]
            if self._serve_static(rel):
                return
            self._write(404, b"not found", "text/plain")
            return

        if path == "/api/models":
            self._json(load_registry())
            return

        if path == "/api/runs":
            self._json({"runs": list_runs()})
            return

        if path.startswith("/api/runs/"):
            run_id = path[len("/api/runs/"):]
            run_dir = RUNS_DIR / run_id
            if not run_dir.exists() or not run_dir.is_dir():
                self._json({"error": "run not found"}, 404)
                return
            self._json(run_detail(run_dir))
            return

        if path == "/api/compare":
            ids = parse_qs(parsed.query).get("ids", [""])[0]
            run_ids = [i for i in ids.split(",") if i]
            out = []
            for rid in run_ids[:10]:
                rd = RUNS_DIR / rid
                if not rd.exists():
                    continue
                mode = infer_mode(rd.name)
                series: List[Dict[str, Any]] = []
                events_path = rd / "events.jsonl"
                if events_path.exists():
                    try:
                        with events_path.open("r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    ev = json.loads(line)
                                    series.append({
                                        "t": ev.get("t"),
                                        "divergence": (ev.get("diagnostics") or {}).get("divergence"),
                                        "entropy": ((ev.get("diagnostics") or {}).get("spectral") or {}).get("spectral_entropy"),
                                        "hidden": ev.get("hidden_post_norm"),
                                    })
                                except Exception:
                                    pass
                    except Exception:
                        pass
                out.append({
                    "id": rid,
                    "mode": mode,
                    "series": series,
                    "headline": summarize_run(rd).get("headline", {}),
                })
            self._json({"runs": out})
            return

        if path == "/api/status":
            self._json({"job": JOBS.snapshot()})
            return

        if path == "/api/capabilities":
            self._json(_capabilities_payload())
            return

        if path == "/api/stream":
            self._stream()
            return

        self._write(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/launch":
            try:
                JOBS.start(self._read_body())
            except Exception as e:
                self._json({"error": str(e)}, 400)
                return
            self._json({"job": JOBS.snapshot()}, 201)
            return
        if path == "/api/stop":
            try:
                JOBS.stop()
            except Exception as e:
                self._json({"error": str(e)}, 400)
                return
            self._json({"job": JOBS.snapshot()})
            return
        self._write(404, b"not found", "text/plain")

    # ---- SSE

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        q = BUS.subscribe()
        # Send snapshot of current job on connect.
        try:
            self._send_sse("hello", {"job": JOBS.snapshot()})
        except Exception:
            BUS.unsubscribe(q)
            return

        try:
            while True:
                try:
                    msg = q.get(timeout=15)
                except queue.Empty:
                    # heartbeat
                    try:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                    except Exception:
                        break
                    continue
                try:
                    self._send_sse(msg["event"], msg["data"])
                except Exception:
                    break
        finally:
            BUS.unsubscribe(q)

    def _send_sse(self, event: str, data: Any) -> None:
        payload = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")
        self.wfile.write(payload)
        self.wfile.flush()


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, int(args.port)), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"[observer-console] serving at {url}")
    print("[observer-console] press Ctrl-C to stop")
    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[observer-console] shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

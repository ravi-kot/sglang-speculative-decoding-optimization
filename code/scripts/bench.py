"""Serving benchmark for stock vs speculative (NEXTN) vs candidate SGLang configs.

  python scripts/bench.py run configs/l4.yaml [--variants A,B] [--workloads code_8k,...]
                              [--rounds N] [--hypothesis "..."] [--profile torch|nsys]
  python scripts/bench.py fingerprint configs/l4.yaml

Every variant is a fresh `sglang.launch_server` process group with identical common flags;
variants differ only in config. Requests are streamed one at a time (concurrency 1) from
fixed token-id prompt sets (data/build_prompts.py). Per run this writes
experiments/runs/<run_id>/{manifest.json, summary.json, requests.jsonl, server_*.log},
appends one line to experiments/ledger.jsonl and prints a few summary lines.

Effective TPOT = (e2e - TTFT) / (committed tokens - 1), committed tokens counted from the
response. Round time = (e2e - TTFT) / spec_verify_ct. Acceptance comes from SGLang meta_info.
"""

import argparse
import gzip
import hashlib
import json
import os
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
PROMPTS = REPO / "data" / "prompts"
RUNS = REPO / "experiments" / "runs"
SPEC_KEYS = ("spec_verify_ct", "spec_accept_length", "spec_accept_rate", "spec_num_correct_drafts",
             "spec_num_proposed_drafts", "spec_correct_drafts_histogram")


# ---------- small utils ----------
def sh(cmd, timeout=60):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:  # fingerprint fields are best effort
        return f"ERR {e}"


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))] if v else None


def r3(x):
    return None if x is None else round(x, 3)


def smi():
    return shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"


def gpu_used_free():
    out = sh(f"{smi()} --query-gpu=memory.used,memory.free --format=csv,noheader,nounits", 10)
    used, free = (int(x) for x in out.split("\n")[0].split(","))
    return used, free


def http(port, path, body=None, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    try:
        return json.loads(raw or b"null")
    except ValueError:  # some endpoints (e.g. /start_profile) answer in plain text
        return raw.decode(errors="replace")[:200]


def load_set(name):
    with gzip.open(PROMPTS / f"{name}.json.gz", "rt") as fh:
        return json.load(fh)


# ---------- environment fingerprint ----------
def fingerprint(cfg):
    py = cfg.get("python", sys.executable)
    probe = ("import json,sys,torch,triton\n"
             "d={'python':sys.version.split()[0],'torch':torch.__version__,'torch_cuda':torch.version.cuda,"
             "'triton':triton.__version__}\n"
             "p=torch.cuda.get_device_properties(0);d.update(gpu=p.name,cc=f'{p.major}.{p.minor}',"
             "sm_count=p.multi_processor_count,vram_gb=round(p.total_memory/2**30,2))\n"
             "for m in ('sglang','flashinfer','sgl_kernel'):\n"
             "  try: d[m]=__import__(m).__version__\n"
             "  except Exception as e: d[m]=None\n"
             "import sglang,os;r=os.path.dirname(os.path.dirname(sglang.__file__))\n"
             "d['sglang_path']=r\n"
             "print(json.dumps(d))")
    env = {**os.environ, **cfg.get("env", {})}
    out = subprocess.run([py, "-c", probe], capture_output=True, text=True, env=env, timeout=300)
    fp = json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 else {"probe_error": out.stderr[-500:]}
    src = fp.get("sglang_path", "")
    fp["sglang_commit"] = sh(f"git -C {src} rev-parse HEAD 2>/dev/null") or os.environ.get("SGLANG_COMMIT", "unknown")
    fp["driver"] = sh(f"{smi()} --query-gpu=driver_version --format=csv,noheader")
    fp["cuda_driver_api"] = sh(f"{smi()} | grep -o 'CUDA Version: [0-9.]*'")
    fp["cpu"] = sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2").strip()
    fp["nproc"] = os.cpu_count()
    fp["ram_gb"] = round(int(sh("grep MemTotal /proc/meminfo | tr -dc 0-9")) / 2**20, 1)
    fp["disk_free_gb"] = round(shutil.disk_usage(REPO).free / 2**30, 1)
    fp["nsys"] = sh("nsys --version 2>/dev/null") or None
    fp["ncu"] = sh("ncu --version 2>/dev/null | tail -1") or None
    fp["image"] = os.environ.get("IMAGE_REF")  # set by scripts/ec2.sh (repo@sha256:...)
    fp["host"] = socket.gethostname()
    fp["env"] = {k: v for k, v in env.items() if k.startswith(("SGLANG", "CUDA", "NCCL", "TORCH", "HF_HUB", "FLASHINFER"))}
    return fp


def git_state():
    head = sh(f"git -C {REPO} rev-parse --verify -q HEAD") or "uncommitted"
    diff = subprocess.run(["git", "-C", str(REPO), "diff", "HEAD"], capture_output=True).stdout if head != "uncommitted" else b""
    dirty = sh(f"git -C {REPO} status --porcelain --untracked-files=no 2>/dev/null")
    return {"commit": head, "dirty": bool(dirty) or head == "uncommitted",
            "diff_sha256": hashlib.sha256(diff).hexdigest() if diff else None}


# ---------- server lifecycle ----------
class Server:
    def __init__(self, cfg, variant, flags, log_path, prefix=(), extra_env=None):
        self.port = cfg.get("port", 30200)
        self.cmd = [*prefix, cfg.get("python", sys.executable), "-m", "sglang.launch_server",
                    "--model-path", str(cfg["model"]), "--port", str(self.port),
                    *map(str, cfg.get("server_common", [])), *map(str, flags)]
        if cfg.get("model_revision") and not Path(str(cfg["model"])).exists():
            self.cmd += ["--revision", cfg["model_revision"]]
        self.env = {**os.environ, **{k: str(v) for k, v in {**cfg.get("env", {}), **(extra_env or {})}.items()}}
        self.log_path, self.variant, self.proc = log_path, variant, None

    def start(self, timeout):
        self.log = open(self.log_path, "w")
        self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT,
                                     env=self.env, start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited rc={self.proc.returncode}; see {self.log_path.name}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5)
                return time.time() - t0
            except Exception:
                time.sleep(3)
        raise TimeoutError(f"server not healthy after {timeout}s")

    def pids(self):
        if self.proc is None:
            return []
        out = []
        for d in Path("/proc").iterdir():
            if d.name.isdigit():
                try:
                    if os.getpgid(int(d.name)) == self.proc.pid:
                        out.append(int(d.name))
                except OSError:
                    pass
        return out

    def cpu_seconds(self):
        tick, total = os.sysconf("SC_CLK_TCK"), 0.0
        for pid in self.pids():
            try:
                f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                total += (int(f[11]) + int(f[12])) / tick
            except OSError:
                pass
        return total

    def stop(self, grace=30):
        if self.proc is None:
            return
        for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 10)):
            try:
                os.killpg(self.proc.pid, sig)  # only our own process group
            except ProcessLookupError:
                break
            t0 = time.time()
            while time.time() - t0 < wait and self.pids():
                time.sleep(1)
            if not self.pids():
                break
        self.log.close()


class VramSampler(threading.Thread):  # also samples SM clock / power: a clock-capped run is not comparable
    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.stop_evt, self.clk, self.pwr = 0, threading.Event(), [], []
        self.mem_avail_min, self.swap0 = None, self._swap()  # host RAM / swap: WSL has ~7 GB

    @staticmethod
    def _swap():
        v = dict(l.split() for l in Path("/proc/vmstat").read_text().splitlines() if l.startswith(("pswpin", "pswpout")))
        return int(v.get("pswpin", 0)) + int(v.get("pswpout", 0))

    def run(self):
        # laptop GPU: the enforced power limit moves (CPU/GPU power sharing); a capped GPU throttles bandwidth
        q = (f"{smi()} --query-gpu=memory.used,clocks.sm,power.draw,clocks.mem,enforced.power.limit,"
             f"clocks_event_reasons.sw_power_cap --format=csv,noheader,nounits")
        self.memclk, self.plim, self.capped = [], [], 0
        while not self.stop_evt.is_set():
            try:
                f = [x.strip() for x in sh(q, 10).splitlines()[0].split(",")]
                u, c, w = float(f[0]), float(f[1]), float(f[2])
                self.peak = max(self.peak, int(u))
                ma = int(sh("grep MemAvailable /proc/meminfo | tr -dc 0-9", 5)) // 1024
                self.mem_avail_min = ma if self.mem_avail_min is None else min(self.mem_avail_min, ma)
                if w > 15:  # only while the GPU is working
                    self.clk.append(c)
                    self.pwr.append(w)
                    self.memclk.append(float(f[3]))
                    self.plim.append(float(f[4]))
                    self.capped += f[5] == "Active"
            except Exception:
                pass
            time.sleep(0.5)

    def stats(self):
        if not self.clk:
            return {}
        return {"sm_clock_mhz_p10_p50": [pct(self.clk, .1), pct(self.clk, .5)],
                "power_w_p50": round(pct(self.pwr, .5), 1), "busy_samples": len(self.clk),
                "host_mem_avail_min_mib": self.mem_avail_min, "swap_pages_io": self._swap() - self.swap0,
                "mem_clock_mhz_p10_p50": [pct(self.memclk, .1), pct(self.memclk, .5)],
                "power_limit_w_min_p50": [min(self.plim), pct(self.plim, .5)],
                "power_capped_frac": round(self.capped / len(self.clk), 3)}


# ---------- one request ----------
def stream_one(port, ids, max_new, sampling):
    body = {"input_ids": ids, "stream": True,
            "sampling_params": {"max_new_tokens": max_new, **sampling}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    arrivals, out_ids, meta = [], [], {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            meta = chunk.get("meta_info", meta)
            n = meta.get("completion_tokens", 0)
            new = chunk.get("output_ids") or []
            out_ids = list(new) if len(new) == n else out_ids + list(new)  # cumulative or incremental
            arrivals.append((time.perf_counter() - t0, n))
    t_end = time.perf_counter() - t0
    first = next((t for t, n in arrivals if n >= 1), None)
    n_out = meta.get("completion_tokens", len(out_ids))
    rec = {"ttft_ms": r3(first * 1e3) if first else None, "e2e_ms": r3(t_end * 1e3),
           "completion_tokens": n_out, "finish_reason": (meta.get("finish_reason") or {}).get("type"),
           "finish_message": ((meta.get("finish_reason") or {}).get("message") or "")[:160] or None,
           "chunks": len(arrivals), "output_ids": out_ids}
    if first and n_out > 1:
        rec["eff_tpot_ms"] = r3((t_end - first) * 1e3 / (n_out - 1))
        gaps = [b[0] - a[0] for a, b in zip(arrivals, arrivals[1:]) if b[1] > a[1]]
        rec["chunk_gap_ms_p50"] = r3(statistics.median(gaps) * 1e3) if gaps else None
    for k in SPEC_KEYS:
        if k in meta:
            rec[k] = meta[k]
    if rec.get("spec_verify_ct") and first:
        rec["round_ms"] = r3((t_end - first) * 1e3 / rec["spec_verify_ct"])
    return rec


def summarize(recs):
    def col(k):
        return [r[k] for r in recs if r.get(k) is not None]

    aborted = sum(r.get("finish_reason") == "abort" for r in recs)
    recs = [r for r in recs if r.get("finish_reason") != "abort"]
    if not recs:
        return {"n": 0, "aborted": aborted, "tokens_mean": None, "eff_tpot_ms_p50": None, "ttft_ms_p50": None}
    s = {"n": len(recs), "aborted": aborted, "tokens_mean": r3(statistics.mean(col("completion_tokens"))),
         "ttft_ms_p50": r3(pct(col("ttft_ms"), .5)), "ttft_ms_p95": r3(pct(col("ttft_ms"), .95)),
         "eff_tpot_ms_p50": r3(pct(col("eff_tpot_ms"), .5)), "eff_tpot_ms_p95": r3(pct(col("eff_tpot_ms"), .95)),
         "e2e_ms_p50": r3(pct(col("e2e_ms"), .5)),
         "out_tok_s": r3(sum(col("completion_tokens")) / (sum(col("e2e_ms")) / 1e3)),
         "chunk_gap_ms_p50": r3(pct(col("chunk_gap_ms_p50"), .5))}
    if col("spec_verify_ct"):
        s.update({"accept_len_mean": r3(statistics.mean(col("spec_accept_length"))),
                  "accept_rate_mean": r3(statistics.mean(col("spec_accept_rate"))),
                  "round_ms_p50": r3(pct(col("round_ms"), .5)),
                  "verify_ct_total": sum(col("spec_verify_ct"))})
        hists = col("spec_correct_drafts_histogram")
        if hists:
            m = max(len(h) for h in hists)
            s["accept_hist"] = [sum(h[i] for h in hists if i < len(h)) for i in range(m)]
    return s


# ---------- profiling (server stays up, normal CUDA graph config) ----------
def capture_profile(srv, tool, wl, prompts, sampling, out_dir):
    ids = prompts[0]
    acts = {"nsys": ["CUDA_PROFILER"], "torch": ["GPU"], "torch_cpu": ["CPU", "GPU"]}[tool]  # torch_cpu: attribution only
    # long dedicated request so the window lands in steady decode, not in prefill or after the end
    n_out = wl.get("profile_output_len", 1024)
    th = threading.Thread(target=stream_one, args=(srv.port, ids, n_out, {**sampling, "ignore_eos": True}),
                          daemon=True)
    th.start()
    time.sleep(wl.get("profile_delay_s", 1.0))
    res = http(srv.port, "/start_profile", {"output_dir": str(out_dir), "activities": acts,
                                            "num_steps": wl.get("profile_steps", 10 if tool == "torch_cpu" else 30),
                                            **({"with_stack": True} if tool == "torch_cpu" else {})})
    th.join()
    return {"tool": tool, "workload": wl["name"], "activities": acts, "output_len": n_out, "start_profile": res}


# ---------- commands ----------
def cmd_run(cfg, a):
    variants = a.variants.split(",") if a.variants else list(cfg["variants"])
    wls = [w for w in cfg["workloads"] if not a.workloads or w["name"] in a.workloads.split(",")]
    rounds = a.rounds or cfg.get("rounds", 1)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{cfg['name']}"
    out = RUNS / run_id
    out.mkdir(parents=True)
    trace_dir = Path(cfg.get("trace_dir", out / "traces")) / run_id
    manifest = {"run_id": run_id, "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "git": git_state(), "config_file": a.config, "config": cfg, "variants": variants,
                "workloads": [w["name"] for w in wls], "rounds": rounds, "hypothesis": a.hypothesis,
                "prompt_meta": json.loads((PROMPTS / "meta.json").read_text())["sets"],
                "fingerprint": fingerprint(cfg), "servers": [], "status": "running"}
    sampling = cfg.get("sampling", {"temperature": 0})
    reqf = open(out / "requests.jsonl", "w")
    results = {}
    try:
        for rnd in range(rounds):
            order = variants if rnd % 2 == 0 else variants[::-1]  # A B, B A, ...
            for v in order:
                used0, free0 = gpu_used_free()
                need = cfg.get("min_free_mib", 0)
                if free0 < need:
                    raise RuntimeError(f"VRAM gate: {free0} MiB free < {need} MiB; not launching")
                prefix = ()
                if a.profile == "nsys" and rnd == 0:
                    trace_dir.mkdir(parents=True, exist_ok=True)
                    prefix = ("nsys", "profile", "--trace=cuda,nvtx,osrt", "--trace-fork-before-exec=true",
                              "--cuda-graph-trace=node", "--capture-range=cudaProfilerApi",
                              "--capture-range-end=stop", "-o", str(trace_dir / f"{v}"), "-f", "true")
                spec = cfg["variants"][v] or []  # list of flags, or {flags: [...], env: {...}} (e.g. a source overlay)
                flags, venv = (spec.get("flags", []), spec.get("env", {})) if isinstance(spec, dict) else (spec, {})
                srv = Server(cfg, v, flags, out / f"server_{v}_r{rnd}.log", prefix, venv)
                sinfo = {"variant": v, "round": rnd, "cmd": srv.cmd, "variant_env": venv, "vram_used_before_mib": used0}
                manifest["servers"].append(sinfo)
                vs = VramSampler()
                try:
                    sinfo["startup_s"] = round(srv.start(cfg.get("startup_timeout_s", 1200)), 1)
                    info = http(srv.port, "/get_server_info")
                    sinfo["resolved"] = {k: info.get(k) for k in (
                        "speculative_algorithm", "speculative_num_steps", "speculative_eagle_topk",
                        "speculative_num_draft_tokens", "attention_backend", "mem_fraction_static",
                        "cuda_graph_max_bs", "disable_cuda_graph", "max_total_num_tokens", "version")}
                    vs.start()
                    cpu0, t0 = srv.cpu_seconds(), time.time()
                    for wl in wls:
                        prompts = load_set(wl["set"])[: wl.get("n", 8) + cfg.get("warmup", 2)]
                        warm, meas = prompts[len(prompts) - cfg.get("warmup", 2):], prompts[: wl.get("n", 8)]
                        for ids in warm:
                            stream_one(srv.port, ids, wl["output_len"], sampling)
                        for i, ids in enumerate(meas):
                            rec = stream_one(srv.port, ids, wl["output_len"], sampling)
                            rec.update(variant=v, round=rnd, workload=wl["name"], prompt=i, input_len=len(ids))
                            reqf.write(json.dumps(rec) + "\n")
                            reqf.flush()
                            results.setdefault((v, wl["name"]), []).append(rec)
                    sinfo["cpu_util_pct"] = round(100 * (srv.cpu_seconds() - cpu0) / max(1e-6, time.time() - t0), 1)
                    try:
                        sinfo["avg_spec_accept_length"] = (http(srv.port, "/get_server_info").get(
                            "internal_states") or [{}])[0].get("avg_spec_accept_length")
                    except Exception:
                        pass
                    if a.profile and rnd == 0:
                        pw = next(w for w in cfg["workloads"] if w["name"] == (a.profile_workload or cfg.get("profile_workload", wls[0]["name"])))
                        trace_dir.mkdir(parents=True, exist_ok=True)
                        sinfo["profile"] = capture_profile(srv, a.profile, pw, load_set(pw["set"]), sampling,
                                                           trace_dir / v if a.profile.startswith("torch") else trace_dir)
                        if a.profile.startswith("torch"):  # trace export is asynchronous; wait before teardown
                            t1 = time.time()
                            while time.time() - t1 < 120 and not list((trace_dir / v).glob("*.json*")):
                                time.sleep(2)
                                if time.time() - t1 > 30:  # window not closed by num_steps: close it
                                    try:
                                        http(srv.port, "/stop_profile", {})
                                    except Exception:
                                        pass
                            last = -1  # the file appears when export starts: wait until its size is stable
                            while time.time() - t1 < 180:
                                size = sum(p.stat().st_size for p in (trace_dir / v).glob("*.json*"))
                                if size and size == last:
                                    break
                                last = size
                                time.sleep(3)
                            sinfo["profile"]["report"] = [str(p) for p in (trace_dir / v).glob("*.json*")]
                        if a.profile == "nsys":  # wait for nsys to finish writing before teardown
                            t1 = time.time()
                            while time.time() - t1 < 180 and not (trace_dir / f"{v}.nsys-rep").exists():
                                time.sleep(2)
                                if time.time() - t1 > 60:  # documented fallback: end the session explicitly
                                    sh("nsys stop")
                            sinfo["profile"]["report"] = str(trace_dir / f"{v}.nsys-rep")
                            sinfo["profile"]["report_exists"] = (trace_dir / f"{v}.nsys-rep").exists()
                finally:
                    vs.stop_evt.set()
                    srv.stop()
                    sinfo["peak_vram_mib"] = vs.peak
                    sinfo.update(vs.stats())
                    time.sleep(2)
                    sinfo["vram_used_after_mib"] = gpu_used_free()[0]
                    sinfo["vram_released"] = sinfo["vram_used_after_mib"] <= used0 + 300
        manifest["status"] = "ok"
        if any(rs and any(r.get("finish_reason") == "abort" for r in rs) for rs in results.values()):
            manifest["status"] = "ok_with_aborted_requests"
    except BaseException as e:
        manifest["status"] = f"failed: {type(e).__name__}: {e}"
        raise
    finally:
        reqf.close()
        summary = {f"{v}|{w}": summarize(rs) for (v, w), rs in results.items()}
        peak = {s["variant"]: max(x.get("peak_vram_mib", 0) for x in manifest["servers"] if x["variant"] == s["variant"])
                for s in manifest["servers"]}
        for k, s in summary.items():
            s["peak_vram_mib"] = peak.get(k.split("|")[0])
        manifest["trace_dir"] = str(trace_dir) if a.profile else None
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        led = {"run_id": run_id, "config": cfg["name"], "git": manifest["git"]["commit"][:12],
               "dirty": manifest["git"]["dirty"], "hypothesis": a.hypothesis, "status": manifest["status"],
               "metrics": {k: {m: s.get(m) for m in ("eff_tpot_ms_p50", "accept_len_mean", "ttft_ms_p50",
                                                      "peak_vram_mib")} for k, s in summary.items()},
               "verdict": a.verdict}
        with open(REPO / "experiments" / "ledger.jsonl", "a") as fh:
            fh.write(json.dumps(led) + "\n")
        print(f"RUN {run_id} status={manifest['status']} git={led['git']}{'+dirty' if led['dirty'] else ''}")
        for k, s in summary.items():
            print(f"  {k:<24} tok={s['tokens_mean']} TPOT={s['eff_tpot_ms_p50']}ms TTFT={s['ttft_ms_p50']}ms "
                  f"acc={s.get('accept_len_mean', '-')} vram={s['peak_vram_mib']}MiB"
                  + (f" ABORTED={s['aborted']}" if s.get("aborted") else ""))
        rel = [s.get("vram_released") for s in manifest["servers"]]
        clk = {s["variant"]: s.get("sm_clock_mhz_p10_p50") for s in manifest["servers"]}
        pw = {s["variant"]: (s.get("power_w_p50"), s.get("power_limit_w_min_p50"), s.get("power_capped_frac"),
                             s.get("mem_clock_mhz_p10_p50")) for s in manifest["servers"]}
        print(f"  servers={len(rel)} vram_released={all(rel)} sm_clock_p10_p50={clk}")
        print(f"  power_w_p50,limit_min_p50,capped_frac,memclk_p10_p50={pw}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "fingerprint"))
    ap.add_argument("config")
    ap.add_argument("--variants")
    ap.add_argument("--workloads")
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--hypothesis", default="")
    ap.add_argument("--verdict", default="")
    ap.add_argument("--profile", choices=("torch", "torch_cpu", "nsys"))
    ap.add_argument("--profile-workload", help="override config profile_workload")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.cmd == "fingerprint":
        print(json.dumps(fingerprint(cfg), indent=1))
    else:
        cmd_run(cfg, a)


if __name__ == "__main__":
    main()

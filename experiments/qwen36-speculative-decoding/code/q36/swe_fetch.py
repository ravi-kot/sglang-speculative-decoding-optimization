"""Run 4 data, step 1 (no GPU, no full download): pick independent in-domain SWE-agent trajectories from
nebius/SWE-rebench-openhands-trajectories (CC-BY-4.0; 67,074 OpenHands trajectories, Qwen3-Coder-480B, 1,823 Python repos)
by reading the remote parquet through HTTP Range requests: footer + small metadata columns first, then only the row groups that
hold the chosen trajectories.
Selection: at most one trajectory per repo; repos whose name matches any evaluation/training trace prefix are excluded (MLCommons
trace ids start with the project name, e.g. "httpx-streaming-json-iteration"); --resolved-frac of the picks resolved, the rest
unresolved (error recovery); trajectories with fewer than --min-asst assistant messages dropped after fetch. Row groups are chosen
greedily by how many new repos they add, so few groups are read.
Output: JSON lines {session, repo, instance_id, resolved, tools, messages} for build_train.py --source swe.
Usage: python swe_fetch.py --exclude turns_*.jsonl --n 140 --out swe_traj.jsonl [--max-groups 6]"""
import argparse, collections, json, random, urllib.request
import pyarrow.parquet as pq

U = "https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
ap = argparse.ArgumentParser()
ap.add_argument("--url", default=U)
ap.add_argument("--exclude", nargs="+", required=True, help="turn files whose 'session' names (eval and training traces) define excluded repos")
ap.add_argument("--n", type=int, default=140)
ap.add_argument("--resolved-frac", type=float, default=0.7)
ap.add_argument("--min-asst", type=int, default=6)
ap.add_argument("--max-groups", type=int, default=6)
ap.add_argument("--seed", type=int, default=4)
ap.add_argument("--out", required=True)
ap.add_argument("--avoid", nargs="*", default=[], help="earlier swe_fetch outputs: their repos are not picked again (disjoint data slices)")
ap.add_argument("--skip-groups", type=int, nargs="*", default=[])
ap.add_argument("--plan", action="store_true", help="metadata only: print the selection and the bytes a fetch would download, then stop")
a = ap.parse_args()


class RangeFile:
    """Seekable read-only file over HTTP Range requests (8 MB blocks, cached), enough for pyarrow's parquet reader."""
    def __init__(self, url, block=1 << 20):
        r = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
        self.url, self.size, self.pos, self.block, self.cache, self.fetched = r.geturl(), int(r.headers["Content-Length"]), 0, block, {}, 0

    def _blk(self, i):
        if i not in self.cache:
            lo = i * self.block; hi = min(self.size, lo + self.block) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={lo}-{hi}"})
            self.cache[i] = urllib.request.urlopen(req, timeout=120).read(); self.fetched += hi - lo + 1
            if len(self.cache) > 64:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[i]

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        out = bytearray()
        while n > 0:
            i, off = divmod(self.pos, self.block); b = self._blk(i)[off: off + n]
            out += b; self.pos += len(b); n -= len(b)
        return bytes(out)

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def readable(self):
        return True

    closed = False

    def close(self):
        pass


f = RangeFile(a.url); p = pq.ParquetFile(f); md = p.metadata
starts, s = [], 0
for i in range(md.num_row_groups):
    starts.append(s); s += md.row_group(i).num_rows
meta = p.read(columns=["trajectory_id", "instance_id", "repo", "resolved"]).to_pylist()
print(f"{md.num_rows} trajectories, {md.num_row_groups} row groups, metadata read {f.fetched / 1e6:.1f} MB", flush=True)

names = {json.loads(l)["session"].lower() for fn in a.exclude for l in open(fn, encoding="utf-8")}
def banned(repo):   # "encode/httpx" is excluded if any trace id starts with "httpx-"
    n = repo.split("/")[-1].lower()
    return any(t == n or t.startswith(n + "-") for t in names)
group_of = lambda i: max(g for g in range(len(starts)) if starts[g] <= i)
used = {json.loads(l)["repo"] for fn in a.avoid for l in open(fn, encoding="utf-8")}
ok = [(i, r) for i, r in enumerate(meta) if not banned(r["repo"]) and r["repo"] not in used]
print(f"excluded repos (eval/training overlap): {sorted({r['repo'] for r in meta if banned(r['repo'])})}", flush=True)

rng = random.Random(a.seed)
by_group = collections.defaultdict(list)
for i, r in ok:
    by_group[group_of(i)].append((i, r))
want_res = int(round(a.n * 1.5 * a.resolved_frac)); want_un = int(a.n * 1.5) - want_res   # 1.5x: some fail --min-asst
picked, repos, groups = [], set(), []
order = [g for g in sorted(by_group, key=lambda g: -len({r["repo"] for _, r in by_group[g]})) if g not in a.skip_groups]
for g in order[: a.max_groups]:
    groups.append(g); rows = by_group[g][:]; rng.shuffle(rows)
    for i, r in rows:
        if r["repo"] in repos:
            continue
        nres = sum(bool(x["resolved"]) for _, x in picked)
        if (bool(r["resolved"]) and nres < want_res) or (not r["resolved"] and len(picked) - nres < want_un):
            picked.append((i, r)); repos.add(r["repo"])
    if len(picked) >= want_res + want_un:
        break
print(f"candidates {len(picked)} from row groups {groups} ({len(repos)} repos)", flush=True)
cmb = sum(md.row_group(g).column(j).total_compressed_size for g in groups for j in range(md.num_columns)) / 1e6
print(f"PLAN fetch {len(groups)} row groups = {cmb:.0f} MB compressed (whole file {f.size / 1e6:.0f} MB); metadata read so far {f.fetched / 1e6:.1f} MB", flush=True)
if a.plan:
    raise SystemExit

want = {i for i, _ in picked}; kept = []
f.block, f.cache = 8 << 20, {}   # large blocks for whole row groups
for g in sorted(groups):
    t = p.read_row_group(g, columns=["trajectory_id", "instance_id", "repo", "resolved", "trajectory", "tools"]).to_pylist()
    for k, r in enumerate(t):
        if starts[g] + k not in want:
            continue
        msgs = r["trajectory"] or []
        if sum(m["role"] == "assistant" for m in msgs) < a.min_asst:
            continue
        kept.append({"session": "swe-" + r["trajectory_id"], "repo": r["repo"], "instance_id": r["instance_id"], "resolved": bool(r["resolved"]),
                     "tools": r["tools"], "messages": msgs})
    print(f"row group {g}: kept so far {len(kept)}, fetched {f.fetched / 1e6:.0f} MB", flush=True)
rng.shuffle(kept); kept = kept[: a.n]
with open(a.out, "w", encoding="utf-8") as fh:
    for r in kept:
        fh.write(json.dumps(r) + "\n")
asst = [sum(m["role"] == "assistant" for m in r["messages"]) for r in kept]
print(f"SWE_FETCH wrote {len(kept)} trajectories ({sum(r['resolved'] for r in kept)} resolved), repos {len({r['repo'] for r in kept})}, "
      f"assistant msgs median {sorted(asst)[len(asst) // 2] if asst else 0}, total downloaded {f.fetched / 1e6:.0f} MB", flush=True)

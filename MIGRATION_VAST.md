# Migration plan — RunPod → Vast.ai

Status: PROPOSED, nothing implemented. Written 2026-08-03 against the live Vast
API reference (docs.vast.ai) and `pipeline/runpod_infra.py` @ b1ed88e.

The guiding rule is the one this repo already runs on: **one unvalidated change
at a time, cheapest validation first.** Every expensive lesson currently encoded
in `runpod_infra.py` comments (the sshd-killing apt tree, the CPU-only-torch week,
the disk exhaustion that broke seven runs, the 4090 that billed top rate at 7%
utilisation) is provider-independent and must survive the move untouched.

---

## 0. TL;DR

| | RunPod (today) | Vast.ai (after) |
|---|---|---|
| Provision | `POST /pods` with an ordered `gpuTypeIds` allowlist | search `POST /bundles` → accept `PUT /asks/{offer_id}` |
| Cost control | ordered GPU-name list + `gpuTypePriority: custom` | `dph_total lte max_price` filter + sort ascending |
| Driver filter | `allowedCudaVersions: [12.8,12.9,13.0]` | `cuda_max_good gte 12.8` |
| Auto-kill | **server-side** — `runpodctl remove pod` on the box | **none exists** → replaced by 3 local/on-box layers (§4) |
| Billing dimensions | $/hr (RAM+vCPU bundled) | $/hr **+ storage $/GB/hr + bandwidth $/GB** (§5) |
| GPU identity | not retrievable (empty `machine` object) | `gpu_name` **and** `dph_total` both returned |

Two things get **better** (cost is a first-class filter instead of a proxy; the
card's identity is finally visible). One thing gets **worse** and needs real
engineering: there is no server-side self-destruct. That is §4 and it is the only
part of this plan that is not mechanical.

---

## 1. Architecture — a thin provider seam, not a rewrite

`pipeline/runpod_infra.py` is 1136 lines, of which roughly 75% is
provider-agnostic orchestration: the state-file protocol, deadline maths,
rsync/exclude rules, `REMOTE_SETUP`, `apt_setup`, `REMOTE_TASK`,
`engine_install_cmd`, `_verify_ready`, `free_engine`, `--reuse`/`--keep-alive`,
overlay forwarding, `setup_check`, `smoke_test`. Copying that file to
`vast_infra.py` would fork the code that holds all the safety, and the two copies
would drift on the first fix.

```
pipeline/
  remote_infra.py        # git mv from runpod_infra.py — orchestration, unchanged
  cloud/
    __init__.py          # provider(cfg) -> module, from cfg["remote"]["provider"]
    vast.py              # NEW  (~250 lines)
    runpod.py            # the API-touching bits moved out of runpod_infra.py
```

Provider module contract — everything else stays in `remote_infra.py`:

```python
KEY_ENV: str                                  # "VAST_API_KEY" / "RUNPOD_API_KEY"
DEFAULTS: dict                                # provider-specific config defaults
create(rp: dict) -> dict                      # -> {"id": str, ...}; id persisted IMMEDIATELY
get(iid: str) -> dict
destroy(iid: str) -> None                     # raise RuntimeError; "HTTP 404" in str(e) => gone
is_gone(info: dict) -> bool                   # status-string normalisation
list_all() -> list[dict]                      # leak check for `remote status`
ssh_target(info: dict) -> tuple[str, int] | None
price(info: dict) -> float | None             # $/hr actually being billed
arm_watchdog(rp, host, port, seconds) -> None
```

`terminate_pod()`'s retry-then-VERIFY loop stays in `remote_infra.py` and drives
`destroy` + `get` + `is_gone`. That loop is the single most important piece of
code in the module and must not be reimplemented per provider.

Keep `cloud/runpod.py` alive through validation; delete it (and `RUNPOD_API_KEY`)
after three green Vast runs. Rationale: this project has lost runs to capacity
before, and a marketplace is a different capacity risk than a datacenter.

---

## 2. Provisioning — search, rank, accept

RunPod takes a GPU-type allowlist and schedules for you. Vast is a marketplace:
you query offers and accept one. That inverts the cost problem in our favour —
`gpu_type_ids` + `gpu_type_priority: custom` was an elaborate proxy for "pick the
cheapest adequate card", and it still landed on a 4090 for a week. On Vast, price
is a filter and a sort key.

**Search** — `POST https://console.vast.ai/api/v0/bundles`, `Authorization: Bearer`:

```jsonc
{
  "rentable":        {"eq": true},
  "type":            "ondemand",          // matches today's interruptible: false
  "num_gpus":        {"eq": 1},
  "gpu_ram":         {"gte": 16},         // measured peak ~12 GB (config.gpu.yaml)
  "cuda_max_good":   {"gte": 12.8},       // replaces allowed_cuda_versions
  "disk_space":      {"gte": 40},         // replaces container_disk_gb
  "inet_down":       {"gte": 500},        // kills the 4-min-vs-60-min torch lottery
  "inet_down_cost":  {"lte": 0.02},       // $/GB — a NEW cost line, see §5
  "direct_port_count": {"gte": 2},        // direct SSH; the proxy is too slow for rsync
  "static_ip":       {"eq": true},
  "reliability2":    {"gte": 0.98},
  "verified":        {"eq": true},        // closest analogue to cloud_type: SECURE
  "dph_total":       {"lte": 0.40},       // THE cost lever
  "order":           [["dph_total", "asc"]],
  "limit":           20
}
```

**Rank client-side** on *effective* $/hr, not `dph_total` alone:

```
eff = dph_total + storage_cost * disk_gb / 730 + est_gb_transferred * inet_down_cost / hours
```

Try the top `offer_attempts` (default 3) in order with `cancel_unavail: true`, so
a stale offer fails immediately instead of hanging — the marketplace equivalent
of RunPod's "no instances currently available".

**Accept** — `PUT /api/v0/asks/{offer_id}`:

```jsonc
{"image": "runpod/pytorch:1.1.0-cu1290-torch291-ubuntu2404",
 "disk": 40, "runtype": "ssh_direct", "label": "dubadabidu",
 "onstart": "<L3 watchdog, see §4>", "cancel_unavail": true,
 "target_state": "running"}
```

→ `{"success": true, "new_contract": 1234568}`

> **The single most dangerous bug in this migration:** the id to persist and
> later DELETE is `new_contract`, **not** the offer `id` you PUT to. Destroying
> the offer id silently does nothing and the instance bills until someone looks
> at the console. Guarded by a test (§7).

**Keep the image unchanged.** `runpod/pytorch:...` is a plain Docker Hub image and
runs on Vast. Swapping to `vastai/pytorch` at the same time as the provider would
confound the first failure. Revisit after the ladder in §8 is green.

**Drop the create-time `TRANSLATE_API_KEY` injection.** On Vast `env` is a string
of docker flags (`"-e K=V -p 22:22"`), not a dict, so injecting a secret there
means quoting it into a string that ends up on the instance object. `remote_run`
already exports the key inline over SSH at task time (`runpod_infra.py:920`), so
the create-time copy is redundant. Removing it is a free security win.

---

## 3. SSH

Vast attaches account-level SSH keys at create time, and
`POST /api/v0/instances/{id}/ssh` `{"ssh_key": "<pub>"}` attaches post-create for
Docker instances. Plan: human registers the pubkey once in the console (same
one-time step as RunPod today), **and** `create()` calls the attach endpoint
right after — idempotent, and it removes a class of "why can't I connect" from
the validation ladder.

`ssh_target()` for Vast — direct first, proxy as an explicitly-degraded fallback:

```python
def ssh_target(inst):
    m = (inst.get("ports") or {}).get("22/tcp") or []
    ip = (inst.get("public_ipaddr") or "").strip()
    if ip and m and m[0].get("HostPort"):
        return ip, int(m[0]["HostPort"])
    if inst.get("ssh_host") and inst.get("ssh_port"):
        log.warning("falling back to the Vast SSH PROXY — rsync will be slow")
        return inst["ssh_host"], int(inst["ssh_port"])
    return None
```

Must never raise on an unexpected shape. `test_ssh_target_string_ports_never_crash`
exists because that exact crash leaked a pod on the first smoke test; the Vast
version of that test is not optional.

**Readiness:** poll `GET /api/v0/instances/{id}` until `actual_status == "running"`
*and* `ssh_target()` resolves *and* `wait_ssh()` succeeds. Vast's `loading` state
includes the image pull, which on a slow host is minutes — raise
`provision_timeout_s` 420 → 900. Fail fast (and destroy) if `status_msg` carries
an image-pull or capacity error.

---

## 4. The watchdog gap — the one real regression

`arm_pod_watchdog()` works today because `runpodctl` is **self-authenticated on a
RunPod pod**: the box can delete itself with no credentials. Vast has no such
tool, **and no server-side scheduled-destroy or duration field on create**
(confirmed against the create-instance reference and the instances docs). A
credential-carrying watchdog is not an option — "the RunPod full-access key never
reaches the pod" is a stated invariant of this design and should stay one.

Four layers replace it. State plainly what each does and does not cover:

**L1 — client-side terminate (unchanged).** `finally` block, retried, verified,
state-file driven. Covers every normal and most abnormal exits.

**L2 — detached LOCAL reaper (new; this is the real replacement).** On provision,
spawn a fully detached process (`Popen(..., start_new_session=True)`, log to
`work/.reaper.log`) that sleeps to the deadline, re-reads the state file, and
DELETEs if the instance is still tracked. It survives `SIGKILL` of the CLI,
Ctrl-C, and the parent shell exiting — the exact failures `arm_pod_watchdog` was
written for. It does **not** survive laptop shutdown. Exits silently if the state
file is already gone.

**L3 — on-instance self-halt via `onstart` (new).**
`nohup sh -c 'sleep N; kill -9 -1' &`. Killing PID 1 exits the container, which
stops **GPU** billing. Storage keeps billing until destroy — at ~$0.10–0.20/GB/mo
on 40 GB that is roughly **$0.005/h**. So L3 turns a runaway from ~$0.40/h into
half a cent an hour. Not zero. Say so in the log line rather than implying the
box is gone.

**L4 — `remote status` becomes a hard check (new, cheap, high value).**
`GET /api/v0/instances` lists every instance on the account. Make `remote status`
exit non-zero when any instance exists that the state file does not track, print
the exact destroy command, and wire the same check into `dubadabidu doctor`.
On RunPod an orphan was bounded by the pod-side watchdog; on Vast nothing reaps
it, so the account-level sweep is now load-bearing.

**Honest summary for the docs:** *Vast's worst case is a ~$0.005/h storage leak
that the next `remote status` or `doctor` surfaces, instead of RunPod's
server-side self-destruct. This is the one dimension the migration makes weaker;
L2 and L4 buy most of it back.*

---

## 5. Cost model — three dimensions, not one

RunPod bundles vCPU/RAM into the GPU tier and bills $/hr. Vast bills **per second
across three dimensions, all host-set**: GPU $/hr, storage $/GB/hr (charged on
*stopped* instances too), and bandwidth $/GB **both directions**. Reported
bandwidth rates run around $2.50/100 GB and vary per host.

That matters here more than it looks. A fresh bootstrap pulls ~4–6 GB of CUDA
torch plus the qwen checkpoints — call it ~10 GB — so on a billed-bandwidth host
that is **~$0.25 per fresh instance, comparable to half an hour of GPU**. Hence
`inet_down_cost lte 0.02` in the search filter, preferring 0.

Consequences for `config.gpu.yaml`:

- **`assumed_price_per_hr: 0.70` becomes unnecessary.** Today the deadline is
  `min(budget/assumed_price, max_runtime_hours)` with a deliberately pessimistic
  assumed price (13% off by construction). On Vast the accepted offer's
  `dph_total` is known the moment the instance exists, so: provisional deadline
  from `max_price_per_hr` before create, **tightened to `budget/dph_total` after**.
  Strictly better than the RunPod path.
- **Never `stop`, always `destroy`.** Storage bills in the stopped state. The
  existing code has no stop path — keep it that way and say why.
- **The `network_volume_id` argument in config.gpu.yaml flips.** It currently
  argues against a persistent volume on RunPod cost grounds; with bandwidth
  billed, avoiding a repeat 10 GB download has real value.
  **Recommendation: still no volume** — lean harder on the existing
  `--reuse` + `--keep-alive` path, which now saves money as well as wall clock,
  and carries no region-lock or idle-storage cost.
- `container_disk_gb: 40` → `disk: 40` on create **plus** `disk_space gte 40` in
  the search filter (on Vast the disk ceiling comes from the offer). Do not cut
  it — that value is what broke seven consecutive runs on 2026-07-30.

---

## 6. Config surface

Rename `runpod:` → `remote:` so both providers coexist during validation.
`remote_infra` reads `cfg["remote"]` and falls back to `cfg["runpod"]`, so
nothing breaks mid-migration.

```yaml
remote:
  provider: vast                 # vast | runpod
  budget_usd: 10.0
  max_runtime_hours: 6.0         # keep — measured against a 1-hour lesson × 5 langs
  ssh_key: ~/.ssh/id_ed25519_vast
  ssh_user: root
  remote_dir: ~/dubadabidu
  provision_timeout_s: 900       # was 420; Vast's image pull is inside this window
  image: runpod/pytorch:1.1.0-cu1290-torch291-ubuntu2404   # unchanged on purpose
  engine_setup: {qwen: "..."}    # unchanged
  vast:
    max_price_per_hr: 0.40
    min_gpu_ram_gb: 16
    min_cuda: 12.8               # MUST move with the torch pin in REMOTE_SETUP
    min_inet_down_mbps: 500
    max_inet_down_cost_per_gb: 0.02
    min_reliability: 0.98
    verified_only: true
    disk_gb: 40
    offer_attempts: 3
  runpod: {...}                  # existing keys, verbatim, until deletion
```

State file: `work/.runpod_active.json` → `work/.remote_active.json`, and the
payload gains `"provider"`. **A state file written by a RunPod run must never be
handed to a Vast DELETE** — record the provider and refuse the mismatch. Read the
old filename during the transition so an orphaned RunPod pod is not stranded.
`.gitignore` keeps both lines for one release.

---

## 7. Tests (`tests/test_runpod.py`, 448 lines)

**Keep, repointed at `remote_infra`:** `_deadline` ×3, `engine_install_cmd`,
`_verify_ready` ×2, overlay forwarding, `_remote_stages`, torch-pin coupling,
the `torchaudio.load` AST guard, `max_runtime_hours` sizing, the experiments
`--reuse/--keep-alive` and `variant_key` collision tests.

**Rewrite:**
- `ssh_target` shape tests → Vast shapes (direct-port, proxy fallback, not-ready,
  and the never-raise case). RunPod shapes stay under `cloud/runpod.py` while it
  lives.
- `test_gpu_type_priority_defaults_to_custom...` + `test_config_puts_cheap_gpus_first_and_4090_last`
  → **`test_offer_query_is_cost_bounded`**: the search body must carry a
  `dph_total lte max_price_per_hr` and sort ascending. Same intent — never
  silently land on the expensive card — asserted at the layer that now decides it.
- `test_torch_pin_and_cuda_filter_move_together` → same assertion against
  `vast.min_cuda >= 12.6`. `test_torch_wheel_matches_the_oldest_allowed_driver`
  becomes `cu128 wheel <= min_cuda`, which is *simpler* on Vast: one number, not
  a list.
- `test_provisioning_logs_the_price_not_a_gpu_name` → **log both.** Vast returns
  `gpu_name` and `dph_total`; the RunPod workaround existed only because the card
  was unknowable.

**New:**
- `test_create_persists_new_contract_not_offer_id` — the §2 footgun.
- `test_state_file_records_the_provider` — no cross-provider DELETE.
- `test_reaper_is_detached_from_the_parent_session` — L2 is only a backstop if it
  outlives `SIGKILL`; assert `start_new_session=True`.
- `test_no_secret_in_the_create_payload` — the §2 env-string change.
- `test_search_filters_bandwidth_cost` — §5's new cost line, easy to drop silently.

---

## 8. Validation ladder — cheapest first, stop at the first surprise

| # | Step | Cost | Proves |
|---|---|---|---|
| 1 | `remote status` with zero instances | $0 | auth, list parsing, L4 |
| 2 | `remote offers` (NEW cmd — print top 10 ranked, effective $/hr) | $0 | the search query and the cost ranking, **before spending anything** |
| 3 | `remote smoke` — create → dump raw instance JSON → `nvidia-smi` → destroy | ~$0.02 | `new_contract`, `ssh_target`, direct ports, status strings, destroy-verify |
| 4 | **kill-switch drill** — provision, `kill -9` the CLI, confirm the reaper destroys it | ~$0.02 | L2, i.e. the layer replacing `runpodctl`. **Blocking gate for anything long.** |
| 5 | `remote setup-check --overlay config.gpu.yaml` | ~$0.30 | torch 2.11 + CUDA true + `qwen_tts` imports on Vast hardware |
| 6 | `remote bakeoff sketch60 --langs en` | ~$0.30 | scorecard comparable to the last RunPod run (see risk below) |
| 7 | one full production lesson, `--budget 3` | ~$1 | end to end |
| 8 | delete `cloud/runpod.py`, `RUNPOD_API_KEY`, the compat shims | $0 | — |

Step 2 is the highest-value new command in this plan: it makes the cost decision
inspectable for free. The week spent unknowingly on a 4090 was invisible for
exactly the lack of it.

---

## 9. Docs and stragglers

- `README.md:72,84,88,210-215`
- `AUTOPILOT.md:50-52,68,78-79` (the infra-agent section names runpodctl)
- `IMPROVEMENT_PLAN.md:1,10`
- `THIRD_PARTY.md:20-31` — **"torch stays at 2.6.0 — RunPod's HOST DRIVER is the
  ceiling"** is already stale (the pin is 2.11.0) and its reasoning is
  RunPod-specific. Rewrite around `cuda_max_good`. Also `:62`, `:214`.
- `pyproject.toml:9-12` comments (same stale CUDA-12.4 claim)
- `.env.example:8-10` — `RUNPOD_API_KEY` → `VAST_API_KEY`
- `.gitignore:27-28`, `.github/workflows/tests.yml:20`, `specs/batch.yaml:15`,
  `pipeline/device.py:53`, `dub.py:106-115` (doctor's remote-readiness block)
- `pipeline/experiments.py` — no API surface, but its comments say "pod"

---

## 10. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| No server-side auto-destroy | **High** | §4 L2/L3/L4; step 4 of the ladder is a blocking gate |
| Persisting the offer id instead of `new_contract` | **High** | test + step 3 raw dump |
| Marketplace host quality — RunPod SECURE was chosen after community spot pods died 4× | Medium | `verified: true` + `reliability2 >= 0.98`; the manifest content-hash cache already makes a killed run resumable |
| Bandwidth billing, ~$0.25/fresh bootstrap | Medium | `inet_down_cost` filter; prefer `--reuse` |
| No `direct_port_count` → SSH proxy → slow 238 MB rsync | Medium | filter on it; log loudly on fallback |
| API version drift: create is `/api/v0/asks/{id}`, but show-instances documents `/api/v1/instances` | Medium | pin v0 everywhere, confirm both in step 3 |
| Different GPU → metrics not comparable to historical scorecards | Medium | step 6 compares one language against the last RunPod bakeoff; note `floor_*.wav` is deliberately synced so `sim_cal` stays comparable |
| Image expects RunPod's env / no sshd under `ssh_direct` | Low | step 3 catches it; `vastai/pytorch` is the fallback, changed *after* the ladder is green |

---

## 11. Effort

- Provider seam + `cloud/vast.py`: ~1 day
- Test rework: ~½ day
- Validation ladder: ~$1 and ~½ day of wall clock
- Docs: ~2 h

Nothing here needs a GPU until step 3, and steps 1–2 are free.

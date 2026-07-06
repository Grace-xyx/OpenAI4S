# Remote GPU compute

Model-weight-bound work runs its heavy step on a remote GPU, not the local kernel. There are two paths.

## 1 · `host.compute` — general BYOC / SSH job dispatcher

A job is dispatched non-blocking (`create → submit_job → wait → result`): the daemon stages inputs, runs the job in a confined sandbox, and harvests `out.tar.gz` back into the workspace under `hpc/<job_id>/`. Two provider families are built in:

- **`ssh:<alias>`** — run jobs over an SSH connection to a machine you already have ([`skills/remote-compute-ssh`](../skills/remote-compute-ssh)).
- **`byoc:<id>`** — a bring-your-own-compute provider discovered from `skills/remote-compute-<id>/` (`provider.json` + `provider.py`).

The bundled **NVIDIA NIM** provider ([`skills/remote-compute-nvidia`](../skills/remote-compute-nvidia)) uses only the `docker` CLI (no SDK):

| form | needs | where the job runs |
|---|---|---|
| `self_hosted` | Docker + NVIDIA Container Toolkit + `NGC_API_KEY` | an `nvcr.io` NIM container on a local GPU (`--gpus all`) |
| `hosted` | Docker + an `nvapi-…` `NVIDIA_API_KEY` | the managed `integrate.api.nvidia.com` gateway (no local GPU) |

```python
c   = host.compute.create("byoc:nvidia", provider_params={"nvidia": {"mode": "hosted"}})
job = c.submit_job(intent="run esmfold2 on 1 seq", command="python run_esmfold.py ./seq.fasta",
                   inputs=[{"src": "seq.fasta"}], outputs=["*.pdb"], timeout_seconds=3600)
result = job.result()   # non-blocking once the compute_done notification arrives
```

The daemon forwards **only** the keys a provider declares in its `provider.json` `secret_env` into the confined job sandbox (over the helper's stdin) — never your whole environment.

## 2 · `host.fold` / `host.score_mutations` — purpose-built science services over SSH

- **`host.fold(seq)`** runs **real single-sequence Protenix (AlphaFold3-class) inference** on a GPU host (the in-repo runner is [`scripts/fold_remote.sh`](../scripts/fold_remote.sh)). It is single-sequence (no MSA) and returns a PDB structure with per-residue pLDDT. The reference host is an 8×A100-80GB box; a single fold uses one GPU.
- **`host.score_mutations(...)`** runs **real ESM masked-marginal** variant scoring.

Both are governed by a strict **no-fabrication policy** — when no host is configured they *refuse and error* rather than invent a structure or scores — and each result records a reproducibility-provenance snapshot into its artifact.

### Auto-provisioning

You don't have to hand-configure model services. Register an SSH GPU host in **Settings → Compute**; when a GPU/protein task needs a service that isn't set up yet, the agent calls the built-in **`REMOTE_GPU_PROVISIONER`** specialist, which SSHes in, installs the real wrappers, **verifies** them, and only then **registers** the capability (no fake registration — registration only succeeds after the remote service is confirmed). Inspect the registry with `host.remote_gpu_status()`. Prefer **project-scoped** permission rules for remote work (e.g. `ssh my-gpu-host *`).

### Config

| env var | for |
|---|---|
| `NVIDIA_API_KEY` / `NGC_API_KEY` | NVIDIA NIM (`hosted` / `self_hosted`) |
| `OPENAI4S_FOLD_SSH` · `OPENAI4S_FOLD_SCRIPT` · `OPENAI4S_FOLD_JOBS_DIR` | `host.fold` over SSH |
| `OPENAI4S_ESM_JOBS_DIR` | `host.score_mutations` scratch dir |

SSH auth stays in your `~/.ssh/config` / ssh-agent — the registry stores no secrets.

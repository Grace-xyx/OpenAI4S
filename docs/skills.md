# Skills — capabilities as code, not schemas

A Skill is a directory under [`skills/`](../skills):

```
skills/example_stats/
    SKILL.md      recipe-centric doc (code examples, not a JSON schema)
    kernel.py     importable sidecar module (helper functions)
```

Skills are consumed by **writing code**. The loader surfaces each `SKILL.md` to the model via *progressive disclosure* (only a one-line summary up front; the full doc is fetched on demand with `host.search_skills(query)`), the kernel adds `skills/` to `sys.path`, and the agent runs e.g. `from example_stats.kernel import summary`. A Skill's capability lands as **callable Python inside the kernel** — the same principle as the core paradigm, not another tool schema.

## Bundled Skills (24)

| category | Skills |
|---|---|
| **Structure prediction** (GPU) | `alphafold2` · `openfold3` · `boltz` · `chai1` · `esmfold2` |
| **Sequence / omics / docking** (GPU) | `fair-esm2` · `evo2` · `borzoi` · `scgpt` · `scvi-tools` · `diffdock` |
| **Protein design** (GPU) | `proteinmpnn` · `ligandmpnn` · `solublempnn` |
| **Research workflow** | `literature-review` · `pdf-explore` · `paper-narrative` · `figure-composer` · `figure-style` · `indication-dossier` |
| **Platform** | `remote-compute-nvidia` · `remote-compute-ssh` · `using-model-endpoint` |

`example_stats` is the reference example Skill (pure-stdlib descriptive-statistics helpers).

## Writing a Skill

1. Create `skills/<name>/SKILL.md` with a short YAML frontmatter (`name`, `description`, optional `origin`, `category`, `requirements: [gpu]`) followed by a body of **runnable code examples**.
2. Optionally add a `kernel.py` with importable helper functions.
3. That's it — the loader discovers it on the next run and surfaces its one-line summary to the agent. Bundled skills (`origin: openai4s`) are read-only; skills you author or import are editable from the UI (**Customize → Skills**).

GPU/model Skills (`requirements: [gpu]`) run their heavy step on a remote GPU through [`host.compute`](compute.md); everything else runs directly in the kernel.

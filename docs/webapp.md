# The web app

`openai4s serve` starts a full **scientific coding-agent workspace** (pure-stdlib HTTP + WebSocket, no framework) at `http://127.0.0.1:8760/`.

- **Projects & sessions** — folder-grouped and date-grouped; ⌘K opens a **command palette** over sessions, artifacts, Skills, and actions.
- **Live agent turns** — prose streams token-by-token over WebSocket; each code cell shows as an activity card; every file the agent writes is auto-captured as a **versioned artifact** (open images, tables, Markdown, and `.pdb`/`.cif` structures in a built-in **3Dmol** viewer) with version history + one-click revert.
- **Notebook** — a live REPL sharing the agent's kernel, with Stop / Start / Restart and an install-extra-packages box; figures appear on the running cell as the agent produces them.
- **Background & resume** — a turn keeps running server-side even if you close the tab; reopen and the live stream replays from where it was.
- **Customize** — Skills, Agents, Compute (host info, packages, tracked background Jobs), Network, cross-session Memory, and Models.
- **Plan / Explore modes**, voice dictation, file paste / drag-drop, an animated canvas favicon, bilingual **中文 / EN** UI, and one-click **export a session to Markdown**.

## Demo session (seeded on first boot)

On first boot the app seeds a **live demo session** — a NIF3/DUF34 protein-family analysis that calls the real **UniProt** and **RCSB PDB** APIs plus a bundled MCP connector, running six deterministic notebook cells (**no LLM key required**) to produce real artifacts:

1. A UniProt REST pull of the NIF3/DUF34 family sequences.
2. A bundled MCP connector call.
3. A Kyte-Doolittle hydropathy plot.
4. A `family_biochemistry.csv` (length / MW / pI / GRAVY / pairwise % identity).
5. An RCSB PDB search + coordinate download → `nif3_structure.pdb` (opens in the 3Dmol viewer).
6. A reproducible `nif3_report.md`.

Everything is real data under a strict no-fabrication policy — steps that can't reach the network are honestly skipped, never faked.

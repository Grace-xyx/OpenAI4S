#!/usr/bin/env bash
# os-fold: single-sequence protein structure prediction on the local A100s.
# Real Protenix (AF3-class) inference, fully offline (cached weights + CCD).
#
# Usage:
#   fold.sh --seq <AA_SEQUENCE> [--name NAME] [--out DIR] [--gpu N]
#           [--cycle C] [--step P] [--sample E]
#   fold.sh --fasta path.fasta [ ... ]
#
# Emits into <out>: model.pdb, model.cif, confidence.json, plddt.csv
# and prints a one-line JSON manifest between ===FOLD_RESULT_JSON=== markers.
set -euo pipefail

OSF_ROOT="${OSF_ROOT:-/opt/os-fold}"
PROOT="$OSF_ROOT/proot"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
MODEL=protenix_base_default_v1.0.0

SEQ=""; FASTA=""; NAME="job"; OUT="."; GPU="0"; CYCLE=10; STEP=40; SAMPLE=1
while [ $# -gt 0 ]; do
  case "$1" in
    --seq) SEQ="$2"; shift 2;;
    --fasta) FASTA="$2"; shift 2;;
    --name) NAME="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    --cycle) CYCLE="$2"; shift 2;;
    --step) STEP="$2"; shift 2;;
    --sample) SAMPLE="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

if [ -n "$FASTA" ]; then
  SEQ=$(grep -v '^>' "$FASTA" | tr -d '\n\r \t')
fi
if [ -z "$SEQ" ]; then echo "error: no sequence (--seq or --fasta)" >&2; exit 2; fi
# sanitize: uppercase, strip non-AA
SEQ=$(printf '%s' "$SEQ" | tr 'a-z' 'A-Z' | tr -cd 'ACDEFGHIKLMNPQRSTVWY')
NAME=$(printf '%s' "$NAME" | tr -cd 'A-Za-z0-9_-'); NAME=${NAME:-job}

mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
WORK="$OUT/_ptx"; mkdir -p "$WORK"

# offline env
[ -f "$CONDA_SH" ] && source "$CONDA_SH"
conda activate ptx
export PROTENIX_ROOT_DIR="$PROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

printf '[{"sequences":[{"proteinChain":{"sequence":"%s","count":1}}],"modelSeeds":[],"name":"%s"}]\n' \
  "$SEQ" "$NAME" > "$WORK/input.json"

echo ">>> os-fold: folding $NAME (len ${#SEQ}) on GPU $GPU (cycle=$CYCLE step=$STEP sample=$SAMPLE)" >&2
protenix pred -i "$WORK/input.json" -o "$WORK/out" \
  -s 101 -c "$CYCLE" -p "$STEP" -e "$SAMPLE" -n "$MODEL" \
  --use_msa false --use_default_params true >&2

# locate best sample (sample_0)
CIF=$(find "$WORK/out" -name "${NAME}_sample_0.cif" | head -1)
CONF=$(find "$WORK/out" -name "${NAME}_summary_confidence_sample_0.json" | head -1)
if [ -z "$CIF" ]; then echo "error: no CIF produced" >&2; exit 1; fi

cp -f "$CONF" "$OUT/confidence.json"
cp -f "$CIF"  "$OUT/model.cif"

python - "$CIF" "$OUT" <<'PY' >&2
import sys, gemmi, csv, json
cif, out = sys.argv[1], sys.argv[2]
st = gemmi.read_structure(cif); st.setup_entities()
open(f"{out}/model.pdb","w").write(st.make_pdb_string())
rows=[]
for model in st:
    for chain in model:
        for res in chain:
            ca = res.find_atom("CA", "*")
            if ca is not None:
                rows.append((chain.name, res.seqid.num, res.name, round(ca.b_iso,2)))
    break
with open(f"{out}/plddt.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["chain","resid","resname","plddt"]); w.writerows(rows)
print(f"wrote model.pdb ({len(rows)} residues) + plddt.csv")
PY

# final manifest for the agent
PLDDT=$(python -c "import json;print(round(json.load(open('$OUT/confidence.json'))['plddt'],2))")
PTM=$(python -c "import json;print(round(json.load(open('$OUT/confidence.json'))['ptm'],3))")
NRES=$(( $(wc -l < "$OUT/plddt.csv") - 1 ))
echo "===FOLD_RESULT_JSON==="
printf '{"name":"%s","length":%d,"residues_modeled":%d,"mean_plddt":%s,"ptm":%s,"model_pdb":"%s/model.pdb","model_cif":"%s/model.cif","plddt_csv":"%s/plddt.csv","confidence_json":"%s/confidence.json","engine":"protenix_base_default_v1.0.0","msa":false}\n' \
  "$NAME" "${#SEQ}" "$NRES" "$PLDDT" "$PTM" "$OUT" "$OUT" "$OUT" "$OUT"
echo "===END_FOLD_RESULT_JSON==="

# Inline the deliverables as base64 in stdout so the caller gets everything back
# through the harvested stdout log (no separate file download needed).
echo "===FOLD_PDB_B64==="; base64 -w0 "$OUT/model.pdb"; echo
echo "===FOLD_PLDDT_CSV_B64==="; base64 -w0 "$OUT/plddt.csv"; echo
echo "===FOLD_CONFIDENCE_JSON_B64==="; base64 -w0 "$OUT/confidence.json"; echo
echo "===FOLD_DONE==="

"""
Generate DePlag_DeBERTa_Kaggle.ipynb from DePlag_DeBERTa_Training.ipynb.
Replaces ALL Colab-specific content with Kaggle equivalents.
"""
import json, copy

with open("DePlag_DeBERTa_Training.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

nb = copy.deepcopy(nb)

# ── 1. Metadata ────────────────────────────────────────────────────────────────
nb["metadata"] = {
    "kaggle": {
        "accelerator": "gpu",
        "dataSources": [],
        "dockerImageVersionId": 30919,
        "isGpuEnabled": True,
        "isInternetEnabled": True,
        "language": "python",
        "sourceType": "notebook"
    },
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10.14"
    }
}

cells = nb["cells"]

# ── 2. Title cell (index 0, markdown) ─────────────────────────────────────────
cells[0]["source"] = [
    "# DePlag \u2014 DeBERTa-v3 Cross-Encoder Training (Kaggle)\n",
    "\n",
    "**Fixes 7 root causes** to reach 93%+ accuracy.\n",
    "\n",
    "**Runtime:** Settings \u2192 Accelerator \u2192 **GPU (T4 or P100)**  \u2714\ufe0f\n",
    "\n",
    "**Internet:** Settings \u2192 **Internet on** (needed to download HuggingFace datasets)\n",
    "\n",
    "**Key fixes:** No punctuation stripping, official dataset splits, source balancing, "
    "STS-B gap filtering, leakage prevention, tuned hyperparameters.\n",
    "\n",
    "> Artifacts are saved to `/kaggle/working/DePlag_DeBERTa/` and downloadable from the **Output tab**."
]

# ── 3. Find cells by content keyword ──────────────────────────────────────────
def find_cell(keyword, cell_type="code"):
    for i, c in enumerate(cells):
        if c["cell_type"] == cell_type and any(keyword in line for line in c["source"]):
            return i
    return None

def find_md(keyword):
    for i, c in enumerate(cells):
        if c["cell_type"] == "markdown" and any(keyword in line for line in c["source"]):
            return i
    return None

# ── 4. Replace "Mount Google Drive" markdown header ───────────────────────────
md_drive = find_md("Mount Google Drive")
if md_drive is not None:
    cells[md_drive]["source"] = ["## Cell 2: Setup Output Directory (Kaggle)"]

# ── 5. Replace the Google Drive mount CODE cell ───────────────────────────────
drive_cell = find_cell("google.colab")
if drive_cell is not None:
    cells[drive_cell]["source"] = [
        "import os\n",
        "\n",
        "# Kaggle: /kaggle/working/ persists for the session.\n",
        "# After the run, download artifacts from the Output tab,\n",
        "# or publish them as a Kaggle Dataset for reuse.\n",
        "SAVE_DIR = \"/kaggle/working/DePlag_DeBERTa\"\n",
        "os.makedirs(SAVE_DIR, exist_ok=True)\n",
        "print(f\"Artifacts -> {SAVE_DIR}\")"
    ]

# ── 6. Replace /content/ paths in ALL code cells ──────────────────────────────
for cell in cells:
    if cell["cell_type"] == "code":
        cell["source"] = [
            line
            .replace("/content/deberta_checkpoints", "/kaggle/working/deberta_checkpoints")
            .replace("/content/drive/MyDrive/DePlag_DeBERTa", "/kaggle/working/DePlag_DeBERTa")
            .replace("/content/drive", "/kaggle/working")
            .replace("/content/", "/kaggle/working/")
            for line in cell["source"]
        ]

# ── 7. Replace Deployment Instructions (last code cell) ───────────────────────
last_code = None
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        last_code = i
if last_code is not None:
    cells[last_code]["source"] = [
        'print("""\n',
        "========================================================\n",
        "  DEPLOYMENT INSTRUCTIONS (Kaggle)\n",
        "========================================================\n",
        "\n",
        "  1. After training completes, go to the Output tab.\n",
        "     Download these files:\n",
        "       DePlag_DeBERTa/model/      <- all model weights\n",
        "       DePlag_DeBERTa/results.json\n",
        "       DePlag_DeBERTa/test_predictions.csv\n",
        "\n",
        "  2. (Optional) Publish as a Kaggle Dataset for reuse:\n",
        "       Output tab -> New Dataset -> deplag-deberta-model\n",
        "       Then reference it in other notebooks via:\n",
        "       /kaggle/input/deplag-deberta-model/\n",
        "\n",
        "  3. Copy to your local DePlag project:\n",
        "       cp -r DePlag_DeBERTa/model/* models/\n",
        "       cp DePlag_DeBERTa/results.json models/\n",
        "       cp DePlag_DeBERTa/test_predictions.csv models/\n",
        "\n",
        "  4. Run: python server.py\n",
        "\n",
        "  IMPORTANT: Model trained WITHOUT punctuation stripping.\n",
        "  Ensure nlp/preprocess.py normalize() uses whitespace-only\n",
        "  cleaning (no re.sub punctuation removal).\n",
        '========================================================\n""")'
    ]

# ── 8. Write output ────────────────────────────────────────────────────────────
out_path = "DePlag_DeBERTa_Kaggle.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Created: {out_path}")

# ── 9. Sanity check ───────────────────────────────────────────────────────────
with open(out_path, encoding="utf-8") as f:
    check = json.load(f)

colab_refs = []
for i, c in enumerate(check["cells"]):
    for line in c["source"]:
        if "google.colab" in line or "/content/drive" in line:
            colab_refs.append((i, line.strip()))

if colab_refs:
    print("WARNING: Remaining Colab references:")
    for ref in colab_refs:
        print(f"  Cell {ref[0]}: {ref[1]}")
else:
    print("OK: No Colab references found.")

print(f"Total cells: {len(check['cells'])}")
print(f"Metadata keys: {list(check['metadata'].keys())}")

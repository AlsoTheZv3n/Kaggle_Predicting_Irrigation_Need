"""
Lokaler End-to-End-Lauf des Notebooks mit den echten Kaggle-Daten.
Patcht das Run-Profil auf "Schnell" damit der Test in <30 min durchläuft.
"""
import sys, os, time
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NB_IN  = "kaggle_s6e4_irrigation.ipynb"
NB_OUT = "kaggle_s6e4_irrigation.local.ipynb"

# Schnell-Profil für lokalen Test
PATCHES = [
    ("SEEDS    = [42, 2024]",       "SEEDS    = [42]"),
    ("N_SPLITS = 5",                "N_SPLITS = 3"),
    ("N_ITERS  = 1500",             "N_ITERS  = 500"),
    ("PSEUDO_CONF_THRESHOLD = 0.95", "PSEUDO_CONF_THRESHOLD = 0.95"),
]

def main():
    nb = nbformat.read(NB_IN, as_version=4)
    print(f"Loaded {NB_IN}: {len(nb.cells)} cells")

    patched = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        new_src = cell.source
        for old, new in PATCHES:
            if old in new_src:
                new_src = new_src.replace(old, new)
                patched += 1
        cell.source = new_src
    print(f"✓ {patched} patches applied (Schnell-Profil)")

    client = NotebookClient(nb, timeout=3600, kernel_name="kaggle-s6e4")
    t0 = time.time()
    try:
        client.execute()
    except CellExecutionError:
        print("\n❌ EXECUTION FAILED")
        nbformat.write(nb, NB_OUT)
        for i, c in enumerate(nb.cells):
            if c.cell_type != "code":
                continue
            for o in c.get("outputs", []):
                if o.get("output_type") == "error":
                    print(f"\n--- Cell {i} ERROR ---")
                    print("\n".join(o.get("traceback", []))[-3000:])
        sys.exit(1)
    dt = time.time() - t0
    print(f"\n✅ Notebook executed in {dt/60:.1f} min")
    nbformat.write(nb, NB_OUT)

    print("\n========== KEY OUTPUTS ==========")
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                txt = o.get("text", "")
                if any(k in txt for k in ["BalAcc", "OOF", "Optimal", "Ensemble",
                                           "submission", "scaling", "weights",
                                           "Train", "Test", "Features", "DATA_DIR",
                                           "Class mapping", "Pseudo", "Round",
                                           "is_generated", "External", "LGB seed",
                                           "CB seed", "XGB seed", "Round 1",
                                           "Round 2", "Augmented", "Final"]):
                    print(f"\n--- Cell {i} ---")
                    print(txt.rstrip()[:2500])

    if os.path.exists("submission.csv"):
        import pandas as pd
        s = pd.read_csv("submission.csv")
        print("\n========== submission.csv ==========")
        print(f"shape: {s.shape}")
        print(f"label dist:\n{s['Irrigation_Need'].value_counts().to_dict()}")
        assert list(s.columns) == ["id", "Irrigation_Need"]
        assert set(s["Irrigation_Need"].unique()) <= {"Low","Medium","High"}
        assert len(s) == 270000
        print("\n✅ submission.csv valid (270 000 rows)")

if __name__ == "__main__":
    main()

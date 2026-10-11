"""Rebuild report tables from the saved notebook and export its original plots."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ml_gay_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "report_assets"
NOTEBOOK = ROOT / "completion_breast_cancer_matrix.ipynb"


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    plot_names = {
        (6, 3): "variance_cumulative.png",
        (6, 4): "feature_correlations.png",
        (18, 2): "pca_rank_validation.png",
        (20, 2): "pca_svd_rank_validation.png",
        (22, 2): "common_rule_equivalence.png",
        (24, 1): "method_errors.png",
        (25, 1): "per_feature_errors.png",
        (28, 1): "feature_group_sensitivity.png",
        (30, 1): "mask_probability_sensitivity.png",
    }
    for (cell_id, output_id), filename in plot_names.items():
        images = [o["data"]["image/png"] for o in notebook["cells"][cell_id]["outputs"] if "image/png" in o.get("data", {})]
        image_index = 1 if filename == "feature_correlations.png" else 0
        image = images[image_index]
        (ASSETS / filename).write_bytes(base64.b64decode("".join(image)))

    # Execute the notebook's code in order so the report also has exact means,
    # mask examples and tables that were not printed in the saved outputs.
    namespace: dict[str, object] = {"__name__": "__main__"}
    plt.show = lambda *args, **kwargs: plt.close("all")
    for cell_id, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        code = "".join(cell["source"])
        exec(compile(code, f"{NOTEBOOK}:cell_{cell_id}", "exec"), namespace)
        namespace["display"] = lambda *args, **kwargs: None

    names = namespace["feature_names"]
    targets = namespace["selected_features"]
    x_train = namespace["X_train"]
    x_test = namespace["X_test"]
    m_train = namespace["M_train"]
    m_test = namespace["M_test"]
    means = namespace["column_means"]
    masked_means = namespace["masked_train_means"]

    namespace["results"].to_csv(ASSETS / "results.csv", index=False)
    namespace["per_feature"].to_csv(ASSETS / "per_feature.csv", index=False)
    namespace["rank_scores"].to_csv(ASSETS / "pca_rank_scores.csv", index=False)
    namespace["svd_rank_scores"].to_csv(ASSETS / "svd_rank_scores.csv", index=False)
    namespace["common_completion_results"].to_csv(
        ASSETS / "common_completion_results.csv", index=False
    )
    namespace["pattern_results"].to_csv(ASSETS / "pattern_results.csv", index=False)
    namespace["sensitivity"].to_csv(ASSETS / "sensitivity.csv", index=False)

    results = namespace["results"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, split in zip(axes, ("Validation", "Test")):
        rows = results[results["split"] == split]
        values = np.array(
            [
                [rows.loc[rows["metodo"] == f"{method} (r=6)", "RMSE"].item(),
                 rows.loc[rows["metodo"] == f"{method} (r=24, validation)", "RMSE"].item()]
                for method in ("PCA", "SVD")
            ]
        )
        x = np.arange(2)
        ax.bar(x - .18, values[:, 0], .36, label="r = 6", color="#94a3b8")
        ax.bar(x + .18, values[:, 1], .36, label="r = 24", color="#0f766e")
        ax.set_xticks(x, ["PCA iterativa", "SVD esplicita"])
        ax.set_title(split)
        ax.set_ylabel("RMSE sulle celle nascoste")
        ax.set_ylim(0, .052)
        for xpos, value in zip((x - .18, x + .18), (values[:, 0], values[:, 1])):
            for xj, yj in zip(xpos, value):
                ax.text(xj, yj + .0008, f"{yj:.4f}", ha="center", fontsize=8)
    axes[0].legend(loc="upper right")
    fig.suptitle("Effetto del rango: 6 componenti contro 24")
    fig.tight_layout()
    fig.savefig(ASSETS / "rank6_vs_rank24.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    baseline = pd.DataFrame(
        {
            "feature": [names[j] for j in targets],
            "training_mean_complete": means[targets],
            "training_mean_observed_only": masked_means[targets],
            "training_observed": m_train[:, targets].sum(axis=0),
            "training_hidden": (~m_train[:, targets]).sum(axis=0),
            "test_observed": m_test[:, targets].sum(axis=0),
            "test_hidden": (~m_test[:, targets]).sum(axis=0),
        }
    )
    baseline.to_csv(ASSETS / "baseline_and_masks.csv", index=False)

    example_rows = np.flatnonzero((~m_test[:, targets]).any(axis=1))[:8]
    example = []
    for row in example_rows:
        for j in targets:
            example.append(
                {
                    "test_row": int(row),
                    "feature": names[j],
                    "observed": int(m_test[row, j]),
                    "true_value": x_test[row, j],
                    "visible_value": x_test[row, j] if m_test[row, j] else np.nan,
                    "mean_completion": namespace["X_test_mean"][row, j],
                    "linear_masked_completion": namespace["X_test_linear_masked"][row, j],
                    "ridge_masked_completion": namespace["X_test_ridge_masked"][row, j],
                    "pca_r24_completion": namespace["X_test_lowrank"][row, j],
                    "svd_r24_completion": namespace["X_test_svd"][row, j],
                }
            )
    pd.DataFrame(example).to_csv(ASSETS / "completion_examples.csv", index=False)

    # A compact matrix excerpt, with eight test objects and six features.
    excerpt_features = [0, 1, 6, 7, 20, 21]
    pd.DataFrame(x_test[:8, excerpt_features], columns=[names[j] for j in excerpt_features]).to_csv(
        ASSETS / "matrix_excerpt.csv", index_label="test_row"
    )
    pd.DataFrame(m_test[:8, excerpt_features].astype(int), columns=[names[j] for j in excerpt_features]).to_csv(
        ASSETS / "mask_excerpt.csv", index_label="test_row"
    )

    summary = {
        "n_train": len(x_train),
        "n_train_complete_case": namespace["masked_linear_models"]["n_training_rows"],
        "linear_predictors": len(namespace["linear_predictors"]),
        "linear_coefficient_shape": list(namespace["linear_models"]["W"].shape),
        "n_validation": len(namespace["X_val"]),
        "n_test": len(x_test),
        "rank_pca": namespace["rank"],
        "rank_svd": namespace["svd_rank"],
        "variance_r6": float(namespace["cumulative"][5]),
        "variance_r24": float(namespace["cumulative"][23]),
        "max_common_prediction_gap": float(
            namespace["common_completion_results"]["max differenza celle nascoste"].max()
        ),
        "max_common_projector_gap": float(
            namespace["common_completion_results"]["max differenza proiettori"].max()
        ),
        "mask_counts": {
            split: int((~namespace[f"M_{split}"]).sum())
            for split in ("train", "val", "test")
        },
    }
    (ASSETS / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

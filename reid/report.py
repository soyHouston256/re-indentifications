"""Export evaluation metrics and retrieval examples."""

import argparse
import json
from pathlib import Path
import h5py
from .data import image_from_array
from .paths import DATA_DIR, RESULTS_DIR


def build(data, root, variants, split="testing"):
    root = Path(root)
    rows = []
    cases = []
    for variant in variants:
        folder = root / variant
        metrics = json.loads((folder / f"{split}_metrics.json").read_text())
        examples = json.loads((folder / f"{split}_examples.json").read_text())
        for condition in ["original", "low_resolution"]:
            values = metrics[condition]
            rows.append(
                {
                    "model": variant,
                    "condition": condition,
                    **values,
                    "mAP_drop_pp": metrics["mAP_drop_pp"],
                }
            )
        if variant == variants[0]:
            for correct in [True, False]:
                selected = [
                    e for e in examples["original"] if e["correct_top1"] == correct
                ][:5]
                if len(selected) < 5:
                    raise RuntimeError("Need at least five successes and five failures")
                cases.extend(
                    [dict(e, kind="acierto" if correct else "fallo") for e in selected]
                )
    assets = root / "report_assets"
    assets.mkdir(exist_ok=True)
    with h5py.File(Path(data) / (split + ".h5"), "r") as f:
        for number, case in enumerate(cases):
            q = case["query_index"]
            case["query_image"] = str(
                (assets / f"case_{number:02d}_query.png").resolve()
            )
            image_from_array(f["query/images"][q]).save(case["query_image"])
            case["query_camera"] = int(f["query/camera_ids"][q])
            case["gallery_images"] = []
            case["matches"] = []
            case["gallery_cameras"] = []
            for rank, g in enumerate(case["top_gallery_indices"][:3]):
                path = assets / f"case_{number:02d}_rank_{rank+1}.png"
                image_from_array(f["gallery/images"][g]).save(path)
                case["gallery_images"].append(str(path.resolve()))
                case["matches"].append(
                    int(f["gallery/person_ids"][g]) == case["person_id"]
                )
                case["gallery_cameras"].append(int(f["gallery/camera_ids"][g]))
    (root / "comparison.json").write_text(json.dumps(rows, indent=2))
    (root / "cases.json").write_text(json.dumps(cases, indent=2))
    text = [
        "# Resultados medidos",
        "",
        f"Partición: {split}. Métricas en porcentajes. Caída en puntos porcentuales.",
        "",
        "| Modelo | Consulta | mAP | R1 | R5 | R10 | Caída mAP |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        text.append(
            "| "
            + r["model"]
            + " | "
            + r["condition"]
            + " | "
            + " | ".join(
                f"{100*r[k]:.2f}" for k in ["mAP", "Rank-1", "Rank-5", "Rank-10"]
            )
            + f" | {r['mAP_drop_pp']:.2f} |"
        )
    text += [
        "",
        "## Ejemplos",
        "Selección fija: primeras cinco consultas correctas y primeras cinco incorrectas en orden del dataset. No representa una tasa de aciertos.",
    ]
    for n, c in enumerate(cases):
        text += [
            "",
            f"### {c['kind'].capitalize()} {n+1}: consulta {c['query_index']}",
            f"PID de evaluación {c['person_id']}, cámara {c['query_camera']}. AP={100*c['ap']:.2f}%.",
            f"![Consulta]({c['query_image']})",
        ]
        for rank, (path, match) in enumerate(zip(c["gallery_images"], c["matches"])):
            text.append(
                f"Rank {rank+1}, {'misma persona' if match else 'otra persona'}: ![Recuperación]({path})"
            )
    (root / "RESULTADOS.md").write_text("\n".join(text))
    return rows, cases


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DATA_DIR))
    p.add_argument("--root", default=str(RESULTS_DIR))
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--split", choices=["validation", "testing"], default="testing")
    a = p.parse_args()
    build(a.data, a.root, a.variants, a.split)

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def short_name(name: str) -> str:
    if name.startswith("NAVI"):
        return "NAVI"
    if name.startswith("wbcp") or name.startswith("WBCP"):
        return "WBCP"
    return name.split()[0]


def qpu_ids(item: dict[str, Any]) -> list[str]:
    return sorted(item["flushes"]["qpu_summary"].keys(), key=lambda x: int(x))


def metric_by_qpu(item: dict[str, Any], metric: str) -> list[float]:
    return [float(item["flushes"]["qpu_summary"][qid][metric]) for qid in qpu_ids(item)]


def distance_hist(item: dict[str, Any]) -> dict[int, int]:
    hist = item["flushes"].get("input_pair_distance_hist", {})
    return {int(k): int(v) for k, v in hist.items()}


def save_qpu_load(compilers: list[dict[str, Any]], out_dir: Path) -> None:
    metrics = [
        ("input_twoq_count", "Pre-routing 2q gates"),
        ("input_size", "Pre-routing local gates"),
        ("added_gates", "Added gates after routing"),
    ]
    qids = qpu_ids(compilers[0])
    x = np.arange(len(qids))
    width = 0.34

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for ax, (metric, title) in zip(axes, metrics):
        for idx, item in enumerate(compilers):
            offset = (idx - (len(compilers) - 1) / 2) * width
            values = metric_by_qpu(item, metric)
            ax.bar(x + offset, values, width, label=short_name(item["compiler"]))
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f"QPU{qid}" for qid in qids])
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Mapped Local Workload Per QPU")
    fig.savefig(out_dir / "qpu_load.png", dpi=180)
    plt.close(fig)


def save_distance_hist(compilers: list[dict[str, Any]], out_dir: Path) -> None:
    max_dist = max(max(distance_hist(item).keys()) for item in compilers)
    xs = np.arange(1, max_dist + 1)
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True)
    for idx, item in enumerate(compilers):
        hist = distance_hist(item)
        counts = np.array([hist.get(int(x), 0) for x in xs], dtype=float)
        offset = (idx - (len(compilers) - 1) / 2) * width
        axes[0].bar(xs + offset, counts, width, label=short_name(item["compiler"]))
        denom = counts.sum() if counts.sum() else 1.0
        axes[1].bar(xs + offset, counts / denom, width, label=short_name(item["compiler"]))

    axes[0].set_title("2q Pair Distance Counts")
    axes[0].set_xlabel("Backend shortest-path distance")
    axes[0].set_ylabel("Count")
    axes[1].set_title("Normalized Distance Distribution")
    axes[1].set_xlabel("Backend shortest-path distance")
    axes[1].set_ylabel("Fraction")
    for ax in axes:
        ax.set_xticks(xs)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(out_dir / "distance_hist.png", dpi=180)
    plt.close(fig)


def save_block_diagram(compilers: list[dict[str, Any]], out_dir: Path) -> None:
    rows = []
    for item in compilers:
        for qid in qpu_ids(item):
            data = item["flushes"]["qpu_summary"][qid]
            rows.append(
                {
                    "compiler": short_name(item["compiler"]),
                    "qpu": f"QPU{qid}",
                    "input_size": float(data["input_size"]),
                    "input_twoq": int(data["input_twoq_count"]),
                    "added": int(data["added_gates"]),
                    "ratio": float(data["ratio"]),
                }
            )

    max_size = max(row["input_size"] for row in rows)
    max_added = max(row["added"] for row in rows)
    fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    y = np.arange(len(rows))
    for idx, row in enumerate(rows):
        color = plt.cm.YlOrRd(0.25 + 0.7 * row["added"] / max(1, max_added))
        ax.barh(idx, row["input_size"], color=color, edgecolor="#444", height=0.72)
        label = (
            f"{row['compiler']} {row['qpu']}  "
            f"in={int(row['input_size'])}, 2q={row['input_twoq']}, "
            f"added={row['added']}, x{row['ratio']:.2f}"
        )
        text_x = min(row["input_size"] + max_size * 0.015, max_size * 0.78)
        ax.text(text_x, idx, label, va="center", fontsize=8.5)

    ax.set_yticks(y)
    ax.set_yticklabels(["" for _ in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Pre-routing local circuit size sent to Qiskit")
    ax.set_title("Mapped Local Circuit Blocks")
    ax.set_xlim(0, max_size * 1.35)
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(out_dir / "mapped_circuit_blocks.png", dpi=180)
    plt.close(fig)


def save_text_summary(compilers: list[dict[str, Any]], out_dir: Path) -> None:
    lines = []
    for item in compilers:
        name = short_name(item["compiler"])
        qpu_twoq = metric_by_qpu(item, "input_twoq_count")
        qpu_added = metric_by_qpu(item, "added_gates")
        total_twoq = sum(qpu_twoq)
        max_twoq = max(qpu_twoq)
        max_qpu = qpu_ids(item)[qpu_twoq.index(max_twoq)]
        hist = distance_hist(item)
        weighted = sum(dist * count for dist, count in hist.items())
        avg_dist = weighted / max(1, sum(hist.values()))
        lines.append(
            f"{name}: max QPU{max_qpu} twoq={int(max_twoq)}/{int(total_twoq)} "
            f"({max_twoq / max(1, total_twoq):.1%}), "
            f"added_total={int(sum(qpu_added))}, avg_distance={avg_dist:.3f}"
        )
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", nargs="?", default="outputs/local_gate_shapes_qft200.json")
    parser.add_argument("--out-dir", default="outputs/local_gate_shapes_qft200_viz")
    args = parser.parse_args()

    data = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    compilers = data["compilers"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_qpu_load(compilers, out_dir)
    save_distance_hist(compilers, out_dir)
    save_block_diagram(compilers, out_dir)
    save_text_summary(compilers, out_dir)
    print(f"Wrote visualizations to {out_dir}")


if __name__ == "__main__":
    main()

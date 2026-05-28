import argparse
import json
import re
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit
from qiskit.circuit import Gate


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def make_gate(name: str, arity: int) -> Gate:
    return Gate(name=name, num_qubits=arity, params=[])


def circuit_from_ops(ops: list[dict[str, Any]], width: int | None = None) -> QuantumCircuit:
    if width is None:
        max_q = max((max(op["qids"]) for op in ops if op.get("qids")), default=-1)
        width = max_q + 1
    qc = QuantumCircuit(width)
    for op in ops:
        name = str(op["gate"])
        qids = [int(q) for q in op["qids"]]
        if not qids:
            continue
        qc.append(make_gate(name, len(qids)), qids)
    return qc


def draw_circuit(qc: QuantumCircuit, base: Path, fold: int) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".txt").write_text(str(qc.draw(output="text", fold=fold)), encoding="utf-8")
    try:
        fig = qc.draw(output="mpl", fold=fold, idle_wires=False)
        fig.savefig(base.with_suffix(".png"), dpi=180, bbox_inches="tight")
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:
            pass
    except Exception as exc:
        base.with_suffix(".png.error.txt").write_text(str(exc), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", nargs="?", default="outputs/local_gate_shapes_qft200.json")
    parser.add_argument("--out-dir", default="outputs/local_gate_shapes_qft200_circuit_slices")
    parser.add_argument("--top-flushes", type=int, default=1)
    parser.add_argument("--ops", type=int, default=40)
    parser.add_argument("--fold", type=int, default=100)
    args = parser.parse_args()

    data = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    manifest = []

    for compiler in data["compilers"]:
        compiler_name = compiler["compiler"]
        short = "NAVI" if compiler_name.startswith("NAVI") else "WBCP" if "wbcp" in compiler_name.lower() else safe_name(compiler_name)
        flushes = compiler["flushes"]["top_flushes_by_added"][: args.top_flushes]
        for idx, flush in enumerate(flushes, start=1):
            ops = flush.get("first_ops", [])[: args.ops]
            qc = circuit_from_ops(ops, width=int(flush.get("input_width", 0)) or None)
            base = out_dir / f"{short}_top{idx}_qpu{flush['qpu_id']}_first{len(ops)}ops"
            draw_circuit(qc, base, args.fold)
            manifest.append(
                {
                    "compiler": compiler_name,
                    "file": str(base),
                    "qpu_id": flush["qpu_id"],
                    "input_size": flush["input_size"],
                    "input_width": flush["input_width"],
                    "input_ops": flush["input_ops"],
                    "shown_ops": len(ops),
                }
            )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote circuit slices to {out_dir}")


if __name__ == "__main__":
    main()

"""Storage dtype of every declared abliteration, read from the safetensors headers.

The rounding error a release carries depends on the precision it is stored in, so
plot_paper.py needs it to compute the residual that rounding alone would give.
Reads only the file headers (no weights). Writes results/storage_dtype.json:
{repo_id: "BF16" | "F16" | "F32" | ...}, the dtype holding most parameters.
"""
import json

from huggingface_hub import get_safetensors_metadata

from labels import load_rows, RESULTS


def main():
    ids = sorted(r["id"] for r in load_rows()
                 if r.get("resid_best") is not None and r["claim"] == "ablation")
    out = {}
    for rid in ids:
        counts = dict(get_safetensors_metadata(rid).parameter_count)
        out[rid] = max(counts, key=counts.get)
        print(f"{out[rid]:5} {rid}")
    (RESULTS / "storage_dtype.json").write_text(json.dumps(out, indent=1))
    print("->", RESULTS / "storage_dtype.json")


if __name__ == "__main__":
    main()

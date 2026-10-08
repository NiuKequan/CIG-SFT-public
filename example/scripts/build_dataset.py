# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Fetch the raw training mixtures, check them and build the parquet datasets that the trainer reads.

Two mixtures are supported, both fetched from public Hugging Face repositories and checked against a recorded sha256
before anything is built:

* `numina_cot` -- the math mixture (sizes 10k / 30k / 100k), hosted as jsonl on `chichi56/ASFT`. Exact duplicates are
  dropped keeping the first occurrence, then the task instruction is appended to every problem.
* `tulu_code` -- the code mixture (30k), hosted as one parquet on `allenai/tulu-3-sft-personas-code`. Its first 30000
  rows are normalized to the same shape and the problems are left untouched.

The built parquet goes to `example/data/<task>/<size>/train.parquet` and is registered in
`example/data/dataset_info.json`.

Usage:
    python example/scripts/build_dataset.py                    # every size of numina_cot
    python example/scripts/build_dataset.py --sizes 30k
    python example/scripts/build_dataset.py --task tulu_code
    python example/scripts/build_dataset.py --sizes 10k 30k --overwrite
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


EXAMPLE = Path(__file__).resolve().parent.parent
DEFAULT_DEST = EXAMPLE / "data"

#: the task instruction appended to every math problem, the recipe the paper reports
BOXED_USER_SUFFIX = "\nLet's think step by step and output the final answer within \\boxed{}."

NUMINA_REPO = "chichi56/ASFT"
#: size -> (file name, sha256, row count) of the math mixture on `NUMINA_REPO`
NUMINA_SOURCES: dict[str, tuple[str, str, int]] = {
    "10k": (
        "numina_cot_10k.jsonl",
        "a172916f9661937772d123e614cf1066b6c0586d2a871a8548988750d5a96b18",
        10000,
    ),
    "30k": (
        "numina_cot_30k.jsonl",
        "ff07c61ad51b3d3fccd7d683aa6e599a1079f81d637d6338525925a3e326024e",
        30000,
    ),
    "100k": (
        "numina_cot_100k.jsonl",
        "0c978394aa05b4254b0094926dca56159d4edb2bd409e313439d8344a25a6d06",
        100000,
    ),
}

TULU_REPO = "allenai/tulu-3-sft-personas-code"
TULU_FILE = "data/train-00000-of-00001.parquet"
TULU_SHA256 = "e343c319aa0b4c577236d1e52433528ea7801dabd5696242ba3244242f76eb57"
TULU_ROWS = 30000

#: task -> sizes, the prompt suffix and the prompt style
TASKS: dict[str, dict] = {
    "numina_cot": {"sizes": ("10k", "30k", "100k"), "suffix": BOXED_USER_SUFFIX, "prompt_style": "qwen-boxed-user"},
    "tulu_code": {"sizes": ("30k",), "suffix": "", "prompt_style": "plain"},
}


def sha256(path: Path) -> str:
    r"""Hash a file without holding it in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()


def fetch(repo_id: str, file_name: str, expected_digest: str) -> Path:
    r"""Fetch one raw file from the hub into its cache and verify it against `expected_digest`."""
    print(f"[down] {repo_id}/{file_name}")
    cached = Path(hf_hub_download(repo_id=repo_id, filename=file_name, repo_type="dataset"))
    actual_digest = sha256(cached)
    if actual_digest != expected_digest:
        raise SystemExit(
            f"{file_name} has sha256 {actual_digest}, expected {expected_digest}. "
            "The file on the hub changed, do not train on it before checking why."
        )

    return cached


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def assistant_content(messages: object) -> str:
    r"""Return the assistant turn of a chat, whatever container `messages` is wrapped in."""
    try:
        turns = list(messages)  # a list, a numpy array or a pandas cell
    except TypeError:
        return ""

    for turn in turns:
        if isinstance(turn, dict) and turn.get("role") == "assistant":
            return str(turn.get("content", "")).strip()

    return ""


def load_numina(size: str) -> tuple[list[dict], str]:
    r"""Fetch one math mixture and check its row count, returning the raw rows and the source provenance."""
    file_name, expected_digest, expected_rows = NUMINA_SOURCES[size]
    path = fetch(NUMINA_REPO, file_name, expected_digest)
    rows = read_jsonl(path)
    if len(rows) != expected_rows:
        raise SystemExit(f"{file_name} holds {len(rows)} rows, expected {expected_rows}.")

    return rows, f"{NUMINA_REPO}/{file_name}"


def load_tulu() -> tuple[list[dict], str]:
    r"""Fetch the code mixture and normalize it to the `instruction` / `response` shape."""
    path = fetch(TULU_REPO, TULU_FILE, TULU_SHA256)
    frame = pd.read_parquet(path).head(TULU_ROWS)
    rows = []
    for index, row in frame.iterrows():
        prompt = str(row.get("prompt") or "").strip()
        response = str(row.get("response") or row.get("completion") or assistant_content(row.get("messages"))).strip()
        if not prompt or not response:
            raise SystemExit(f"{TULU_REPO} row {index} is missing a prompt or a response")

        rows.append({"sample_id": str(row.get("id", index)), "instruction": prompt, "response": response})

    return rows, f"{TULU_REPO}/{TULU_FILE}"


def normalize(text: str) -> str:
    r"""Fold a problem down to the characters that identify it, the way the release recipe does."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def relative(path: Path) -> str:
    r"""Render a path against the example directory, so that the manifest never records an absolute path."""
    try:
        return str(path.resolve().relative_to(EXAMPLE))
    except ValueError:
        return str(path)


def deduplicate(rows: list[dict]) -> tuple[list[dict], list[int]]:
    r"""Drop exact duplicates while keeping the first occurrence, returning the kept rows and their source indices."""
    seen: set[str] = set()
    kept: list[dict] = []
    kept_indices: list[int] = []
    for index, row in enumerate(rows):
        key = hashlib.sha256(normalize(str(row["instruction"])).encode("utf-8")).hexdigest()
        if key in seen:
            continue

        seen.add(key)
        kept.append(row)
        kept_indices.append(index)

    return kept, kept_indices


def build_size(task: str, size: str, dest_dir: Path, overwrite: bool) -> dict:
    r"""Build one size of one task and return its manifest."""
    spec = TASKS[task]
    target = dest_dir / task / size / "train.parquet"
    if target.exists() and not overwrite:
        raise SystemExit(f"{target} exists, pass --overwrite to rebuild it")

    if task == "numina_cot":
        raw_rows, source = load_numina(size)
    else:
        raw_rows, source = load_tulu()

    rows, indices = deduplicate(raw_rows)
    frame = pd.DataFrame(
        {
            "prompt": [str(row["instruction"]) + spec["suffix"] for row in rows],
            "response": [str(row["response"]) for row in rows],
            "sample_id": [str(index) for index in indices],
            "source_index": indices,
            "prompt_style": [spec["prompt_style"]] * len(rows),
        }
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)

    manifest = {
        "task": task,
        "size": size,
        "source": source,
        "source_rows": len(raw_rows),
        "exact_duplicates": len(raw_rows) - len(rows),
        "rows": len(rows),
        "prompt_style": spec["prompt_style"],
        "parquet": relative(target),
    }
    (target.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"[ ok ] {task}/{size}: {len(raw_rows)} raw rows, {manifest['exact_duplicates']} exact duplicates, "
        f"{len(rows)} written to {target}"
    )
    return manifest


def refresh_dataset_info(dest_dir: Path, task: str, sizes: list[str]) -> Path:
    r"""Register every built size under its own name, leaving the entries of the other sizes alone."""
    path = dest_dir / "dataset_info.json"
    info = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for size in sizes:
        info[f"cig_sft_{task}_{size}"] = {
            "file_name": f"{task}/{size}/train.parquet",
            "formatting": "alpaca",
            "columns": {"prompt": "prompt", "query": None, "response": "response"},
        }

    dest_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[ ok ] registered {', '.join(f'cig_sft_{task}_{size}' for size in sorted(sizes))} in {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=sorted(TASKS), default="numina_cot", help="which mixture to build")
    parser.add_argument("--sizes", nargs="+", default=None, help="sizes to build (default: every size of the task)")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help=f"dataset directory (default: {DEFAULT_DEST})")
    parser.add_argument("--overwrite", action="store_true", help="rebuild a parquet that already exists")
    args = parser.parse_args()

    spec = TASKS[args.task]
    sizes = args.sizes or list(spec["sizes"])
    unknown = [size for size in sizes if size not in spec["sizes"]]
    if unknown:
        raise SystemExit(
            f"unknown size(s) for task {args.task}: {', '.join(unknown)} (choices: {', '.join(spec['sizes'])})"
        )

    for size in sizes:
        build_size(args.task, size, args.dest, args.overwrite)

    refresh_dataset_info(args.dest, args.task, sizes)
    print("\nNext: bash example/scripts/train_math_30k.sh   (or see example/scripts/smoke_train.sh for a quick check)")


if __name__ == "__main__":
    sys.exit(main())

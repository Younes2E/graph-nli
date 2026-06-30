"""
python3 scripts/process_dataset.py --dataset snli --split validation --out out/snli_val --device cuda:1 --gpu-mem 0.9
python3 scripts/process_dataset.py --dataset snli --split test --out out/snli_test --device cuda:1 --gpu-mem 0.9
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.serialize import read_jsonl, append_many_jsonl
from datasets import load_dataset
from tqdm import tqdm

_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--dataset", required=True, choices=["snli", "anli"])
_ap.add_argument("--split", required=True, help="snli: train/validation/test ; anli: train_r1/dev_r1/test_r1 ...")
_ap.add_argument("--out", required=True, help="répertoire de sortie")
_ap.add_argument("--device", default="cuda:0", help="cuda:0, cuda:1, ...")
_ap.add_argument("--gpu-mem", type=float, default=0.9, help="fraction du GPU pour le modèle chargé (défaut 0.9 = GPU vide ; " "baisser à ~0.5 si le GPU est partagé)")
args = _ap.parse_args()

if args.device.startswith("cuda:"):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[1]

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DATASET_REPOS = {"snli": "stanfordnlp/snli", "anli": "facebook/anli"}
BATCH_SIZE = 512
MAX_MODEL_LEN = 4096


def run_pass(builder, which, desc, items, out_path, key_of, process):
    """Une phase reprenable : charge le modèle `which`, traite `items` par batch en
    appendant dans out_path (avec tqdm), puis libère la VRAM.

    - key_of(item) -> clé pour sauter ce qui est déjà dans out_path (reprise).
    - process(batch) -> list[dict] : lignes JSONL à écrire pour ce batch.
    """
    done = {o["text"] for o in read_jsonl(out_path)}
    todo = [it for it in items if key_of(it) not in done]
    print(f"[{desc}] {len(done)} déjà faits, {len(todo)} à traiter.")
    if not todo:
        return

    builder.load(which)
    n_batches = (len(todo) + BATCH_SIZE - 1) // BATCH_SIZE
    bar = tqdm(total=len(todo), unit="phr", desc=desc, smoothing=0.1)
    for b, i in enumerate(range(0, len(todo), BATCH_SIZE), 1):
        batch = todo[i:i + BATCH_SIZE]
        append_many_jsonl(out_path, process(batch))
        bar.set_postfix_str(f"batch {b}/{n_batches}")
        bar.update(len(batch))
    bar.close()
    builder.unload()


def main():
    from utils.graph import GraphBuilder
    out = args.out
    os.makedirs(out, exist_ok=True)
    f_props = os.path.join(out, "propositions.jsonl")
    f_triples = os.path.join(out, "triples.jsonl")
    f_graphs = os.path.join(out, "graphs.jsonl")

    ds = load_dataset(DATASET_REPOS[args.dataset], split=args.split)
    ds = ds.filter(lambda x: x["label"] != -1)

    # Phrases uniques (premise ∪ hypothesis).
    sentences = sorted({s for ex in ds for s in (ex["premise"], ex["hypothesis"])})
    print(f"{len(sentences)} phrases uniques.")

    builder = GraphBuilder(gpu_mem_total=args.gpu_mem, max_model_len=MAX_MODEL_LEN)

    # PHASE 1 : Propositioneur -> propositions.jsonl (une phrase par ligne).
    run_pass(
        builder, "prop", "prop", sentences, f_props, key_of=lambda s: s,
        process=lambda batch: [
            {"text": t, "props": sorted(p)}
            for t, p in zip(batch, builder.extract_atomic_prop_batch(batch))
        ],
    )

    # PHASE 2 : Qwen -> triples.jsonl (lit propositions.jsonl, regroupe par phrase).
    prop_rows = list(read_jsonl(f_props))

    def triples_for(batch):
        flat, owners = [], []
        for i, r in enumerate(batch):
            for p in r["props"]:
                flat.append(p); owners.append(i)
        per_text = [set() for _ in batch]
        for owner, trs in zip(owners, builder.extract_triples_batch(flat) if flat else []):
            per_text[owner].update(trs)
        return [{"text": r["text"], "triples": sorted([list(t) for t in s])}
                for r, s in zip(batch, per_text)]

    run_pass(builder, "qwen", "triples", prop_rows, f_triples,
             key_of=lambda r: r["text"], process=triples_for)

    # PHASE 3 : reconstruction des paires (premise, hypothesis, label) + triplets (CPU).
    cache = {o["text"]: o["triples"] for o in read_jsonl(f_triples)}
    rows = [{
        "premise": ex["premise"],
        "hypothesis": ex["hypothesis"],
        "label": ex["label"],
        "premise_triples": cache[ex["premise"]],
        "hypothesis_triples": cache[ex["hypothesis"]],
    } for ex in ds if ex["premise"] in cache and ex["hypothesis"] in cache]
    if os.path.exists(f_graphs):
        os.remove(f_graphs)
    append_many_jsonl(f_graphs, rows)
    print(f"{len(rows)} exemples -> {f_graphs}")


if __name__ == "__main__":
    main()

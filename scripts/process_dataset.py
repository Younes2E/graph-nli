"""Transforme un dataset NLI (SNLI/ANLI) en graphes de triplets.

Usage :
    python scripts/process_dataset.py --dataset snli --split test --out out/snli_test --device cuda:1

Écrit dans --out : triples.jsonl (cache phrase->triplets, reprenable) et graphs.jsonl
(sortie finale = {premise, hypothesis, label, premise_triples, hypothesis_triples}).
Relancer la commande saute les phrases déjà traitées.
"""
import argparse
import os
import sys

# --device est traduit en CUDA_VISIBLE_DEVICES AVANT tout import de vllm/torch.
_ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--dataset", required=True, choices=["snli", "anli"])
_ap.add_argument("--split", required=True,
                 help="snli: train/validation/test ; anli: train_r1/dev_r1/test_r1 ...")
_ap.add_argument("--out", required=True, help="répertoire de sortie")
_ap.add_argument("--device", default="cuda:0", help="cuda:0, cuda:1, ...")
args = _ap.parse_args()

if args.device.startswith("cuda:"):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[1]

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets import load_dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from utils.serialize import read_jsonl, append_many_jsonl  # noqa: E402

DATASET_REPOS = {"snli": "stanfordnlp/snli", "anli": "facebook/anli"}
BATCH_SIZE = 512
GPU_MEM_TOTAL = 0.7  # fraction TOTALE du GPU pour les deux 7B (réparti en deux en interne)


def main():
    out = args.out
    os.makedirs(out, exist_ok=True)
    f_triples = os.path.join(out, "triples.jsonl")
    f_graphs = os.path.join(out, "graphs.jsonl")

    ds = load_dataset(DATASET_REPOS[args.dataset], split=args.split)
    ds = ds.filter(lambda x: x["label"] != -1)

    # 1. Phrases uniques (premise ∪ hypothesis).
    sentences = sorted({s for ex in ds for s in (ex["premise"], ex["hypothesis"])})
    print(f"{len(sentences)} phrases uniques.")

    # 2. Triplets par phrase (reprise : on saute les phrases déjà faites).
    done = {o["text"] for o in read_jsonl(f_triples)}
    todo = [s for s in sentences if s not in done]
    print(f"{len(done)} déjà faites, {len(todo)} à traiter.")
    if todo:
        from utils.graph import GraphBuilder
        builder = GraphBuilder(gpu_mem_total=GPU_MEM_TOTAL)
        for i in tqdm(range(0, len(todo), BATCH_SIZE), desc="graph"):
            batch = todo[i:i + BATCH_SIZE]
            graphs = builder.build_batch(batch)
            append_many_jsonl(f_triples, [
                {"text": kg.get_text(),
                 "triples": sorted([list(t) for t in kg.get_rel(sentence=True)])}
                for kg in graphs
            ])

    # 3. Reconstruction des paires (premise, hypothesis, label) + triplets.
    cache = {o["text"]: o["triples"] for o in read_jsonl(f_triples)}
    rows = [{
        "premise": ex["premise"],
        "hypothesis": ex["hypothesis"],
        "label": ex["label"],
        "premise_triples": cache[ex["premise"]],
        "hypothesis_triples": cache[ex["hypothesis"]],
    } for ex in ds]
    if os.path.exists(f_graphs):
        os.remove(f_graphs)
    append_many_jsonl(f_graphs, rows)
    print(f"{len(rows)} exemples -> {f_graphs}")


if __name__ == "__main__":
    main()

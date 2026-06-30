from .graph import KnowledgeGraph
import json

def kg_to_dict(kg):
    return {
        "text": kg.get_text(),
        "triples": sorted([list(t) for t in kg.get_rel(sentence=True)]),
    }


def kg_from_dict(d):
    triples = {tuple(t) for t in d.get("triples", [])}
    return KnowledgeGraph(d["text"], triples)


def read_jsonl(path):
    try:
        f = open(path, encoding="utf-8")
    except FileNotFoundError:
        return
    with f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path, obj):
    """Ajoute un objet (une ligne) à un fichier JSONL, avec flush immédiat."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def append_many_jsonl(path, objs):
    """Ajoute plusieurs objets d'un coup (un flush à la fin du chunk)."""
    with open(path, "a", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()

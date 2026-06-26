from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import networkx as nx
import json
import os

# On décode en greedy (temperature=0) : le sampler FlashInfer de vLLM est inutile, et
# sa compilation JIT échoue dans cet environnement. Doit être positionné AVANT vllm.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_HERE, os.pardir, "data")

PROP_MODEL_NAME = "Zual/MPropositioneur-V2-large"
TRIPLES_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


class GraphBuilder:
    """Construit des KnowledgeGraph à partir de texte via deux LLM servis par vLLM :
    le Propositioneur (texte -> propositions atomiques) puis Qwen (proposition ->
    triplets closedIE). Les deux modèles sont chargés au démarrage.

    Méthodes principales : `build`/`build_batch` (texte -> KnowledgeGraph),
    et les briques `extract_atomic_prop`/`extract_triples` (versions batch en _batch).
    """

    def __init__(self, gpu_mem_total=None, max_model_len=8192,
                 enforce_eager=True, kv_cache_gb=2.0, **llm_kwargs):
        # Deux régimes selon gpu_mem_total :
        # - None (défaut, mode LÉGER) : chaque modèle ne réserve que ses poids + ~kv_cache_gb
        #   de KV cache. On calcule gpu_memory_utilization juste pour ce besoin (poids estimés
        #   + cache) / mémoire totale du GPU. Démarre même si le GPU est partiellement occupé.
        # - float (mode BATCH, GPU vide) : fraction TOTALE du GPU pour les DEUX modèles,
        #   répartie en deux ; gros KV cache -> débit max. Ex: gpu_mem_total=0.9.
        # enforce_eager=True : la compilation JIT de vLLM échoue dans cet environnement.
        from vllm import LLM

        common = dict(dtype="float16", max_model_len=max_model_len,
                      enforce_eager=enforce_eager, **llm_kwargs)

        def util_for(weights_gb):
            # gpu_memory_utilization est un ratio sur la mémoire TOTALE du GPU ; vLLM exige
            # aussi que le free memory au démarrage le couvre. En mode léger on demande juste
            # poids + cache. En mode batch on répartit gpu_mem_total en deux.
            if gpu_mem_total is not None:
                return gpu_mem_total / 2
            import torch
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return min(0.95, (weights_gb + kv_cache_gb) / total_gb)

        # Poids estimés en VRAM : Propositioneur ~8 GB (bf16, ~4B), Qwen2.5-7B ~14.3 GB.
        self.prop_tokenizer = AutoTokenizer.from_pretrained(PROP_MODEL_NAME)
        self.prop_llm = LLM(model=PROP_MODEL_NAME,
                            gpu_memory_utilization=util_for(8.5), **common)

        self.qwen_tokenizer = AutoTokenizer.from_pretrained(TRIPLES_MODEL_NAME)
        self.qwen_llm = LLM(model=TRIPLES_MODEL_NAME,
                            gpu_memory_utilization=util_for(15.0), **common)

        with open(os.path.join(_DATA_DIR, "relations.json"), encoding="utf-8") as file:
            relations_raw = json.load(file)
            self.relations = [
                {k: v for k, v in item.items() if k in {"relation", "description"}}
                for item in relations_raw["definitions"] if item["relation"] in relations_raw["ie"]
            ]

        with open(os.path.join(_DATA_DIR, "exemples.json"), encoding="utf-8") as file:
            exemples_raw = json.load(file)
            lines = []
            for line in exemples_raw:
                lines.append(f'Input: "{line["input"]}"')
                lines.append("Output:")
                for triple in line["output"]:
                    lines.append(" | ".join(triple))
                lines.append("\n")
            self.exemples = "\n".join(lines)

    def extract_atomic_prop_batch(self, texts):
        from vllm import SamplingParams
        prompts = [
            self.prop_tokenizer.apply_chat_template(
                [{"role": "user", "content": f"Atomize: {t}"}],
                tokenize=False, add_generation_prompt=True)
            for t in texts
        ]
        outputs = self.prop_llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=2048))

        results = []
        for o in outputs:
            try:
                results.append(set(json.loads(o.outputs[0].text.strip())))
            except Exception:
                print(f"\nSortie Propositioneur non-parsable : {o.outputs[0].text!r}\n")
                results.append(set())
        return results

    def extract_triples_batch(self, props):
        """list[str] -> list[list[tuple]] : triplets (s, rel, o) par proposition."""
        from vllm import SamplingParams
        prompts = [self._triples_prompt(p) for p in props]
        outputs = self.qwen_llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=512))
        return [self._parse_triples(o.outputs[0].text) for o in outputs]

    def _triples_prompt(self, text):
        prompt = f"""
        Extract all factual ([SUBJECT], [RELATION], [OBJECT]) triples from sentence.
        One triple per line in the format:
        [SUBJECT] | [RELATION] | [OBJECT]

        No explanations. If no triple can be extracted, write nothing.

        ALLOWED RELATIONS:
        {json.dumps(self.relations, indent=1)}
        Do NOT use any relation outside of this list.

        ADDITIONAL RULES:
        1. If an action is negated in the sentence (e.g., "didn't call", "is not eating"), you MUST capture the negation inside the [VERB] node using 'not' (e.g., 'not_call', 'not_eating').
        2. When an action involves multiple elements at once (e.g., an actor, a target, a recipient, a tool, or a location), do NOT link the secondary elements to each other. Instead, make the action the central hub and ALL links must involve the action.

        EXEMPLES:
        {self.exemples}


        Sentence: {text}"""
        messages = [
            {"role": "system", "content": "You are a deterministic Information Extraction expert."},
            {"role": "user", "content": prompt},
        ]
        return self.qwen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    @staticmethod
    def _parse_triples(content):
        triples = []
        for line in content.split("\n"):
            t = line.split(" | ")
            try:
                triples.append((t[0].strip().lower().replace(" ", "_"),
                                t[1].strip(),
                                t[2].strip().lower().replace(" ", "_")))
            except Exception:
                if line.strip():
                    print(f"\nMauvais format de triplet : {t}.\n")
        return triples

    def build_batch(self, texts):
        props_per_text = self.extract_atomic_prop_batch(texts)

        flat_props, owners = [], []
        for i, props in enumerate(props_per_text):
            for p in props:
                flat_props.append(p)
                owners.append(i)

        triples_per_prop = self.extract_triples_batch(flat_props)

        per_text = [set() for _ in texts]
        for owner, triples in zip(owners, triples_per_prop):
            per_text[owner].update(triples)
        return [KnowledgeGraph(t, tr) for t, tr in zip(texts, per_text)]

    def build(self, text):
        """str -> KnowledgeGraph."""
        return self.build_batch([text])[0]


class KnowledgeGraph:
    def __init__(self, text, triples):
        self.text = text
        self.triples_sentence = set(triples)
        self.triples_augmented = set()
        self.entities = {t[i]: True for t in triples for i in (0, 2)}

    def augment(self, triples):
        self.triples_augmented.update([(t[0],t[1],t[2]) for t in triples])
        for t in triples:
            self.entities.setdefault(t[0], False)
            self.entities.setdefault(t[2], False)

    def get_entities(self, sentence = None):
        # sentence : si ce sont des nodes qui viennent du text original ou augmented
        if sentence is None:
            return set(self.entities.keys())
        elif isinstance(sentence, bool):
            return set([ent for ent, v in self.entities.items() if v is sentence])
        else:
            raise ValueError("'sentence' must be True, False or None.")

    def get_rel(self, sentence = None):
        if sentence is None:
            return self.triples_sentence | self.triples_augmented
        elif isinstance(sentence, bool):
            return self.triples_sentence if sentence else self.triples_augmented
        else:
            raise ValueError("'sentence' must be True, False or None.")

    def get_text(self):
        return self.text

    def display(self, file=None):
        G = nx.MultiDiGraph()

        for subj, rel, obj in self.triples_sentence:
            G.add_edge(subj, obj, relation=rel, origin='base')

        for subj, rel, obj in self.triples_augmented:
            exists_in_base = False
            if G.has_edge(subj, obj):
                for key, data in G[subj][obj].items():
                    if data['relation'] == rel and data['origin'] == 'base':
                        G[subj][obj][key]['origin'] = 'both'
                        exists_in_base = True
                        break

            if not exists_in_base:
                G.add_edge(subj, obj, relation=rel, origin='augmented')

        plt.figure(figsize=(10, 7))
        pos = nx.spring_layout(G, k=2.0, seed=42)

        node_colors = []
        for node in G.nodes():
            if self.entities.get(node) is True:
                node_colors.append("skyblue")
            else:
                node_colors.append("lightcoral")

        nx.draw_networkx_nodes(G, pos, node_size=2500, node_color=node_colors, alpha=0.9)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", font_family="sans-serif")

        pair_counts = {}
        for u, v, k, d in G.edges(keys=True, data=True):
            pair = tuple(sorted([u, v]))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        pair_seen = {}

        RAD_STEP = 0.4

        for u, v, k, d in G.edges(keys=True, data=True):
            pair = tuple(sorted([u, v]))
            idx = pair_seen.get(pair, 0)
            pair_seen[pair] = idx + 1
            total = pair_counts[pair]

            if total == 1:
                rad = 0.0
            else:
                rad = (idx - (total - 1) / 2) * RAD_STEP
                if u > v:
                    rad = -rad

            if d['origin'] == 'base':
                color = 'black'
                style = 'solid'
            elif d['origin'] == 'augmented':
                color = 'red'
                style = 'solid'
            else:
                color = 'purple'
                style = 'dashed'

            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                arrowstyle="->", arrowsize=20,
                edge_color=color, width=1.5, style=style,
                min_source_margin=30, min_target_margin=30,
                connectionstyle=f"arc3,rad={rad}"
            )

            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels={(u, v): d['relation']},
                font_size=8, font_color=color,
                label_pos=0.5,
                connectionstyle=f"arc3,rad={rad}"
            )

        title_text = self.text if hasattr(self, 'text') else "Knowledge Graph"
        plt.title(title_text, fontsize=10, fontweight="bold")
        plt.axis("off")
        plt.tight_layout()

        if file:
            plt.savefig(file, format="png", dpi=300)
            plt.close()
            print(f"Graphe sauvegardé avec succès dans : {os.path.abspath(file)}")
        else:
            plt.show()
            plt.close()

# Semantic Centrality demo (Lee & Chen, 2022 style)
# -------------------------------------------------
# pip install tensorflow tensorflow_hub            # (Option A: USE)
# pip install networkx scikit-learn pandas numpy matplotlib

from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx
import os
import json

# ------------------ Hardcoded parameters ------------------ #
os.chdir('/Users/UChicago/CASNL/storyfest/storyfest/scripts/preprocessing')
_THISDIR = os.getcwd()
SAVE_PATH = os.path.normpath(os.path.join(_THISDIR, '../../data/pupil/3_processed/17_semantic_centrality'))
EVENTS_PATH = os.path.normpath(os.path.join(_THISDIR, '../../experiment/Storyfest_Event_Segmentation.xlsx'))
COARSE_EVENTS_PATH = os.path.normpath(os.path.join(_THISDIR, '../../experiment/eventsegmentation_coarse.xlsx'))

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

STORIES = ['Pool Party', 'Sea Ice', 'Natalie Wood', 'Grandfather Clocks', 'Impatient Billionaire', 'Dont Look']

STORY_VALENCE = {
    'Pool Party': 'positive',
    'Sea Ice': 'neutral',
    'Natalie Wood': 'negative',
    'Impatient Billionaire': 'positive',
    'Grandfather Clocks': 'neutral',
    'Dont Look': 'negative'
}

# ---- Embedding backends ------------------------------------------------------

def get_use_encoder():
    """Try to load Universal Sentence Encoder from TF-Hub."""
    import tensorflow_hub as hub
    model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    return hub.load(model_url)

class TextEmbedder:
    """
    Wrapper that prefers USE (as in Lee & Chen), but will
    gracefully fall back to a sentence-transformers model if USE isn't available.
    """
    def __init__(self):
        self.backend = None
        self.mode = None
        try:
            self.backend = get_use_encoder()
            self.mode = "use"
        except Exception:
            pass

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.mode == "use":
            # USE returns tf.Tensor
            import tensorflow as tf
            emb = self.backend(texts)
            return emb.numpy() if isinstance(emb, tf.Tensor) else np.array(emb)
        elif self.mode == "st":
            return self.backend.encode(texts, normalize_embeddings=False)
        else:
            raise RuntimeError("No embedding backend available.")

# ---- Core computations --------------------------------------------------------

def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """Row-normalized cosine similarity matrix for embeddings X."""
    # (n_events x d) -> (n_events x n_events)
    # robust safe cosine
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.clip(X_norm, 1e-12, None)
    S = Xn @ Xn.T
    # numerical quirks: clip into [-1, 1]
    np.clip(S, -1.0, 1.0, out=S)
    # zero diagonal (we don't connect an event to itself)
    np.fill_diagonal(S, 0.0)
    return S

def build_graph_from_similarity(S: np.ndarray,
                                event_ids: list[str] | None = None,
                                prune_below: float | None = None) -> nx.Graph:
    """
    Build a weighted undirected graph from similarity matrix S.
    Optionally prune weak edges (e.g., prune_below=0.1 keeps edges ≥ 0.1).
    """
    n = S.shape[0]
    if event_ids is None:
        event_ids = [f"event_{i+1}" for i in range(n)]

    G = nx.Graph()
    G.add_nodes_from(event_ids)

    for i in range(n):
        for j in range(i + 1, n):
            w = float(S[i, j])
            if prune_below is not None and w < prune_below:
                continue
            if w != 0.0:
                G.add_edge(event_ids[i], event_ids[j], weight=w)
    return G

def weighted_degree_centrality(G: nx.Graph) -> pd.Series:
    """
    Semantic centrality = weighted node degree
    (sum of incident edge weights).
    """
    deg = {n: sum(d.get("weight", 1.0) for _, _, d in G.edges(n, data=True))
           for n in G.nodes()}
    return pd.Series(deg, name="semantic_centrality").sort_values(ascending=False)

def zscore(x: pd.Series) -> pd.Series:
    return (x - x.mean()) / (x.std(ddof=1) + 1e-12)

# ---- Saving helpers -----------------------------------------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)

def save_edge_list(G: nx.Graph, path: str):
    rows = []
    for u, v, d in G.edges(data=True):
        rows.append({"source": u, "target": v, "weight": float(d.get("weight", 1.0))})
    pd.DataFrame(rows).to_csv(path, index=False)

def save_numpy(arr: np.ndarray, path: str):
    np.save(path, arr)

def save_graph(G, path: str):
    """
    Save the NetworkX graph. Prefer gpickle submodule; fall back to gzip+pickle.
    """
    try:
        # NetworkX 3.x: use the submodule explicitly
        from networkx.readwrite.gpickle import write_gpickle
        write_gpickle(G, path)  # e.g., ".../graph.gpickle"
    except Exception:
        # Fallback: compressed pickle (adds .pkl.gz)
        import gzip, pickle, os
        alt_path = path if path.endswith(".pkl.gz") else f"{path}.pkl.gz"
        with gzip.open(alt_path, "wb") as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def plot_and_save_network(G: nx.Graph, out_png: str, title: str):
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(G, seed=7)
    weights = [d["weight"] for *_, d in G.edges(data=True)]
    plt.figure()
    nx.draw_networkx_nodes(G, pos, node_size=800)
    nx.draw_networkx_labels(G, pos, font_size=10)
    if weights:
        nx.draw_networkx_edges(G, pos, width=[3*w for w in weights])
    else:
        nx.draw_networkx_edges(G, pos)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_and_save_centrality_bar(df_cent: pd.DataFrame, out_png: str):
    import matplotlib.pyplot as plt
    plt.figure()
    ax = df_cent.sort_values("semantic_centrality", ascending=False)
    ax = ax.set_index("event_id")["semantic_centrality"].plot(kind="bar")
    plt.ylabel("Weighted degree (semantic centrality)")
    plt.xlabel("Event")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_and_save_similarity_heatmap(S: np.ndarray, event_ids: list[str], out_png: str):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(S, interpolation="nearest", aspect="auto")
    plt.colorbar(label="Cosine similarity")
    plt.xticks(ticks=np.arange(len(event_ids)), labels=[str(e) for e in event_ids], rotation=90)
    plt.yticks(ticks=np.arange(len(event_ids)), labels=[str(e) for e in event_ids])
    plt.title("Event-by-event semantic similarity")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# ---- Convenience: run the full pipeline -------------------------------------

def compute_semantic_centrality(
        descriptions: list[str],
        event_ids: list[str] | None = None,
        prune_below: float | None = None,
        return_all: bool = False
    ):
    """
    End-to-end: embeddings -> cosine similarities -> graph -> centrality.

    Args
    - descriptions: list of event descriptions (one per event).
    - event_ids: optional string IDs (len must match); else "event_1"... used.
    - prefer_use: try Universal Sentence Encoder first.
    - prune_below: prune edges with weight < prune_below (e.g., 0.1 or 0.2).
    - return_all: if True, also returns embeddings, similarity matrix, and graph.

    Returns
    - centrality (pd.DataFrame with raw & z-scored), and optionally (E, S, G).
    """
    assert len(descriptions) >= 2, "Need at least 2 events."

    embedder = TextEmbedder()
    E = embedder.encode(descriptions)              # (n x d)
    S = cosine_similarity_matrix(E)                # (n x n)
    G = build_graph_from_similarity(S, event_ids, prune_below=prune_below)

    cent = weighted_degree_centrality(G)
    out = pd.DataFrame({
        "event_id": cent.index,
        "semantic_centrality": cent.values
    })
    out["centrality_z"] = zscore(out["semantic_centrality"])
    out = out.reset_index(drop=True)

    if return_all:
        return out, E, S, G
    return out

# ---- Example usage -----------------------------------------------------------

if __name__ == "__main__":
    xl = pd.ExcelFile(EVENTS_PATH)
    for story in STORIES:
        # story_dir = SAVE_PATH

        # Read events for this story
        sheet = xl.parse(story)
        df = sheet[["event_number", "Transcript"]].copy()
        df["event_number"] = pd.to_numeric(df["event_number"], errors="coerce")
        df["Transcript"]   = df["Transcript"].astype("string").str.strip()
        df = df.dropna(subset=["event_number", "Transcript"])
        descriptions = df["Transcript"].tolist()
        event_ids = df["event_number"].astype(int).astype(str).tolist()
        # descriptions = sheet["Transcript"].astype(str).tolist()
        # event_ids = sheet["event_number"].astype(str).tolist()

        # Compute
        centrality_df, E, S, G = compute_semantic_centrality(
            descriptions=descriptions,
            event_ids=event_ids,
            prune_below=0.10,
            return_all=True
        )

        # ---- Save all artifacts ----
        # 1) centrality table
        centrality_path = os.path.join(SAVE_PATH, "centrality")
        if not os.path.exists(centrality_path):
            os.makedirs(centrality_path)
        save_csv(centrality_df, os.path.join(centrality_path, f"{story}_centrality.csv"))

        # 2) similarity matrix (NxN)
        sim_df = pd.DataFrame(S, index=event_ids, columns=event_ids)
        matrix_path = os.path.join(SAVE_PATH, "matrix")
        if not os.path.exists(matrix_path):
            os.makedirs(matrix_path)
        save_csv(sim_df.reset_index().rename(columns={"index": "event_id"}),
                 os.path.join(matrix_path, f"{story}_similarity_matrix.csv"))

        # 3) edge list
        edge_path = os.path.join(SAVE_PATH, "edge")
        if not os.path.exists(edge_path):
            os.makedirs(edge_path)
        save_edge_list(G, os.path.join(edge_path, f"{story}_edge_list.csv"))

        # 4) embeddings
        embedding_path = os.path.join(SAVE_PATH, "embedding")
        if not os.path.exists(embedding_path):
            os.makedirs(embedding_path)
        save_numpy(E, os.path.join(embedding_path, f"{story}_embeddings.npy"))

        # 5) graph object
        graph_path = os.path.join(SAVE_PATH, "graph")
        if not os.path.exists(graph_path):
            os.makedirs(graph_path)
        save_graph(G, os.path.join(graph_path, f"{story}_graph.gpickle"))

        # 6) quick plots
        network_path = os.path.join(SAVE_PATH, "network")
        if not os.path.exists(network_path):
            os.makedirs(network_path)
        bar_path = os.path.join(SAVE_PATH, "bar")
        if not os.path.exists(bar_path):
            os.makedirs(bar_path)
        heatmap_path = os.path.join(SAVE_PATH, "heatmap")
        if not os.path.exists(heatmap_path):
            os.makedirs(heatmap_path)
        try:
            plot_and_save_network(G, os.path.join(network_path, f"{story}_network.png"),
                                  title=f"{story}: Semantic narrative network")
            plot_and_save_centrality_bar(centrality_df,
                                         os.path.join(bar_path, f"{story}_centrality_bar.png"))
            plot_and_save_similarity_heatmap(S, event_ids,
                                             os.path.join(heatmap_path, f"{story}_similarity_heatmap.png"))
        except Exception as e:
            print(f"[{story}] Plotting skipped:", e)

        # Console summary
        print(f"\n=== {story} — top events by semantic centrality ===")
        print(centrality_df.sort_values("semantic_centrality", ascending=False).head(10).to_string(index=False))
        print(f"Saved outputs → {SAVE_PATH}")

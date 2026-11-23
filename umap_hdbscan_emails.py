from typing import List, Tuple

import hdbscan
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import umap


def generate_mock_email_embeddings(
    n_per_cluster: int = 45,
    n_noise: int = 15,
    dim: int = 768,
    seed: int = 42,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Gera embeddings sintéticos de 768 dimensões para e-mails de atendimento
    divididos em 3 temas principais + e-mails atípicos/ruído.
    """
    rng = np.random.default_rng(seed)

    topics = {
        0: "Cobrança / Fatura / Boleto",
        1: "Acesso / Senha / Autenticação",
        2: "Prazo / Entrega / Status do Pedido",
    }

    embeddings_list: List[np.ndarray] = []
    labels_true: List[int] = []

    for topic_id in topics:
        # Protótipo base em 768D + ruído isotrópico de alta dimensionalidade
        center = rng.normal(0.0, 0.25, size=(1, dim))
        cluster_vecs = center + rng.normal(0.0, 0.45, size=(n_per_cluster, dim))
        embeddings_list.append(cluster_vecs)
        labels_true.extend([topic_id] * n_per_cluster)

    if n_noise > 0:
        noise_vecs = rng.normal(0.0, 0.85, size=(n_noise, dim))
        embeddings_list.append(noise_vecs)
        labels_true.extend([-1] * n_noise)

    x_emb = np.vstack(embeddings_list).astype(np.float32)
    y_true = np.asarray(labels_true, dtype=np.int32)

    perm = rng.permutation(len(x_emb))
    return x_emb[perm], [topics.get(int(lbl), "Ruído / Atípico") for lbl in y_true[perm]], y_true[perm]


def main() -> None:
    x_768d, _, y_true = generate_mock_email_embeddings(
        n_per_cluster=45, n_noise=15, dim=768, seed=42
    )

    # Mostra como distâncias euclidianas brutas em 768D tendem a ficar concentradas
    raw_dists = cosine_distances(x_768d[:50])
    off_diag = raw_dists[~np.eye(50, dtype=bool)]
    print(f"Total de e-mails avaliados: {len(x_768d)} | Dimensão original: {x_768d.shape[1]}")
    print(
        f"Distância cosseno bruta (768D) -> média: {off_diag.mean():.4f} | "
        f"desvio: {off_diag.std():.4f} | min-max: [{off_diag.min():.4f}, {off_diag.max():.4f}]\n"
    )

    # 1. Redução de dimensionalidade com UMAP (768D -> 10D)
    reducer = umap.UMAP(
        n_components=10,
        n_neighbors=15,
        min_dist=0.05,
        metric="cosine",
        random_state=42,
    )
    x_umap = reducer.fit_transform(x_768d)

    # 2. Agrupamento hierárquico baseado em densidade com HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=15,
        min_samples=5,
        metric="euclidean",
    )
    cluster_labels = clusterer.fit_predict(x_umap)

    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    print("=== Clusters encontrados (UMAP -> HDBSCAN) ===")
    for cid, cnt in zip(unique_clusters, counts):
        nome = "Ruído / Não agrupado (-1)" if cid == -1 else f"Cluster {cid}"
        print(f"  {nome:<28}: {cnt} e-mails")

    # Resumo de probabilidade de pertinência ao cluster (0.0 para ruído)
    non_noise_mask = cluster_labels != -1
    if np.any(non_noise_mask):
        avg_prob = float(clusterer.probabilities_[non_noise_mask].mean())
        print(f"\nConfiança média nos clusters válidos: {avg_prob:.4f}")


if __name__ == "__main__":
    main()
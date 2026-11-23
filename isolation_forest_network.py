import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report


def generate_network_traffic(
    n_normal: int = 950,
    n_anomalies: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gera features sintéticas de tráfego de rede:
    [bytes_transferred, packets_per_sec, duration_sec, unique_ports]
    Rótulos: 1 = normal, -1 = anomalia (padrão do scikit-learn).
    """
    rng = np.random.default_rng(seed)

    normal_traffic = np.column_stack(
        [
            rng.normal(loc=5000.0, scale=1200.0, size=n_normal),   # bytes
            rng.normal(loc=40.0, scale=10.0, size=n_normal),        # pacotes/s
            rng.normal(loc=2.5, scale=0.8, size=n_normal),          # duração (s)
            rng.integers(1, 6, size=n_normal),                      # portas únicas
        ]
    )

    # Anomalias: exfiltração massiva, varredura de portas ou rajadas atípicas
    anomalous_traffic = np.column_stack(
        [
            rng.normal(loc=45000.0, scale=8000.0, size=n_anomalies),
            rng.normal(loc=320.0, scale=60.0, size=n_anomalies),
            rng.normal(loc=35.0, scale=10.0, size=n_anomalies),
            rng.integers(25, 80, size=n_anomalies),
        ]
    )

    x = np.vstack([normal_traffic, anomalous_traffic]).astype(np.float32)
    y_true = np.array([1] * n_normal + [-1] * n_anomalies, dtype=np.int32)

    perm = rng.permutation(len(x))
    return x[perm], y_true[perm]


def main() -> None:
    x_data, y_true = generate_network_traffic()

    # contamination define a proporção estimada de anomalias no dataset
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
    )

    # Predição: 1 (normal) e -1 (anomalia)
    y_pred = model.fit_predict(x_data)

    # decision_function: valores negativos indicam maior grau de anomalia
    anomaly_scores = model.decision_function(x_data)

    print("Relatório de detecção (1 = normal, -1 = anomalia):\n")
    print(classification_report(y_true, y_pred, target_names=["Anomalia (-1)", "Normal (1)"]))

    # Exibe os 5 registros mais suspeitos (menores scores)
    top_suspicious_idx = np.argsort(anomaly_scores)[:5]
    print("Top 5 conexões mais anômalas:")
    for rank, idx in enumerate(top_suspicious_idx, start=1):
        b, pps, dur, ports = x_data[idx]
        print(
            f"  {rank}. score={anomaly_scores[idx]:+.4f} | pred={y_pred[idx]:+d} | "
            f"bytes={b:.0f}, pps={pps:.1f}, dur={dur:.1f}s, ports={ports:.0f}"
        )


if __name__ == "__main__":
    main()
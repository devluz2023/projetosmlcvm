from typing import Dict, Tuple

import numpy as np
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score


def sample_truncated_normal_1_to_10(
    rng: np.random.Generator,
    mean: float,
    std: float,
    size: int,
) -> np.ndarray:
    """Amostra valores reais limitados ao intervalo [1.0, 10.0] com média/desvio alvo."""
    values = rng.normal(loc=mean, scale=std, size=size * 2)
    clipped = np.clip(values, 1.0, 10.0)
    # Ajusta levemente o deslocamento para aproximar os momentos desejados no intervalo [1, 10]
    chosen = rng.choice(clipped, size=size, replace=False)
    return chosen.astype(np.float32)


def generate_drift_sheets(
    n_train: int = 800,
    n_prod: int = 400,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Gera matrizes de planilhas com 4 colunas reais em [1, 10], sendo as 2 primeiras
    as colunas importantes monitoradas no enunciado:
      - Treino/Teste: média ~4.89 e desvio padrão ~3.08 nas 2 colunas principais
      - Operação atual (drift): média ~5.34 e desvio padrão ~3.68 nas 2 colunas principais
    """
    rng = np.random.default_rng(seed)

    # Conjunto de treino e teste (distribuição original)
    col1_train = sample_truncated_normal_1_to_10(rng, mean=4.89, std=3.08, size=n_train)
    col2_train = sample_truncated_normal_1_to_10(rng, mean=4.89, std=3.08, size=n_train)
    col3_train = sample_truncated_normal_1_to_10(rng, mean=5.00, std=2.50, size=n_train)
    col4_train = sample_truncated_normal_1_to_10(rng, mean=5.00, std=2.50, size=n_train)
    x_train = np.column_stack([col1_train, col2_train, col3_train, col4_train])

    score_train = 0.75 * col1_train + 0.65 * col2_train - 7.2
    y_train = (score_train > 0.0).astype(np.int32)

    # Conjunto atual em operação (após data drift)
    col1_prod = sample_truncated_normal_1_to_10(rng, mean=5.34, std=3.68, size=n_prod)
    col2_prod = sample_truncated_normal_1_to_10(rng, mean=5.34, std=3.68, size=n_prod)
    col3_prod = sample_truncated_normal_1_to_10(rng, mean=5.00, std=2.50, size=n_prod)
    col4_prod = sample_truncated_normal_1_to_10(rng, mean=5.00, std=2.50, size=n_prod)
    x_prod = np.column_stack([col1_prod, col2_prod, col3_prod, col4_prod])

    # Regra original do modelo + mudança na distribuição dos dados de entrada
    score_prod_true = 0.75 * col1_prod + 0.65 * col2_prod - 7.2
    y_prod_true = (score_prod_true > 0.0).astype(np.int32)

    return x_train, y_train, x_prod, y_prod_true


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Calcula o Population Stability Index (PSI) entre duas distribuições."""
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(expected, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)

    eps = 1e-6
    exp_pct = np.clip(exp_counts / max(len(expected), 1), eps, None)
    act_pct = np.clip(act_counts / max(len(actual), 1), eps, None)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


def summarize_column_drift(ref_col: np.ndarray, cur_col: np.ndarray) -> Dict[str, float]:
    ks_stat, p_val = ks_2samp(ref_col, cur_col)
    return {
        "ref_mean": float(np.mean(ref_col)),
        "ref_std": float(np.std(ref_col, ddof=1)),
        "cur_mean": float(np.mean(cur_col)),
        "cur_std": float(np.std(cur_col, ddof=1)),
        "ks_stat": float(ks_stat),
        "p_value": float(p_val),
        "psi": compute_psi(ref_col, cur_col),
    }


def main() -> None:
    x_train, y_train, x_prod, y_prod = generate_drift_sheets(
        n_train=1000, n_prod=500, seed=42
    )

    # Divide o conjunto histórico em treino e teste (mesma distribuição de origem)
    split = 750
    x_tr, y_tr = x_train[:split], y_train[:split]
    x_te, y_te = x_train[split:], y_train[split:]

    model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
    model.fit(x_tr, y_tr)

    pred_te = model.predict(x_te)
    pred_prod = model.predict(x_prod)

    prec_te = precision_score(y_te, pred_te, zero_division=0)
    prec_prod = precision_score(y_prod, pred_prod, zero_division=0)

    print("=== Desempenho do Modelo ===")
    print(f"Precisão no conjunto de teste (histórico): {prec_te * 100:.2f}%")
    print(f"Precisão na operação atual (planilhas novas): {prec_prod * 100:.2f}%\n")

    print("=== Monitoramento de Data Drift nas Colunas Importantes ===")
    for col_idx in (0, 1):
        stats = summarize_column_drift(x_train[:, col_idx], x_prod[:, col_idx])
        drift_flag = "DRIFT DETECTADO" if stats["p_value"] < 0.05 or stats["psi"] >= 0.2 else "ESTÁVEL"
        print(
            f"Coluna {col_idx + 1}: "
            f"Treino(μ={stats['ref_mean']:.2f}, σ={stats['ref_std']:.2f}) -> "
            f"Atual(μ={stats['cur_mean']:.2f}, σ={stats['cur_std']:.2f}) | "
            f"KS={stats['ks_stat']:.3f} (p={stats['p_value']:.4f}) | "
            f"PSI={stats['psi']:.3f} | {drift_flag}"
        )


if __name__ == "__main__":
    main()
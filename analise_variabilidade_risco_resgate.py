import math
from typing import Dict

import numpy as np
from scipy.stats import norm


def calcular_metricas_vetor(v: np.ndarray) -> Dict[str, float]:
    mediana = float(np.median(v))
    media = float(np.mean(v))
    l1 = float(np.sum(np.abs(v)))
    l2 = float(np.linalg.norm(v, ord=2))
    desvio_mediana_l1 = float(np.sum(np.abs(v - mediana)))
    raiz_sq_media = float(np.sqrt(np.sum((v - media) ** 2)))
    return {
        "norma_l1": l1,
        "norma_l2": l2,
        "soma_abs_desvio_mediana": desvio_mediana_l1,
        "raiz_soma_sq_desvio_media": raiz_sq_media,
    }


def bayes_inadimplencia_normal(mu: float = 6.0, sigma: float = 2.0) -> float:
    p_lt_4 = float(norm.cdf((4.0 - mu) / sigma))
    p_4_7 = float(norm.cdf((7.0 - mu) / sigma) - norm.cdf((4.0 - mu) / sigma))
    p_gt_7 = float(1.0 - norm.cdf((7.0 - mu) / sigma))

    num = 0.4 * p_4_7
    den = 0.2 * p_lt_4 + 0.4 * p_4_7 + 0.6 * p_gt_7
    return num / den


def esperado_aprovados_binomial(
    n_candidatos: int = 2048,
    n_questoes: int = 6,
    p_acerto: float = 0.25,
    min_acertos: int = 4,
) -> float:
    prob_aprovacao = sum(
        math.comb(n_questoes, k) * (p_acerto**k) * ((1.0 - p_acerto) ** (n_questoes - k))
        for k in range(min_acertos, n_questoes + 1)
    )
    return n_candidatos * prob_aprovacao


def mediana_exponencial(taxa_lambda: float = 0.05) -> float:
    return math.log(2.0) / taxa_lambda


def main() -> None:
    v = np.array([2.0, -1.0, 4.0, -2.0])
    metricas = calcular_metricas_vetor(v)
    print("=== 1. Métricas de Variabilidade do Vetor ===")
    for k, val in metricas.items():
        print(f"  {k}: {val:.4f}")

    prob_cond = bayes_inadimplencia_normal()
    print("\n=== 2. Inadimplência Condicional (Normal) ===")
    print(f"  P(4 <= X <= 7 | Inadimplente) ≈ {prob_cond:.4f} (106/215 ≈ {106/215:.4f})")

    esp_aprov = esperado_aprovados_binomial()
    print("\n=== 3. Concurso (Binomial) ===")
    print(f"  Valor esperado de aprovados (>=4 acertos): {esp_aprov:.2f}")

    t_mediana = mediana_exponencial()
    print("\n=== 4. Tempo até 50% de Resgates (Exponencial) ===")
    print(f"  Período mediano: {t_mediana:.2f} meses (~{round(t_mediana)} meses)")


if __name__ == "__main__":
    main()
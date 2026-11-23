import math
import numpy as np
from scipy.stats import chi2, norm, t


def q51_poisson_condicional(taxa_15min: float = 1.0) -> float:
    lam_1h = taxa_15min * 4.0  # 4 denúncias/hora
    # Usando aproximação e^(-4) = 0.018 do enunciado:
    exp_minus_4 = 0.018
    p_ge_2_dado_ge_1 = (1.0 - 5.0 * exp_minus_4) / (1.0 - exp_minus_4)
    return p_ge_2_dado_ge_1


def q52_metodo_momentos(amostra: np.ndarray) -> float:
    x_bar = float(np.mean(amostra))
    # E[X] = (theta + 1) / (theta + 2) = x_bar  =>  theta = (1 - 2*x_bar) / (x_bar - 1)
    return (1.0 - 2.0 * x_bar) / (x_bar - 1.0)


def q53_mle_gama_k2(media_amostral: float = 400.0) -> float:
    # f(x) = theta^2 * x * exp(-theta * x) => theta_hat = 2 / x_bar
    return 2.0 / media_amostral


def q54_ic_variancia_li(n: int = 8, s: float = 8.0, alpha: float = 0.05) -> float:
    df = n - 1
    chi2_sup = float(chi2.ppf(1.0 - alpha / 2.0, df=df))
    return (df * (s**2)) / chi2_sup


def q55_teste_t_fii(
    n: int = 16, x_bar: float = 15.85, mu_0: float = 15.0, s: float = 2.0
) -> dict[str, float | bool]:
    df = n - 1
    t_obs = (x_bar - mu_0) / (s / math.sqrt(n))
    t_crit_10 = float(t.ppf(0.90, df=df))
    t_crit_05 = float(t.ppf(0.95, df=df))
    t_crit_01 = float(t.ppf(0.99, df=df))
    return {
        "t_obs": t_obs,
        "t_crit_10": t_crit_10,
        "t_crit_05": t_crit_05,
        "t_crit_01": t_crit_01,
        "rejeita_10": t_obs > t_crit_10,
        "rejeita_05": t_obs > t_crit_05,
        "rejeita_01": t_obs > t_crit_01,
    }


def q56_limiar_media_z(
    n: int = 16, mu_0: float = 4.0, sigma2: float = 25.0, z_crit: float = -1.28
) -> float:
    sigma = math.sqrt(sigma2)
    return mu_0 + z_crit * (sigma / math.sqrt(n))


def main() -> None:
    print(f"Q51 | P(N>=2 | N>=1) = {q51_poisson_condicional():.6f} (455/491 = {455/491:.6f})")
    print(f"Q52 | theta_MM       = {q52_metodo_momentos(np.array([0.3, 0.2, 0.6, 0.7, 0.2])):.4f} (-1/3)")
    print(f"Q53 | theta_MV       = {q53_mle_gama_k2(400.0):.6f} (1/200)")
    print(f"Q54 | LI IC95%(var)  = {q54_ic_variancia_li():.2f} (~28)")
    print(f"Q55 | Teste t FII    = {q55_teste_t_fii()}")
    print(f"Q56 | Limiar x_bar   = {q56_limiar_media_z():.2f}")


if __name__ == "__main__":
    main()
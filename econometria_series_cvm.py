import numpy as np
from scipy.stats import t


def q57_ic_regressao(
    beta1: float = 0.627, ep_beta1: float = 0.300, n: int = 20, alpha: float = 0.05
) -> tuple[float, float, bool]:
    df = n - 2
    t_crit = float(t.ppf(1.0 - alpha / 2.0, df=df))
    li = beta1 - t_crit * ep_beta1
    ls = beta1 + t_crit * ep_beta1
    significante = not (li <= 0.0 <= ls)
    return li, ls, significante


def q58_cov_estimadores_mqo(x: np.ndarray, residuos: np.ndarray) -> float:
    n = len(x)
    x_bar = float(np.mean(x))
    sxx = float(np.sum((x - x_bar) ** 2))
    sigma2_hat = float(np.sum(residuos**2) / (n - 2))
    return -(x_bar * sigma2_hat) / sxx


def q59_ar1_intercepto(media_longo_prazo: float = 4.0, phi1: float = 0.8) -> float:
    # mu = phi0 / (1 - phi1) => phi0 = mu * (1 - phi1)
    return media_longo_prazo * (1.0 - phi1)


def main() -> None:
    li, ls, sig = q57_ic_regressao()
    print(f"Q57 | IC 95% beta1: [{li:.3f}; {ls:.3f}] | Significante (5%)? {sig}")

    cov_b0_b1 = q58_cov_estimadores_mqo(
        x=np.array([1.0, 2.0, 2.0]),
        residuos=np.array([2.0, 1.0, 1.0]),
    )
    print(f"Q58 | Cov(beta0_hat, beta1_hat) = {cov_b0_b1:.2f}")

    phi0 = q59_ar1_intercepto(media_longo_prazo=4.0, phi1=0.8)
    print(f"Q59 | phi0 do AR(1) = {phi0:.2f}")


if __name__ == "__main__":
    main()
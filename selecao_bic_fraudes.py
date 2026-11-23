from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from xgboost import XGBClassifier


@dataclass
class CandidateFraudModel:
    """Representa um modelo candidato avaliado pelo critério BIC."""

    name: str
    num_params: int          # k
    max_likelihood: float    # L_hat
    weight_scale: float      # intensidade de resposta mockada ao escore de fraude


def generate_fraud_data(n_samples: int = 200, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera base sintética de transações bancárias (n = 200):
    colunas: [valor_normalizado, hora_atipica, frequencia_24h]
    y: 1 = fraude, 0 = legítima
    """
    rng = np.random.default_rng(seed)
    x = np.column_stack(
        [
            rng.normal(0.0, 1.0, size=n_samples),
            rng.normal(0.0, 1.0, size=n_samples),
            rng.normal(0.0, 1.0, size=n_samples),
        ]
    ).astype(np.float32)

    logits = 1.2 * x[:, 0] + 0.9 * x[:, 1] + 0.7 * x[:, 2] - 1.1
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n_samples) < probs).astype(np.int32)
    return x, y


def mock_predict_fraud_proba(model: CandidateFraudModel, x: np.ndarray) -> np.ndarray:
    """Produz probabilidades mockadas de fraude usando apenas os `k` primeiros atributos."""
    active_features = x[:, : model.num_params]
    linear_score = active_features.sum(axis=1) * model.weight_scale - 0.5
    return 1.0 / (1.0 + np.exp(-linear_score))


def compute_bic(num_params: int, n_samples: int, max_likelihood: float) -> float:
    """Calcula BIC = k * ln(n) - 2 * ln(L_hat)."""
    if max_likelihood <= 0.0:
        raise ValueError("max_likelihood deve ser maior que zero.")
    return float(num_params * np.log(n_samples) - 2.0 * np.log(max_likelihood))


def run_bic_selection(x_mock: np.ndarray, y_mock: np.ndarray) -> None:
    n_samples = len(y_mock)

    # M1: k=2, L=0.3 | M2: k=2, L=0.4 | M3: k=2, L=0.5 | M4: k=3, L=0.3 | M5: k=3, L=0.5
    models: List[CandidateFraudModel] = [
        CandidateFraudModel(name="M1", num_params=2, max_likelihood=0.3, weight_scale=0.6),
        CandidateFraudModel(name="M2", num_params=2, max_likelihood=0.4, weight_scale=0.8),
        CandidateFraudModel(name="M3", num_params=2, max_likelihood=0.5, weight_scale=1.0),
        CandidateFraudModel(name="M4", num_params=3, max_likelihood=0.3, weight_scale=0.5),
        CandidateFraudModel(name="M5", num_params=3, max_likelihood=0.5, weight_scale=0.9),
    ]

    print(f"=== Seleção via BIC | Amostra n = {n_samples} ===\n")
    print(f"{'Modelo':<8} {'k':<4} {'L_hat':<8} {'k*ln(n)':<10} {'-2*ln(L)':<10} {'BIC':<10} {'Prob. Média Fraude'}")
    print("-" * 74)

    scored = []
    for m in models:
        probs = mock_predict_fraud_proba(m, x_mock)
        bic = compute_bic(m.num_params, n_samples, m.max_likelihood)
        penalty = m.num_params * np.log(n_samples)
        lik_term = -2.0 * np.log(m.max_likelihood)
        scored.append((bic, m, float(probs.mean())))
        print(
            f"{m.name:<8} {m.num_params:<4} {m.max_likelihood:<8.2f} "
            f"{penalty:<10.4f} {lik_term:<10.4f} {bic:<10.4f} {probs.mean():.4f}"
        )

    best_bic, best_model, _ = min(scored, key=lambda item: item[0])
    print(f"\nModelo selecionado (menor BIC): {best_model.name} (BIC = {best_bic:.4f})\n")


def run_xgboost_baseline(x_train: np.ndarray, y_train: np.ndarray) -> None:
    xgb_model = XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    xgb_model.fit(x_train, y_train)
    fraud_probs = xgb_model.predict_proba(x_train)[:, 1]

    print("=== Baseline com XGBoost ===")
    print(f"Instância treinada: {type(xgb_model).__name__}")
    print(f"Amostras avaliadas: {len(y_train)}")
    print(f"Probabilidade média de fraude prevista: {fraud_probs.mean():.4f}")


def main() -> None:
    x_data, y_data = generate_fraud_data(n_samples=200, seed=42)
    run_bic_selection(x_data.astype(np.float64), y_data)
    run_xgboost_baseline(x_data, y_data)


if __name__ == "__main__":
    main()
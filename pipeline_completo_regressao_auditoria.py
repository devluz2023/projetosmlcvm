from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import (
    confusion_matrix,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


UFS_BRASIL: List[str] = [
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO",
    "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI",
    "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO",
]

ESCOLARIDADE_ORDEM: Dict[str, int] = {
    "fundamental": 0,
    "médio": 1,
    "superior": 2,
    "pós-graduação": 3,
}

LOCAIS_TRABALHO: List[str] = ["empresa", "home office", "misto"]


@dataclass
class PipelineConfig:
    idade_min: float = 18.0
    idade_max: float = 90.0
    target_recall: float = 0.90
    cost_false_negative: float = 50.0
    cost_false_positive: float = 5.0


def encode_observation_32d(
    obs: Dict[str, object],
    cfg: PipelineConfig,
) -> np.ndarray:
    """
    Converte uma observação em exatamente 32 valores numéricos em [0, 1]:
      - 1 valor: idade normalizada
      - 1 valor: escolaridade ordinal normalizada em [0, 1]
      - 27 valores: one-hot para 26 estados + DF
      - 3 valores: one-hot para local de trabalho
    Total: 1 + 1 + 27 + 3 = 32
    """
    idade = float(obs["idade"])  # idade >= 18
    idade_norm = np.clip(
        (idade - cfg.idade_min) / max(cfg.idade_max - cfg.idade_min, 1e-6),
        0.0,
        1.0,
    )

    escol_key = str(obs["escolaridade"]).strip().lower()
    escol_rank = ESCOLARIDADE_ORDEM[escol_key]
    escol_norm = escol_rank / float(len(ESCOLARIDADE_ORDEM) - 1)  # 0.0, 1/3, 2/3, 1.0

    uf_val = str(obs["uf"]).strip().upper()
    uf_one_hot = [1.0 if uf == uf_val else 0.0 for uf in UFS_BRASIL]

    local_val = str(obs["local_trabalho"]).strip().lower()
    local_one_hot = [1.0 if loc == local_val else 0.0 for loc in LOCAIS_TRABALHO]

    vec = np.array(
        [idade_norm, escol_norm, *uf_one_hot, *local_one_hot],
        dtype=np.float32,
    )
    assert vec.shape == (32,), f"Esperado vetor de 32 dimensões, obtido {vec.shape}"
    return vec


def generate_synthetic_records(
    n_samples: int = 600,
    missing_rate: float = 0.15,
    cfg: PipelineConfig = PipelineConfig(),
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gera matriz 32D em [0, 1], alvo contínuo de regressão em [0, 1] e rótulo binário
    de auditoria detalhada (1 = exige exame detalhado, 0 = não exige).
    """
    rng = np.random.default_rng(seed)
    x_matrix = np.zeros((n_samples, 32), dtype=np.float32)

    for i in range(n_samples):
        obs = {
            "idade": int(rng.integers(18, 86)),
            "escolaridade": str(rng.choice(list(ESCOLARIDADE_ORDEM.keys()))),
            "uf": str(rng.choice(UFS_BRASIL)),
            "local_trabalho": str(rng.choice(LOCAIS_TRABALHO)),
        }
        x_matrix[i] = encode_observation_32d(obs, cfg)

    # Alvo contínuo de regressão em [0, 1] (ex.: índice de complexidade/risco)
    raw_score = (
        0.35 * x_matrix[:, 0]      # idade normalizada
        + 0.45 * x_matrix[:, 1]    # escolaridade ordinal normalizada
        + 0.25 * x_matrix[:, 24]   # UF 'SP' como exemplo de peso regional
        + 0.20 * x_matrix[:, 30]   # 'misto'
        + rng.normal(0.0, 0.08, size=n_samples)
    )
    y_reg = np.clip(
        (raw_score - raw_score.min()) / max(raw_score.max() - raw_score.min(), 1e-6),
        0.0,
        1.0,
    ).astype(np.float32)

    # Rótulo binário de auditoria: 1 se exige exame detalhado, 0 caso contrário
    y_audit = (y_reg >= 0.55).astype(np.int32)

    # Insere valores ausentes (NaN) na coluna de idade (índice 0) e escolaridade (índice 1)
    x_with_nan = x_matrix.copy()
    mask_age = rng.random(n_samples) < missing_rate
    mask_edu = rng.random(n_samples) < (missing_rate * 0.7)
    x_with_nan[mask_age, 0] = np.nan
    x_with_nan[mask_edu, 1] = np.nan

    return x_with_nan, y_reg, y_audit


def choose_missing_data_strategy(x_nan: np.ndarray) -> str:
    """
    Avalia número de observações restantes após remoção e correlação absoluta média
    entre atributos contínuos/ordinais para decidir entre remoção ou imputação multivariável.
    """
    n_total = len(x_nan)
    complete_mask = ~np.isnan(x_nan).any(axis=1)
    n_complete = int(complete_mask.sum())
    frac_complete = n_complete / max(n_total, 1)

    valid_rows = x_nan[complete_mask]
    if len(valid_rows) > 2:
        corr = np.corrcoef(valid_rows[:, :2], rowvar=False)
        mean_abs_corr = float(np.abs(corr[0, 1]))
    else:
        mean_abs_corr = 0.0

    print(
        f"Diagnóstico de faltantes -> Linhas completas: {n_complete}/{n_total} "
        f"({frac_complete * 100:.1f}%) | |corr(idade, escolaridade)|: {mean_abs_corr:.3f}"
    )

    if frac_complete >= 0.90 and mean_abs_corr < 0.15:
        return "remocao"
    return "imputacao_multivariavel"


def find_threshold_for_recall(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_recall: float,
) -> float:
    """Seleciona o maior limiar que mantém a revocação >= min_recall."""
    thresholds = np.linspace(0.05, 0.95, 181)
    valid = [
        float(th)
        for th in thresholds
        if recall_score(y_true, (y_proba >= th).astype(np.int32), zero_division=0) >= min_recall
    ]
    if valid:
        return max(valid)
    return float(
        max(
            thresholds,
            key=lambda th: recall_score(
                y_true, (y_proba >= th).astype(np.int32), zero_division=0
            ),
        )
    )


def main() -> None:
    cfg = PipelineConfig()
    x_nan, y_reg, y_audit = generate_synthetic_records(
        n_samples=600, missing_rate=0.15, cfg=cfg, seed=42
    )

    # 1. Escolha entre remoção de observações e imputação multivariável
    strategy = choose_missing_data_strategy(x_nan)
    print(f"Estratégia escolhida: {strategy}\n")

    if strategy == "remocao":
        keep = ~np.isnan(x_nan).any(axis=1)
        x_clean = x_nan[keep]
        y_reg_clean = y_reg[keep]
        y_audit_clean = y_audit[keep]
    else:
        imputer = IterativeImputer(max_iter=15, random_state=42)
        x_clean = np.clip(imputer.fit_transform(x_nan), 0.0, 1.0).astype(np.float32)
        y_reg_clean = y_reg
        y_audit_clean = y_audit

    x_train, x_temp, y_reg_tr, y_reg_tmp, y_aud_tr, y_aud_tmp = train_test_split(
        x_clean, y_reg_clean, y_audit_clean, test_size=0.30, stratify=y_audit_clean, random_state=42
    )
    x_val, x_test, y_reg_val, y_reg_te, y_aud_val, y_aud_te = train_test_split(
        x_temp, y_reg_tmp, y_aud_tmp, test_size=0.50, stratify=y_aud_tmp, random_state=42
    )

    # 2. Rede neural de regressão (entrada 32D em [0, 1] -> saída contínua em [0, 1])
    reg_model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=300,
        random_state=42,
    )
    reg_model.fit(x_train, y_reg_tr)
    y_reg_pred = np.clip(reg_model.predict(x_test), 0.0, 1.0)
    mse = mean_squared_error(y_reg_te, y_reg_pred)
    r2 = r2_score(y_reg_te, y_reg_pred)

    print("=== 1. Rede Neural de Regressão (32 features em [0, 1]) ===")
    print(f"Dimensão da entrada: {x_train.shape[1]} | MSE: {mse:.4f} | R²: {r2:.4f}\n")

    # 3. Classificador de triagem de auditoria focado em revocação (recall)
    clf_model = HistGradientBoostingClassifier(
        max_iter=120, learning_rate=0.05, max_depth=4, random_state=42
    )
    clf_model.fit(x_train, y_aud_tr)

    val_proba = clf_model.predict_proba(x_val)[:, 1]
    best_th = find_threshold_for_recall(y_aud_val, val_proba, cfg.target_recall)

    test_proba = clf_model.predict_proba(x_test)[:, 1]
    test_pred = (test_proba >= best_th).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(y_aud_te, test_pred).ravel()
    rec = recall_score(y_aud_te, test_pred, zero_division=0)
    prec = precision_score(y_aud_te, test_pred, zero_division=0)
    custo = fn * cfg.cost_false_negative + fp * cfg.cost_false_positive

    print(f"=== 2. Classificação de Auditoria (threshold={best_th:.2f}) ===")
    print(f"Revocação (Recall): {rec:.4f} | Precisão: {prec:.4f}")
    print(f"Matriz (VP={tp}, VN={tn}, FP={fp}, FN={fn}) | Custo estimado: R$ {custo:,.2f}")


if __name__ == "__main__":
    main()
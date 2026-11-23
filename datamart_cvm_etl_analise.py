import sqlite3
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, t


def demonstrar_reshape_12() -> np.ndarray:
    """Reproduz arr.reshape(2, 3, 2) sobre os inteiros de 1 a 12."""
    arr = np.arange(1, 13, dtype=np.int32)
    return arr.reshape(2, 3, 2)


def executar_etl_datamart_cvm(
    taxa_usd_brl: float = 5.40,
) -> Tuple[sqlite3.Connection, pd.DataFrame, pd.DataFrame]:
    """
    Simula extração de transações CVM em R$ (BRL) e US$ (USD), limpa linhas
    com dados ausentes usando dropna e carrega Dimensão Moeda + Tabela Fato.
    """
    raw_df = pd.DataFrame(
        [
            {"id_transacao": 101, "ativo": "PETR4", "moeda": "BRL", "valor_transacao": 150000.0},
            {"id_transacao": 102, "ativo": "AAPL", "moeda": "USD", "valor_transacao": 25000.0},
            {"id_transacao": 103, "ativo": "VALE3", "moeda": "BRL", "valor_transacao": np.nan},
            {"id_transacao": 104, "ativo": "MSFT", "moeda": "USD", "valor_transacao": 40000.0},
            {"id_transacao": 105, "ativo": "ITUB4", "moeda": "BRL", "valor_transacao": 90000.0},
        ]
    )

    # Limpeza: remove linhas com células vazias/nulas
    df_limpo = raw_df.dropna().copy()

    # Regra de negócio de conversão para moeda base de análise (BRL)
    fato_df = pd.DataFrame(
        {
            "id_transacao": df_limpo["id_transacao"].astype(int),
            "ativo": df_limpo["ativo"].astype(str),
            "moeda_transacao": df_limpo["moeda"].astype(str),
            "valor_moeda_transacao": df_limpo["valor_transacao"].astype(float),
            "taxa_cambio_brl": np.where(
                df_limpo["moeda"] == "USD", taxa_usd_brl, 1.0
            ),
        }
    )
    fato_df["valor_convertido_brl"] = (
        fato_df["valor_moeda_transacao"] * fato_df["taxa_cambio_brl"]
    )

    dim_moeda = pd.DataFrame(
        [
            {"codigo_moeda": "BRL", "nome_moeda": "Real", "mercado": "Nacional"},
            {"codigo_moeda": "USD", "nome_moeda": "Dólar Americano", "mercado": "Internacional"},
        ]
    )

    conn = sqlite3.connect(":memory:")
    dim_moeda.to_sql("DIM_MOEDA", conn, index=False, if_exists="replace")
    fato_df.to_sql("FATO_TRANSACAO_CVM", conn, index=False, if_exists="replace")

    return conn, raw_df, fato_df


def resumo_estatistico_amostra(valores: np.ndarray, mu_0: float = 0.0) -> Dict[str, float]:
    """Calcula média, desvio amostral, estatística Z e quantis de referência (t e chi2)."""
    n = len(valores)
    media = float(np.mean(valores))
    dp = float(np.std(valores, ddof=1))
    erro_padrao = dp / np.sqrt(n)
    z_stat = (media - mu_0) / max(erro_padrao, 1e-9)
    p_bicaudal = float(2.0 * (1.0 - norm.cdf(abs(z_stat))))

    return {
        "n": float(n),
        "media": media,
        "desvio_padrao": dp,
        "z_stat": float(z_stat),
        "p_valor_z_bicaudal": p_bicaudal,
        "t_crit_df15_bicaudal_5pct": float(t.ppf(0.975, df=15)),
        "chi2_crit_df8_sup_5pct": float(chi2.ppf(0.95, df=8)),
    }


def main() -> None:
    # 1. NumPy reshape(2, 3, 2)
    reshaped = demonstrar_reshape_12()
    print("=== 1. Array NumPy reshape(2, 3, 2) ===")
    print(reshaped)
    print(f"Shape: {reshaped.shape}\n")

    # 2. ETL + Data Mart Financeiro CVM
    conn, raw_df, fato_df = executar_etl_datamart_cvm(taxa_usd_brl=5.40)
    print("=== 2. Limpeza Pandas (dropna) e Fato Consolidada ===")
    print(f"Linhas antes do dropna: {len(raw_df)} | Linhas após dropna: {len(fato_df)}")
    print(fato_df.to_string(index=False))

    consulta_dm = conn.execute(
        """
        SELECT
            m.mercado,
            f.moeda_transacao,
            COUNT(*) AS qtd_operacoes,
            ROUND(SUM(f.valor_convertido_brl), 2) AS total_brl
        FROM FATO_TRANSACAO_CVM f
        INNER JOIN DIM_MOEDA m ON f.moeda_transacao = m.codigo_moeda
        GROUP BY m.mercado, f.moeda_transacao
        ORDER BY f.moeda_transacao;
        """
    ).fetchall()

    print("\nAgregação no Data Mart (SQLite):")
    for mercado, moeda, qtd, total_brl in consulta_dm:
        print(f"  {mercado} ({moeda}): {qtd} operações | Total convertido = R$ {total_brl:,.2f}")

    # 3. Apoio estatístico sobre valores convertidos (normalizados)
    stats = resumo_estatistico_amostra(fato_df["valor_convertido_brl"].to_numpy(), mu_0=150000.0)
    print("\n=== 3. Resumo Estatístico ===")
    for chave, valor in stats.items():
        print(f"  {chave}: {valor:.4f}")

    conn.close()


if __name__ == "__main__":
    main()
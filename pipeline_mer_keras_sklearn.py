import sqlite3
from typing import Tuple

import numpy as np
from sklearn.linear_model import Ridge


def linear_dense_layer(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Simula uma camada densa com ativação linear por omissão (f(z) = z),
    equivalente a tf.keras.layers.Dense(units, activation=None).
    """
    return x @ weights + bias


def init_mer_sqlite() -> sqlite3.Connection:
    """
    Cria em memória uma representação relacional simplificada do MER:
      - A, B, C, E: entidades principais
      - G: entidade fraca associada a C via relacionamento F
      - I, J: subtipos/especializações de E (herdam ligação a E e ao contexto de B)
    """
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE A (
            id_a INTEGER PRIMARY KEY,
            nome TEXT NOT NULL,
            id_a_auto INTEGER REFERENCES A(id_a)
        );

        CREATE TABLE C (
            id_c INTEGER PRIMARY KEY,
            descricao TEXT NOT NULL
        );

        CREATE TABLE B (
            id_b INTEGER PRIMARY KEY,
            id_a INTEGER NOT NULL REFERENCES A(id_a),
            id_c INTEGER NOT NULL REFERENCES C(id_c),
            rotulo TEXT NOT NULL
        );

        CREATE TABLE G (
            id_c INTEGER NOT NULL REFERENCES C(id_c),
            seq_g INTEGER NOT NULL,
            detalhe_fraco TEXT NOT NULL,
            PRIMARY KEY (id_c, seq_g)
        );

        CREATE TABLE E (
            id_e INTEGER PRIMARY KEY,
            id_b INTEGER REFERENCES B(id_b),
            tipo_base TEXT NOT NULL
        );

        CREATE TABLE I (
            id_e INTEGER PRIMARY KEY REFERENCES E(id_e),
            atributo_i TEXT NOT NULL
        );

        CREATE TABLE J (
            id_e INTEGER PRIMARY KEY REFERENCES E(id_e),
            atributo_j TEXT NOT NULL
        );

        INSERT INTO A (id_a, nome, id_a_auto) VALUES
            (1, 'Entidade_A1', NULL),
            (2, 'Entidade_A2', 1);

        INSERT INTO C (id_c, descricao) VALUES
            (10, 'Contexto_C1'),
            (20, 'Contexto_C2');

        INSERT INTO B (id_b, id_a, id_c, rotulo) VALUES
            (100, 1, 10, 'Relacao_B1'),
            (101, 2, 20, 'Relacao_B2');

        INSERT INTO G (id_c, seq_g, detalhe_fraco) VALUES
            (10, 1, 'Detalhe_G_via_F_10'),
            (20, 1, 'Detalhe_G_via_F_20');

        INSERT INTO E (id_e, id_b, tipo_base) VALUES
            (1, 100, 'Subtipo_I'),
            (2, 101, 'Subtipo_J');

        INSERT INTO I (id_e, atributo_i) VALUES (1, 'Valor_I_1');
        INSERT INTO J (id_e, atributo_j) VALUES (2, 'Valor_J_2');
        """
    )
    conn.commit()
    return conn


def fetch_subtypes_and_features(conn: sqlite3.Connection) -> Tuple[np.ndarray, np.ndarray]:
    """
    Consulta subtipos I e J (derivados de E e associados a B) e constrói
    matriz numérica de features X e alvo contínuo y.
    """
    rows = conn.execute(
        """
        SELECT
            E.id_e,
            E.id_b,
            CASE WHEN I.id_e IS NOT NULL THEN 1.0 ELSE 0.0 END AS eh_i,
            CASE WHEN J.id_e IS NOT NULL THEN 1.0 ELSE 0.0 END AS eh_j,
            LENGTH(COALESCE(I.atributo_i, J.atributo_j, '')) AS tam_attr
        FROM E
        LEFT JOIN I ON I.id_e = E.id_e
        LEFT JOIN J ON J.id_e = E.id_e
        ORDER BY E.id_e;
        """
    ).fetchall()

    x = np.array(
        [[float(r[1]), float(r[2]), float(r[3]), float(r[4])] for r in rows],
        dtype=np.float64,
    )
    y = np.array([float(r[0]) * 10.0 + float(r[4]) for r in rows], dtype=np.float64)
    return x, y


def main() -> None:
    conn = init_mer_sqlite()

    # Verifica que subtipos I e J podem associar-se a B através de E
    ligacoes_b = conn.execute(
        """
        SELECT
            B.rotulo,
            E.id_e,
            CASE
                WHEN I.id_e IS NOT NULL THEN 'I'
                WHEN J.id_e IS NOT NULL THEN 'J'
                ELSE 'E'
            END AS subtipo
        FROM B
        INNER JOIN E ON E.id_b = B.id_b
        LEFT JOIN I ON I.id_e = E.id_e
        LEFT JOIN J ON J.id_e = E.id_e
        ORDER BY B.id_b;
        """
    ).fetchall()

    print("=== 1. Ligação dos subtipos I e J a B (via E) no SQLite ===")
    for rotulo, id_e, subtipo in ligacoes_b:
        print(f"  Relação={rotulo} | E.id_e={id_e} | Subtipo={subtipo}")

    # Conjunto de treino (X1, y) e conjunto novo (X2)
    x_all, y_all = fetch_subtypes_and_features(conn)
    x1, y1 = x_all, y_all
    rng = np.random.default_rng(42)
    x2 = np.column_stack(
        [
            rng.choice([100.0, 101.0], size=4),
            rng.choice([0.0, 1.0], size=4),
            rng.choice([0.0, 1.0], size=4),
            rng.integers(5, 15, size=4).astype(np.float64),
        ]
    )

    # Estimador scikit-learn: fit(X1, y) e predict(X2)
    clf = Ridge(alpha=1.0)
    clf.fit(x1, y1)
    pred_sklearn = clf.predict(x2)

    # Camada densa com ativação linear (identidade) aplicada sobre as previsões
    w_dense = np.array([[1.0]], dtype=np.float64)
    b_dense = np.array([0.0], dtype=np.float64)
    out_dense_linear = linear_dense_layer(pred_sklearn.reshape(-1, 1), w_dense, b_dense).ravel()

    print("\n=== 2. Estimador scikit-learn (fit / predict) + Camada Linear ===")
    for idx, (val_sk, val_lin) in enumerate(zip(pred_sklearn, out_dense_linear), start=1):
        print(f"  Amostra X2[{idx}] -> predict(X2)={val_sk:.4f} | Dense(linear)={val_lin:.4f}")

    conn.close()


if __name__ == "__main__":
    main()
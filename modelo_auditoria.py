import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer


@dataclass
class AuditConfig:
    cost_false_negative: float = 50.0
    cost_false_positive: float = 5.0
    target_recall: float = 0.90


PT_STOPWORDS = {
    "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "as", "até",
    "com", "como", "da", "das", "de", "dela", "delas", "dele", "deles",
    "depois", "do", "dos", "e", "ela", "elas", "ele", "eles", "em", "entre",
    "era", "eram", "essa", "essas", "esse", "esses", "esta", "estas", "este",
    "estes", "eu", "foi", "foram", "há", "isso", "isto", "já", "lhe", "lhes",
    "mais", "mas", "me", "mesmo", "meu", "meus", "minha", "minhas", "muito",
    "na", "não", "nas", "nem", "no", "nos", "nós", "num", "numa", "o", "os",
    "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "qual", "quando",
    "que", "quem", "são", "se", "sem", "ser", "seu", "seus", "só", "somos",
    "sou", "sua", "suas", "também", "te", "tem", "têm", "ter", "teu", "teus",
    "toda", "todas", "todo", "todos", "tu", "tua", "tuas", "um", "uma", "umas",
    "uns", "você", "vocês", "vos",
}


def nltk_like_tokenize(text: str) -> List[str]:
    """Tokeniza palavras e pontuações de forma semelhante ao word_tokenize."""
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def filter_stopwords_exact(tokens: List[str], stopwords: set[str] = PT_STOPWORDS) -> List[str]:
    """Replica o comportamento literal: w not in stop_words (sensível a maiúsculas/pontuação)."""
    return [w for w in tokens if w not in stopwords]


def analyze_syntax_sample(tokens: List[str]) -> List[Tuple[str, str, str]]:
    """Anotação morfossintática/sintática simplificada para inspeção textual."""
    verbs = {"roeu", "apresentou", "omitiu", "declarou", "divergiu", "auditou", "registrou"}
    annotated: List[Tuple[str, str, str]] = []
    for idx, tok in enumerate(tokens):
        if re.fullmatch(r"[^\w\s]", tok):
            pos, dep = "PUNCT", "punct"
        elif tok.lower() in verbs:
            pos = "VERB"
            dep = "ROOT" if idx > 0 else "root"
        elif tok[0:1].isupper() and idx > 0:
            pos, dep = "PROPN", "nmod"
        else:
            pos, dep = "NOUN", "obj"
        annotated.append((tok, pos, dep))
    return annotated


SQL_SCHEMA_SEED = """
PRAGMA foreign_keys = ON;

CREATE TABLE AUDITOR (
    ID_A INT NOT NULL,
    Nome VARCHAR(20),
    PRIMARY KEY (ID_A)
);

CREATE TABLE AUDITADO (
    ID_O INT NOT NULL,
    Nome VARCHAR(20),
    ID_A INT,
    PRIMARY KEY (ID_O),
    FOREIGN KEY (ID_A) REFERENCES AUDITOR(ID_A)
);

CREATE TABLE TRIAGEM_AUDITORIA (
    ID_TRIAGEM INTEGER PRIMARY KEY AUTOINCREMENT,
    ID_O INT NOT NULL,
    EXAME_DETALHADO INT NOT NULL,
    PROB_RISCO REAL NOT NULL,
    FOREIGN KEY (ID_O) REFERENCES AUDITADO(ID_O)
);

CREATE TABLE RELATOS_AUDITORIA (
    ID_RELATO INTEGER PRIMARY KEY AUTOINCREMENT,
    ID_O INT NOT NULL,
    PAYLOAD_JSON TEXT NOT NULL,
    TEXTO_RELATO TEXT NOT NULL,
    ROTULO_EXAME INT NOT NULL,
    FOREIGN KEY (ID_O) REFERENCES AUDITADO(ID_O)
);

INSERT INTO AUDITOR (ID_A, Nome) VALUES (1, 'Maite');
INSERT INTO AUDITOR (ID_A, Nome) VALUES (2, 'Lucca');
INSERT INTO AUDITOR (ID_A, Nome) VALUES (3, 'Maria Clara');

INSERT INTO AUDITADO (ID_O, Nome, ID_A) VALUES (1, 'Felipe', 1);
INSERT INTO AUDITADO (ID_O, Nome, ID_A) VALUES (2, 'Stella', 2);
INSERT INTO AUDITADO (ID_O, Nome, ID_A) VALUES (3, 'Patricia', NULL);
"""


def build_nlp_audit_dataset(seed: int = 42) -> Tuple[List[Dict[str, object]], List[str], np.ndarray]:
    """Cria textos e metadados JSON para treino e avaliação do classificador NLP."""
    rng = np.random.default_rng(seed)

    casos_complexos = [
        "Divergência relevante na declaração de receitas e transferência atípica no exterior sem comprovante.",
        "Omissão de passivo contingente e inconsistência entre notas fiscais emitidas e escrituração contábil.",
        "Movimentação fracionada suspeita com indícios de fraude e divergência patrimonial significativa.",
        "Lançamento contábil sem lastro documental e inconsistência fiscal em operação de alto valor.",
    ]
    casos_simples = [
        "Documentação regular conferida sem divergências relevantes no período fiscalizado.",
        "Operação rotineira com comprovantes válidos e conciliação bancária sem inconsistências.",
        "Declaração dentro dos parâmetros esperados sem pendências ou divergências cadastrais.",
        "Registro simples revisado e validado sem indícios de irregularidade na documentação.",
    ]

    texts: List[str] = []
    labels: List[int] = []
    records: List[Dict[str, object]] = []

    n_per_class = 180
    for _ in range(n_per_class):
        t_pos = str(rng.choice(casos_complexos))
        texts.append(t_pos)
        labels.append(1)
        records.append(
            {
                "id_o": int(rng.choice([1, 2, 3])),
                "payload": json.dumps(
                    {
                        "responsaveis": ["Joca", "Ana", "Pedro"],
                        "valor_mil": int(rng.integers(150, 900)),
                        "alerta": True,
                    },
                    ensure_ascii=False,
                ),
                "texto": t_pos,
                "rotulo": 1,
            }
        )

        t_neg = str(rng.choice(casos_simples))
        texts.append(t_neg)
        labels.append(0)
        records.append(
            {
                "id_o": int(rng.choice([1, 2, 3])),
                "payload": json.dumps(
                    {
                        "responsaveis": ["Joca", "Ana", "Pedro"],
                        "valor_mil": int(rng.integers(10, 120)),
                        "alerta": False,
                    },
                    ensure_ascii=False,
                ),
                "texto": t_neg,
                "rotulo": 0,
            }
        )

    perm = rng.permutation(len(texts))
    texts_shuf = [texts[i] for i in perm]
    labels_shuf = np.array([labels[i] for i in perm], dtype=np.int32)
    records_shuf = [records[i] for i in perm]
    return records_shuf, texts_shuf, labels_shuf


def lexical_stats_features(texts: List[str]) -> np.ndarray:
    """Extrai estatísticas lexicais simples (tamanho em tokens e presença de termos de risco)."""
    risk_terms = {"divergência", "omissão", "fraude", "inconsistência", "atípica", "exterior", "sem"}
    feats: List[List[float]] = []
    for txt in texts:
        tokens = filter_stopwords_exact(nltk_like_tokenize(txt))
        lower_toks = [t.lower() for t in tokens]
        risk_hits = sum(1 for t in lower_toks if t in risk_terms)
        feats.append([float(len(tokens)), float(risk_hits)])
    return np.asarray(feats, dtype=np.float32)


def find_threshold_for_recall(y_true: np.ndarray, y_proba: np.ndarray, min_recall: float) -> float:
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
            key=lambda th: recall_score(y_true, (y_proba >= th).astype(np.int32), zero_division=0),
        )
    )


def main() -> None:
    cfg = AuditConfig()
    conn = sqlite3.connect(":memory:")
    conn.executescript(SQL_SCHEMA_SEED)

    # 1. Consulta SQL INNER JOIN da questão
    pares = conn.execute(
        """
        SELECT AUDITOR.Nome AS 'Auditor', AUDITADO.Nome AS 'Auditado'
        FROM AUDITOR
        INNER JOIN AUDITADO ON AUDITADO.ID_A = AUDITOR.ID_A;
        """
    ).fetchall()
    print("=== 1. Resultado INNER JOIN (SQLite) ===")
    for auditor, auditado in pares:
        print(f"  Auditor={auditor} | Auditado={auditado}")

    # Exemplo de filtragem de stop words (questão NLTK) e sintaxe
    exemplo_nltk = "Eu sou um analista de dados da CVM!"
    tokens_filtrados = filter_stopwords_exact(nltk_like_tokenize(exemplo_nltk))
    print(f"\nExemplo stop words NLTK: {tokens_filtrados}")
    print(f"Anotação sintática curta: {analyze_syntax_sample(nltk_like_tokenize('O rato roeu a roupa'))[:3]}")

    # 2. Popula tabela de relatos/JSON no SQLite e treina modelo NLP
    records, texts, y = build_nlp_audit_dataset(seed=42)
    conn.executemany(
        """
        INSERT INTO RELATOS_AUDITORIA (ID_O, PAYLOAD_JSON, TEXTO_RELATO, ROTULO_EXAME)
        VALUES (?, ?, ?, ?);
        """,
        [(r["id_o"], r["payload"], r["texto"], r["rotulo"]) for r in records],
    )
    conn.commit()

    x_train_txt, x_temp_txt, y_train, y_temp = train_test_split(
        texts, y, test_size=0.30, stratify=y, random_state=42
    )
    x_val_txt, x_test_txt, y_val, y_test = train_test_split(
        x_temp_txt, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    features = FeatureUnion(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=200)),
            ("lexical", FunctionTransformer(lexical_stats_features, validate=False)),
        ]
    )

    x_train = features.fit_transform(x_train_txt).toarray()
    x_val = features.transform(x_val_txt).toarray()
    x_test = features.transform(x_test_txt).toarray()

    clf = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.06, random_state=42)
    clf.fit(x_train, y_train)

    val_proba = clf.predict_proba(x_val)[:, 1]
    best_th = find_threshold_for_recall(y_val, val_proba, cfg.target_recall)

    test_proba = clf.predict_proba(x_test)[:, 1]
    test_pred = (test_proba >= best_th).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
    custo = fn * cfg.cost_false_negative + fp * cfg.cost_false_positive

    print(f"\n=== 2. Modelo NLP de Auditoria (threshold={best_th:.2f}) ===")
    print(
        f"Revocação (Recall): {recall_score(y_test, test_pred):.4f} | "
        f"Precisão: {precision_score(y_test, test_pred):.4f} | "
        f"F1: {f1_score(y_test, test_pred):.4f} | "
        f"ROC-AUC: {roc_auc_score(y_test, test_proba):.4f}"
    )
    print(f"Matriz (VP={tp}, VN={tn}, FP={fp}, FN={fn}) | Custo estimado: R$ {custo:,.2f}")
    print(
        classification_report(
            y_test, test_pred, target_names=["Dispensa (0)", "Exame detalhado (1)"]
        )
    )

    # 3. Persiste triagem por auditado no SQLite
    rows_sql = conn.execute(
        "SELECT ID_O, PAYLOAD_JSON, TEXTO_RELATO FROM RELATOS_AUDITORIA"
    ).fetchall()
    x_all = features.transform([r[2] for r in rows_sql]).toarray()
    probs_all = clf.predict_proba(x_all)[:, 1]
    preds_all = (probs_all >= best_th).astype(int)

    conn.executemany(
        """
        INSERT INTO TRIAGEM_AUDITORIA (ID_O, EXAME_DETALHADO, PROB_RISCO)
        VALUES (?, ?, ?);
        """,
        [
            (int(row[0]), int(pred), float(prob))
            for row, pred, prob in zip(rows_sql, preds_all, probs_all)
        ],
    )
    conn.commit()

    resumo_auditados = conn.execute(
        """
        SELECT
            COALESCE(AUDITOR.Nome, 'Sem auditor') AS Auditor,
            AUDITADO.Nome AS Auditado,
            COUNT(TRIAGEM_AUDITORIA.ID_TRIAGEM) AS QtdRelatos,
            SUM(TRIAGEM_AUDITORIA.EXAME_DETALHADO) AS QtdExameDetalhado,
            ROUND(AVG(TRIAGEM_AUDITORIA.PROB_RISCO), 4) AS RiscoMedio
        FROM AUDITADO
        LEFT JOIN AUDITOR ON AUDITADO.ID_A = AUDITOR.ID_A
        LEFT JOIN TRIAGEM_AUDITORIA ON TRIAGEM_AUDITORIA.ID_O = AUDITADO.ID_O
        GROUP BY AUDITADO.ID_O, AUDITADO.Nome, AUDITOR.Nome
        ORDER BY AUDITADO.ID_O;
        """
    ).fetchall()

    print("=== 3. Resumo Consolidado no SQLite por Auditado ===")
    for auditor, auditado, qtd, qtd_exame, risco_med in resumo_auditados:
        print(
            f"  Auditado={auditado} | Auditor={auditor} | "
            f"Relatos={qtd} | ExameDetalhado={qtd_exame} | RiscoMédio={risco_med:.4f}"
        )

    conn.close()


if __name__ == "__main__":
    main()
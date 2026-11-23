import csv
from pathlib import Path
from typing import Dict, List


# 1. Encobrimento (generalização) no atributo Profissão:
#    Agrupa profissões específicas em categorias mais amplas
PROFISSAO_MAPPING: Dict[str, str] = {
    "Agente Autonomo": "Agente",
    "Agente Fiduciário": "Agente",
    "Analista de Sistemas": "Profissional de TI",
    "Engenheiro de Software": "Profissional de TI",
}

# Reagrupamento da Situação conforme o Conjunto de Dados B
SITUACAO_MAPPING: Dict[str, str] = {
    "Em Análise": "Sem acesso",
    "Suspenso": "Sem acesso",
    "Cancelado": "Sem acesso",
    "Normal": "Com acesso",
}

# Mapeamento dos registros do Conjunto A para os identificadores/nomes do Conjunto B
IDENTIDADE_B_MAPPING: Dict[int, Dict[str, object]] = {
    10: {"ID": 2, "Nome": "Luis Alberto Souza"},
    11: {"ID": 4, "Nome": "Juliana Lima Quinta"},
    12: {"ID": 6, "Nome": "Dércio Rui Cunha"},
    13: {"ID": 8, "Nome": "Edson Nilo da Silva"},
}


def truncar_cep(cep: str) -> str:
    """Mantém apenas os 5 primeiros dígitos do CEP."""
    digitos = "".join(ch for ch in cep if ch.isdigit())
    return digitos[:5]


def anonimizar_conjunto_a(registros_a: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Aplica encobrimento em Profissão e gera os registros no formato do Conjunto B."""
    registros_b: List[Dict[str, object]] = []

    for row in registros_a:
        id_original = int(row["ID"])
        profissao_original = str(row["Profissão"])
        situacao_original = str(row["Situação"])
        cep_original = str(row["CEP"])

        # Encobrimento (generalização) da profissão; fallback genérico se não mapeada
        profissao_anon = PROFISSAO_MAPPING.get(profissao_original, profissao_original)
        situacao_anon = SITUACAO_MAPPING.get(situacao_original, situacao_original)
        cep_anon = truncar_cep(cep_original)

        identidade = IDENTIDADE_B_MAPPING.get(
            id_original, {"ID": id_original, "Nome": row["Nome"]}
        )

        registros_b.append(
            {
                "ID": identidade["ID"],
                "Nome": identidade["Nome"],
                "Profissão": profissao_anon,
                "CEP": cep_anon,
                "Situação": situacao_anon,
            }
        )

    return registros_b


def salvar_csv(registros: List[Dict[str, object]], caminho: Path) -> None:
    colunas = ["ID", "Nome", "Profissão", "CEP", "Situação"]
    with caminho.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=colunas)
        writer.writeheader()
        writer.writerows(registros)


def main() -> None:
    conjunto_a: List[Dict[str, object]] = [
        {
            "ID": 10,
            "Nome": "Luis Alberto Nascimento",
            "Profissão": "Agente Autonomo",
            "CEP": "20321-560",
            "Situação": "Em Análise",
        },
        {
            "ID": 11,
            "Nome": "Juliana Lopes Quinta",
            "Profissão": "Agente Fiduciário",
            "CEP": "70125-120",
            "Situação": "Suspenso",
        },
        {
            "ID": 12,
            "Nome": "Dércio Roberto Cunha",
            "Profissão": "Analista de Sistemas",
            "CEP": "15080-121",
            "Situação": "Cancelado",
        },
        {
            "ID": 13,
            "Nome": "Edson Nilo do Nascimento",
            "Profissão": "Engenheiro de Software",
            "CEP": "32800-090",
            "Situação": "Normal",
        },
    ]

    conjunto_b = anonimizar_conjunto_a(conjunto_a)

    print("Conjunto de Dados B gerado:\n")
    for item in conjunto_b:
        print(item)

    saida_csv = Path("conjunto_dados_b.csv")
    salvar_csv(conjunto_b, saida_csv)
    print(f"\nArquivo salvo em: {saida_csv.resolve()}")


if __name__ == "__main__":
    main()
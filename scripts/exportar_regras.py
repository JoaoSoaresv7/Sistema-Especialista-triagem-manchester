"""Exporta a base de conhecimento para tabelas CSV de conferência (dados/regras.csv e
dados/tabela_classificacao.csv). Uso: python scripts/exportar_regras.py"""
import csv
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def main() -> None:
    base = json.loads((RAIZ / "base_conhecimento.json").read_text(encoding="utf-8"))
    with open(RAIZ / "dados" / "tabela_classificacao.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cor", "rotulo", "tempo_max_min", "salience"])
        for cor, i in base["classificacoes"].items():
            w.writerow([cor, i["rotulo"], i["tempo_max_min"], i["salience"]])
    with open(RAIZ / "dados" / "regras.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "queixa", "salience", "terminal", "prioridade", "nome", "justificativa"])
        for r in base["regras"]:
            w.writerow([r["id"], r["queixa"], r["salience"], r["terminal"],
                        r["entao"].get("prioridade", ""), r["nome"], r["justificativa"]])
    print(f"{len(base['regras'])} regras exportadas")


if __name__ == "__main__":
    main()

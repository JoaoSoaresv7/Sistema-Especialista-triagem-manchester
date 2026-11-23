"""Gera dados/casos_sinteticos.csv: casos FICTÍCIOS para treinar e testar o código.

O rótulo vem de uma pontuação de gravidade ponderada com ruído, definida abaixo, e NÃO das
regras do motor. Assim o modelo não apenas copia as regras. Mesmo assim, são dados inventados:
as métricas obtidas com eles demonstram o funcionamento do código, não validade clínica.

Uso: python scripts/gerar_casos_sinteticos.py [N] [SEMENTE]
"""
import csv
import random
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from ml_risco import BOOLEANOS, COLUNAS, NUMERICOS, QUEIXAS  # noqa: E402

PESOS = {  # contribuição de cada fator para a pontuação de gravidade
    "vias_aereas_comprometidas": 10, "sinais_de_choque": 10, "convulsao_ativa": 10,
    "alteracao_subita_consciencia": 6, "hemorragia_maior": 6,
    "dor_precordial_tipica": 3.5, "dor_pleuritica": 1.5, "rigidez_nucal": 4.5, "petequias": 4.5,
    "vomitos_continuos": 1.5, "abdome_rigido": 4.5, "vomito_com_sangue": 5,
    "cefaleia_subita_intensa": 5, "deficit_neurologico": 5, "sinais_infeccao_local": 1.2,
    "febre_associada": 1.2, "ferida_profunda_sutura": 1.5, "troca_curativo_simples": -1,
    "sintomas_gripais": -0.5, "queixa_cronica_sem_agudizacao": -2,
}
LIMITE_ALTO, LIMITE_MEDIO = 6.5, 3.0


def pontuacao(c: dict, ruido: float) -> float:
    s = sum(peso * c[k] for k, peso in PESOS.items())
    s += 2.5 * (1 - c["sinais_vitais_estaveis"])
    s += 0.55 * c["dor_escala"] + 0.35 * max(0, 94 - c["saturacao_o2"])
    s += 1.4 * max(0.0, c["febre_temp"] - 37.5) + 0.25 * c["febre_dias"]
    return s + ruido


def gerar_caso(r: random.Random) -> dict:
    c = {k: 0 for k in COLUNAS}
    q = r.choice(QUEIXAS)
    c["queixa"] = q
    p = lambda x: int(r.random() < x)  # noqa: E731
    c["vias_aereas_comprometidas"], c["sinais_de_choque"] = p(.02), p(.02)
    c["convulsao_ativa"], c["alteracao_subita_consciencia"], c["hemorragia_maior"] = p(.015), p(.03), p(.03)
    c["sinais_vitais_estaveis"], c["queixa_cronica_sem_agudizacao"] = p(.85), p(.12)
    c["saturacao_o2"], c["febre_temp"] = NUMERICOS["saturacao_o2"], NUMERICOS["febre_temp"]
    c["febre_dias"] = 0
    if q == "dor_toracica":
        c["dor_escala"] = r.randint(0, 10)
        c["saturacao_o2"] = round(min(100, max(70, r.gauss(96, 3.5))))
        c["dor_precordial_tipica"], c["dor_pleuritica"] = p(.25), p(.2)
    elif q == "febre":
        c["febre_temp"] = round(min(42, max(36, r.gauss(38.6, 1.0))), 1)
        c["febre_dias"] = r.randint(0, 7)
        c["saturacao_o2"] = round(min(100, max(70, r.gauss(96, 3.5))))
        c["rigidez_nucal"], c["petequias"] = p(.05), p(.04)
        c["vomitos_continuos"], c["sintomas_gripais"] = p(.15), p(.5)
    elif q == "dor_abdominal":
        c["dor_escala"] = r.randint(0, 10)
        c["abdome_rigido"], c["vomito_com_sangue"], c["vomitos_continuos"] = p(.1), p(.04), p(.25)
    elif q == "cefaleia":
        c["dor_escala"] = r.randint(0, 10)
        c["cefaleia_subita_intensa"], c["deficit_neurologico"], c["vomitos_continuos"] = p(.08), p(.06), p(.2)
    else:  # ferida_curativo
        c["dor_escala"] = r.randint(0, 4)
        c["sinais_infeccao_local"] = p(.35)
        c["febre_associada"] = p(.4) if c["sinais_infeccao_local"] else 0
        c["ferida_profunda_sutura"], c["troca_curativo_simples"] = p(.2), p(.35)
    s = pontuacao(c, r.gauss(0, 1.0))
    c["risco"] = "alto" if s >= LIMITE_ALTO else "medio" if s >= LIMITE_MEDIO else "baixo"
    return c


def main(n: int = 600, semente: int = 42) -> None:
    r = random.Random(semente)
    casos = [gerar_caso(r) for _ in range(n)]
    destino = RAIZ / "dados" / "casos_sinteticos.csv"
    with open(destino, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS)
        w.writeheader()
        w.writerows(casos)
    print(f"{n} casos SINTÉTICOS gravados em {destino}")
    print("Distribuição:", dict(Counter(c["risco"] for c in casos)))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 600, int(sys.argv[2]) if len(sys.argv) > 2 else 42)

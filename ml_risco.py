"""Componente de aprendizado supervisionado do sistema híbrido.

Um modelo de Regressão Logística estima o nível de risco ("baixo", "medio", "alto") a partir
dos mesmos fatos que alimentam o motor de regras. A previsão entra no motor como um FATO
(`risco_ml`) e só é usada pelas regras HY1/HY2, que ESCALONAM a prioridade (nunca rebaixam).

- Sem scikit-learn ou sem CSV, o sistema continua funcionando só com regras.
- O modelo é treinado na hora a partir de CSV (nada de arquivo binário), sem dados pessoais.
"""
from __future__ import annotations

import csv
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
CSV_SINTETICO = RAIZ / "dados" / "casos_sinteticos.csv"
CSV_OURO = RAIZ / "dados" / "casos_rotulados_ouro.csv"

LIMIAR_CONFIANCA = 0.60   # abaixo disso a previsão é ignorada (risco_ml = None)
MIN_CASOS = 30            # menos que isso: modo demonstração (números não valem)
ROTULOS = ["baixo", "medio", "alto"]

QUEIXAS = ["dor_toracica", "febre", "dor_abdominal", "cefaleia", "ferida_curativo"]
NUMERICOS = {"dor_escala": 0, "saturacao_o2": 98, "febre_temp": 36.5, "febre_dias": 0}
BOOLEANOS = [
    "vias_aereas_comprometidas", "sinais_de_choque", "convulsao_ativa",
    "alteracao_subita_consciencia", "hemorragia_maior", "sinais_vitais_estaveis",
    "queixa_cronica_sem_agudizacao", "dor_precordial_tipica", "dor_pleuritica",
    "rigidez_nucal", "petequias", "vomitos_continuos", "sintomas_gripais",
    "abdome_rigido", "vomito_com_sangue", "cefaleia_subita_intensa", "deficit_neurologico",
    "sinais_infeccao_local", "febre_associada", "ferida_profunda_sutura",
    "troca_curativo_simples",
]
PADRAO_BOOL = {"sinais_vitais_estaveis": True}   # demais booleanos ausentes = False
COLUNAS = ["queixa"] + list(NUMERICOS) + BOOLEANOS + ["risco"]


def vetor(queixa: str, fatos: dict) -> list[float]:
    """Converte (queixa, fatos) em vetor numérico; fatos ausentes recebem valores padrão."""
    v = [1.0 if queixa == q else 0.0 for q in QUEIXAS]
    for nome, padrao in NUMERICOS.items():
        x = fatos.get(nome)
        v.append(float(padrao if x is None else x))
    for nome in BOOLEANOS:
        x = fatos.get(nome)
        v.append(float(PADRAO_BOOL.get(nome, False) if x is None else bool(x)))
    return v


def ler_casos(caminho: str | Path) -> list[tuple[str, dict, str]]:
    """Lê CSV e devolve [(queixa, fatos, rotulo)]. Arquivo ausente ou vazio => []."""
    caminho = Path(caminho)
    if not caminho.exists():
        return []
    casos = []
    with open(caminho, newline="", encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            fatos = {k: float(linha[k]) for k in NUMERICOS}
            fatos["dor_escala"] = int(fatos["dor_escala"])
            fatos["saturacao_o2"] = int(fatos["saturacao_o2"])
            fatos["febre_dias"] = int(fatos["febre_dias"])
            fatos.update({k: bool(int(linha[k])) for k in BOOLEANOS})
            casos.append((linha["queixa"], fatos, linha["risco"]))
    return casos


class ModeloRisco:
    def __init__(self, limiar: float = LIMIAR_CONFIANCA):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.limiar = limiar
        self.pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        )

    def treinar(self, casos) -> "ModeloRisco":
        X = [vetor(q, f) for q, f, _ in casos]
        y = [r for _, _, r in casos]
        self.pipeline.fit(X, y)
        return self

    def prever_lote(self, casos) -> list[tuple[str, float]]:
        X = [vetor(q, f) for q, f, *_ in casos]
        proba = self.pipeline.predict_proba(X)
        classes = self.pipeline.classes_
        return [(str(classes[p.argmax()]), float(p.max())) for p in proba]

    def prever(self, queixa: str, fatos: dict) -> tuple[str, float]:
        return self.prever_lote([(queixa, fatos)])[0]

    def risco_confiavel(self, queixa: str, fatos: dict) -> str | None:
        """Previsão só é usada se a confiança atingir o limiar; senão None."""
        rotulo, prob = self.prever(queixa, fatos)
        return rotulo if prob >= self.limiar else None


def carregar_modelo(limiar: float = LIMIAR_CONFIANCA):
    """Treina com os casos de ouro (se >= MIN_CASOS) ou com os sintéticos.

    Retorna (modelo, fonte, modo_demonstracao). Levanta exceção se não houver dados/biblioteca.
    """
    ouro = ler_casos(CSV_OURO)
    if len(ouro) >= MIN_CASOS:
        casos, fonte, demo = ouro, "casos_rotulados_ouro.csv", False
    else:
        casos, fonte, demo = ler_casos(CSV_SINTETICO), "casos_sinteticos.csv (SINTÉTICO)", True
    if not casos:
        raise FileNotFoundError("nenhum CSV de treino encontrado em dados/")
    return ModeloRisco(limiar).treinar(casos), fonte, demo

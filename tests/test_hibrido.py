import pytest

pytest.importorskip("sklearn")

from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from ml_risco import CSV_SINTETICO, COLUNAS, QUEIXAS, ROTULOS, ModeloRisco, ler_casos  # noqa: E402
from motor_inferencia import MotorInferencia  # noqa: E402
from triagem_hibrida import ORDEM, classificar  # noqa: E402

motor = MotorInferencia()
CASOS = ler_casos(CSV_SINTETICO)
MODELO = ModeloRisco().treinar(CASOS)


def gravidade(cor):
    return ORDEM.index(cor)


def test_dataset_sintetico_integro():
    assert len(CASOS) >= 300
    assert {q for q, _, _ in CASOS} == set(QUEIXAS)
    assert {r for _, _, r in CASOS} == set(ROTULOS)
    with open(CSV_SINTETICO, encoding="utf-8") as f:
        assert f.readline().strip().split(",") == COLUNAS


def test_modelo_nunca_rebaixa_a_classificacao_das_regras():
    for queixa, fatos, _ in CASOS:
        h = classificar(motor, MODELO, queixa, fatos)
        assert gravidade(h.final.cor) >= gravidade(h.so_regras.cor)


def test_modelo_sozinho_nunca_gera_vermelho():
    for queixa, fatos, _ in CASOS:
        h = classificar(motor, MODELO, queixa, fatos)
        if h.final.cor == "vermelho":
            assert h.so_regras.cor == "vermelho"


def test_regras_hy1_e_hy2_escalonam_resultado_verde():
    base = dict(sinais_vitais_estaveis=True, queixa_cronica_sem_agudizacao=False,
                dor_precordial_tipica=False, saturacao_o2=98, dor_escala=2, dor_pleuritica=False)
    assert motor.inferir("dor_toracica", base).cor == "verde"
    assert motor.inferir("dor_toracica", {**base, "risco_ml": "medio"}).cor == "amarelo"
    assert motor.inferir("dor_toracica", {**base, "risco_ml": "alto"}).cor == "laranja"
    assert motor.inferir("dor_toracica", {**base, "risco_ml": None}).cor == "verde"


def test_risco_ml_nao_rebaixa_laranja_das_regras():
    fatos = dict(sinais_vitais_estaveis=True, dor_precordial_tipica=True, saturacao_o2=98, dor_escala=3)
    assert motor.inferir("dor_toracica", {**fatos, "risco_ml": "medio"}).cor == "laranja"


def test_previsao_abaixo_do_limiar_e_ignorada():
    modelo = ModeloRisco(limiar=1.01).treinar(CASOS)   # limiar inalcançável
    queixa, fatos, _ = CASOS[0]
    assert modelo.risco_confiavel(queixa, fatos) is None
    h = classificar(motor, modelo, queixa, fatos)
    assert h.risco_usado is None and h.final.cor == h.so_regras.cor


def test_sem_modelo_equivale_a_so_regras():
    queixa, fatos, _ = CASOS[0]
    h = classificar(motor, None, queixa, fatos)
    assert h.final.cor == h.so_regras.cor and h.ml_rotulo is None


def test_treinamento_reprodutivel():
    a = ModeloRisco().treinar(CASOS).prever_lote(CASOS[:50])
    b = ModeloRisco().treinar(CASOS).prever_lote(CASOS[:50])
    assert a == b


def test_acuracia_em_holdout_sintetico_acima_do_minimo():
    y = [c[2] for c in CASOS]
    treino, teste = train_test_split(CASOS, test_size=0.25, random_state=42, stratify=y)
    previsto = [p for p, _ in ModeloRisco().treinar(treino).prever_lote(teste)]
    assert accuracy_score([c[2] for c in teste], previsto) >= 0.70

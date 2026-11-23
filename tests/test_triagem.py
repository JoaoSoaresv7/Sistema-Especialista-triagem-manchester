import copy

import pytest

from motor_inferencia import MotorInferencia

motor = MotorInferencia()

GERAIS = dict(
    vias_aereas_comprometidas=False, sinais_de_choque=False, convulsao_ativa=False,
    alteracao_subita_consciencia=False, hemorragia_maior=False,
    sinais_vitais_estaveis=True, queixa_cronica_sem_agudizacao=False,
)


def caso(**kw):
    return {**GERAIS, **kw}


CASOS = [
    # Dor torácica (regras da especificação)
    ("dor_toracica", caso(vias_aereas_comprometidas=True, dor_escala=2), "vermelho"),
    ("dor_toracica", caso(dor_precordial_tipica=True, saturacao_o2=98, dor_escala=3), "laranja"),
    ("dor_toracica", caso(dor_precordial_tipica=False, saturacao_o2=88, dor_escala=3), "laranja"),
    ("dor_toracica", caso(dor_precordial_tipica=False, saturacao_o2=97, dor_escala=9), "laranja"),
    ("dor_toracica", caso(dor_precordial_tipica=False, saturacao_o2=97, dor_escala=3, dor_pleuritica=True), "amarelo"),
    ("dor_toracica", caso(dor_precordial_tipica=False, saturacao_o2=97, dor_escala=6, dor_pleuritica=False), "amarelo"),
    ("dor_toracica", caso(dor_precordial_tipica=False, saturacao_o2=98, dor_escala=2, dor_pleuritica=False), "verde"),
    ("dor_toracica", caso(dor_precordial_tipica=False, saturacao_o2=98, dor_escala=0, dor_pleuritica=False,
                          queixa_cronica_sem_agudizacao=True), "azul"),
    # Segurança: dor leve com sinais vitais instáveis cai na regra conservadora (Amarelo)
    ("dor_toracica", caso(dor_precordial_tipica=False, saturacao_o2=98, dor_escala=2, dor_pleuritica=False,
                          sinais_vitais_estaveis=False), "amarelo"),
    # Febre
    ("febre", caso(febre_temp=41.2, febre_dias=1), "laranja"),
    ("febre", caso(febre_temp=38.5, febre_dias=1, rigidez_nucal=True), "laranja"),
    ("febre", caso(febre_temp=39.5, febre_dias=1), "amarelo"),
    ("febre", caso(febre_temp=38.0, febre_dias=4), "amarelo"),
    ("febre", caso(febre_temp=38.0, febre_dias=1, sintomas_gripais=True, vomitos_continuos=False), "verde"),
    # Dor abdominal
    ("dor_abdominal", caso(dor_escala=4, abdome_rigido=True), "laranja"),
    ("dor_abdominal", caso(dor_escala=6), "amarelo"),
    ("dor_abdominal", caso(dor_escala=3), "verde"),
    # Cefaleia
    ("cefaleia", caso(dor_escala=5, cefaleia_subita_intensa=True), "laranja"),
    ("cefaleia", caso(dor_escala=2, deficit_neurologico=True), "laranja"),
    ("cefaleia", caso(dor_escala=3), "verde"),
    # Ferida / curativo
    ("ferida_curativo", caso(troca_curativo_simples=True, sinais_infeccao_local=False), "azul"),
    ("ferida_curativo", caso(sinais_infeccao_local=True, febre_associada=True), "amarelo"),
    ("ferida_curativo", caso(sinais_infeccao_local=True, febre_associada=False), "verde"),
    ("ferida_curativo", caso(hemorragia_maior=True), "laranja"),
]


@pytest.mark.parametrize("queixa,fatos,esperado", CASOS)
def test_classificacao(queixa, fatos, esperado):
    assert motor.inferir(queixa, fatos).cor == esperado


@pytest.mark.parametrize("queixa", list(motor.fluxogramas))
def test_alteracao_consciencia_e_laranja_em_todos_os_fluxos(queixa):
    assert motor.inferir(queixa, caso(alteracao_subita_consciencia=True)).cor == "laranja"


def test_short_circuit_vermelho_ignora_o_resto():
    res = motor.inferir("dor_toracica", caso(sinais_de_choque=True, dor_escala=1, dor_precordial_tipica=True))
    assert res.cor == "vermelho"
    assert res.trilha[-1][0] == "G1"          # parou na regra de maior saliência
    assert all(rid != "DT2" for rid, _, _ in res.trilha)


def test_forward_chaining_deriva_fatos_intermediarios():
    res = motor.inferir("dor_abdominal", caso(dor_escala=9))
    assert res.fatos_derivados["dor_nivel"] == "intensa"
    assert [t[0] for t in res.trilha][0] == "DER1"   # derivação dispara antes da classificação


def test_sem_estado_e_entrada_nao_mutada():
    fatos = caso(dor_escala=3)
    copia = copy.deepcopy(fatos)
    r1 = motor.inferir("cefaleia", fatos)
    motor.inferir("cefaleia", caso(vias_aereas_comprometidas=True))
    r2 = motor.inferir("cefaleia", fatos)
    assert fatos == copia
    assert r1.cor == r2.cor == "verde"


def test_queixa_desconhecida():
    with pytest.raises(ValueError):
        motor.inferir("inexistente", {})


def test_integridade_da_base():
    ids = [r.id for r in motor.regras]
    assert len(ids) == len(set(ids))
    esperado = {"vermelho": 0, "laranja": 10, "amarelo": 60, "verde": 120, "azul": 240}
    assert {c: v["tempo_max_min"] for c, v in motor.classificacoes.items()} == esperado
    for r in motor.regras:
        assert r.queixa == "*" or r.queixa in motor.fluxogramas
        if r.terminal and r.id != "F1":
            cor = r.entao["prioridade"]
            assert r.salience == motor.classificacoes[cor]["salience"], r.id

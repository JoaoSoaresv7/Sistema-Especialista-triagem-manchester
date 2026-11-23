"""Cola entre o modelo supervisionado (ml_risco.py) e o motor de regras (motor_inferencia.py).

Fluxo: o modelo estima o risco (baixo/medio/alto) a partir dos fatos; se a confiança atingir o
limiar, a previsão entra no motor como o fato `risco_ml`. As regras HY1/HY2 da base de
conhecimento só ESCALONAM a prioridade (Laranja/Amarelo); o modelo nunca rebaixa uma
classificação e nunca decide Vermelho (isso é reservado aos discriminadores críticos).
"""
from __future__ import annotations

from dataclasses import dataclass

from motor_inferencia import MotorInferencia, Resultado

ORDEM = ["azul", "verde", "amarelo", "laranja", "vermelho"]   # da menor para a maior gravidade


@dataclass
class Hibrido:
    final: Resultado               # classificação final (regras + modelo)
    so_regras: Resultado           # classificação que as regras dariam sozinhas
    ml_rotulo: str | None          # previsão bruta do modelo (None se não houve modelo)
    ml_confianca: float | None
    risco_usado: str | None        # previsão que entrou no motor (None se abaixo do limiar)

    @property
    def modelo_escalonou(self) -> bool:
        return ORDEM.index(self.final.cor) > ORDEM.index(self.so_regras.cor)


def classificar(motor: MotorInferencia, modelo, queixa: str, fatos: dict) -> Hibrido:
    """Classifica com regras e, se houver modelo, com a previsão dele como fato adicional."""
    so_regras = motor.inferir(queixa, fatos)
    if modelo is None:
        return Hibrido(so_regras, so_regras, None, None, None)
    rotulo, confianca = modelo.prever(queixa, fatos)
    usado = rotulo if confianca >= modelo.limiar else None
    final = motor.inferir(queixa, {**fatos, "risco_ml": usado})
    return Hibrido(final, so_regras, rotulo, confianca, usado)

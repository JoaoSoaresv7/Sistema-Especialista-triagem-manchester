"""Motor de inferência do Sistema Especialista de Triagem Clínica.

Características (conforme a especificação):
- Encadeamento para frente (forward chaining): parte dos fatos e dispara regras.
- Resolução de conflitos por prioridade (salience): a regra de maior peso na agenda dispara primeiro.
- Parada antecipada (short-circuit): regras terminais encerram a inferência.
- Stateless: toda a memória de trabalho é local à chamada e descartada ao retornar.

O motor NÃO conhece nenhuma regra clínica: tudo vem de base_conhecimento.json.
"""
from __future__ import annotations

import json
import operator
from dataclasses import dataclass, field
from pathlib import Path

BASE_PADRAO = Path(__file__).with_name("base_conhecimento.json")

OPERADORES = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "in": lambda a, b: a in b,
}


def avaliar(cond, memoria: dict) -> bool:
    """Avalia uma condição: [fato, op, valor] ou {"all": [...]} / {"any": [...]}."""
    if isinstance(cond, dict):
        if "all" in cond:
            return all(avaliar(c, memoria) for c in cond["all"])
        if "any" in cond:
            return any(avaliar(c, memoria) for c in cond["any"])
        raise ValueError(f"Condição inválida: {cond}")
    fato, op, valor = cond
    try:
        return bool(OPERADORES[op](memoria.get(fato), valor))
    except TypeError:  # fato ausente (None) comparado com número => não satisfeito
        return False


@dataclass
class Regra:
    id: str
    nome: str
    queixa: str
    salience: int
    se: object
    entao: dict
    terminal: bool
    justificativa: str = ""


@dataclass
class Resultado:
    cor: str
    rotulo: str
    tempo_max_min: int
    regra_id: str
    justificativa: str
    trilha: list = field(default_factory=list)       # [(id, nome, salience)] na ordem de disparo
    fatos_derivados: dict = field(default_factory=dict)


class MotorInferencia:
    def __init__(self, caminho_base: str | Path = BASE_PADRAO):
        with open(caminho_base, encoding="utf-8") as f:
            base = json.load(f)
        self.classificacoes = base["classificacoes"]
        self.perguntas_gerais = base["perguntas_gerais"]
        self.fluxogramas = base["fluxogramas"]
        self.regras = [Regra(**r) for r in base["regras"]]

    def inferir(self, queixa: str, fatos: dict) -> Resultado:
        if queixa not in self.fluxogramas:
            raise ValueError(f"Queixa desconhecida: {queixa}")

        memoria = dict(fatos)                      # cópia: não altera a entrada
        memoria["queixa"] = queixa
        memoria["prioridade"] = None               # INDEFINIDO
        entrada = set(memoria)
        disparadas: set[str] = set()
        trilha = []

        while True:
            agenda = [
                r for r in self.regras
                if r.id not in disparadas
                and r.queixa in ("*", queixa)
                and avaliar(r.se, memoria)
            ]
            if not agenda:
                break
            regra = max(agenda, key=lambda r: r.salience)  # empate: ordem do arquivo
            memoria.update(regra.entao)
            disparadas.add(regra.id)
            trilha.append((regra.id, regra.nome, regra.salience))
            if regra.terminal:                     # short-circuit
                break

        cor = memoria["prioridade"] or "amarelo"   # garantia extra: nunca retorna indefinido
        info = self.classificacoes[cor]
        ultima = next((r for r in self.regras if r.id == trilha[-1][0]), None) if trilha else None
        derivados = {k: v for k, v in memoria.items() if k not in entrada}
        return Resultado(
            cor=cor,
            rotulo=info["rotulo"],
            tempo_max_min=info["tempo_max_min"],
            regra_id=ultima.id if ultima else "-",
            justificativa=ultima.justificativa if ultima else "",
            trilha=trilha,
            fatos_derivados=derivados,
        )

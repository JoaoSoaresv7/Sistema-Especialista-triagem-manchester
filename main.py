#!/usr/bin/env python3
"""Interface de terminal do Sistema Especialista de Triagem Clínica (regras + aprendizado supervisionado).

Uso interativo:   python main.py
Uso direto:       python main.py --queixa febre --fatos '{"febre_temp": 39.5, ...}'
Só regras:        python main.py --sem-ml
Auditoria anônima (opcional): python main.py --auditar auditoria.csv
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from motor_inferencia import MotorInferencia
from triagem_hibrida import classificar

AVISO = (
    "AVISO: protótipo acadêmico de apoio à decisão. Não realiza diagnóstico e não substitui\n"
    "o julgamento do profissional de saúde. Nenhum dado pessoal é coletado ou armazenado."
)
ANSI = {"vermelho": "\033[91m", "laranja": "\033[38;5;208m", "amarelo": "\033[93m",
        "verde": "\033[92m", "azul": "\033[94m"}
RESET = "\033[0m"


def perguntar(p: dict):
    tipo = p["tipo"]
    dica = "(s/n)" if tipo == "bool" else f"[{p['min']}-{p['max']}]"
    while True:
        r = input(f"  {p['texto']} {dica}: ").strip().lower().replace(",", ".")
        if tipo == "bool":
            if r in ("s", "sim"):
                return True
            if r in ("n", "nao", "não"):
                return False
            print("   Responda com s ou n.")
            continue
        try:
            v = int(r) if tipo == "int" else float(r)
        except ValueError:
            print("   Informe um número válido.")
            continue
        if p["min"] <= v <= p["max"]:
            return v
        print(f"   Informe um valor entre {p['min']} e {p['max']}.")


def escolher_queixa(motor: MotorInferencia) -> str:
    ids = list(motor.fluxogramas)
    print("\nQueixa principal (fluxograma):")
    for i, qid in enumerate(ids, 1):
        print(f"  {i}. {motor.fluxogramas[qid]['nome']}")
    while True:
        r = input("Escolha o número: ").strip()
        if r.isdigit() and 1 <= int(r) <= len(ids):
            return ids[int(r) - 1]
        print("Opção inválida.")


def exibir(h, explicar: bool = True) -> None:
    res = h.final
    cor = ANSI[res.cor]
    print("\n" + "=" * 60)
    print(f"{cor}CLASSIFICAÇÃO: {res.cor.upper()} - {res.rotulo}{RESET}")
    tempo = "imediato" if res.tempo_max_min == 0 else f"até {res.tempo_max_min} minutos"
    print(f"Tempo-alvo de atendimento: {tempo}")
    print(f"Motivo ({res.regra_id}): {res.justificativa}")
    if h.ml_rotulo is not None:
        usado = "usada" if h.risco_usado else "ignorada (confiança abaixo do limiar)"
        print(f"Modelo supervisionado: risco {h.ml_rotulo.upper()} "
              f"(confiança {h.ml_confianca:.0%}), previsão {usado}.")
        if h.modelo_escalonou:
            print(f"O modelo ESCALONOU a prioridade (somente regras: {h.so_regras.cor.upper()}). "
                  "Revisão humana recomendada.")
    if explicar:
        print("\nTrilha de inferência (ordem de disparo):")
        for rid, nome, sal in res.trilha:
            print(f"  - [{rid}] {nome} (saliência {sal})")
        derivados = {k: v for k, v in res.fatos_derivados.items() if k != "prioridade"}
        if derivados:
            print(f"Fatos deduzidos pelo motor: {derivados}")
    print("=" * 60)
    print("Sugestão do sistema; a decisão final é do profissional de saúde.")


def auditar(caminho: str, cor: str) -> None:
    """Registro anônimo: apenas carimbo de data/hora e cor. Nada dos dados clínicos."""
    novo = not Path(caminho).exists()
    with open(caminho, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if novo:
            w.writerow(["timestamp", "cor"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), cor])


def carregar_modelo_ou_none(sem_ml: bool):
    if sem_ml:
        return None
    try:
        from ml_risco import carregar_modelo
        modelo, fonte, demo = carregar_modelo()
    except Exception as e:  # scikit-learn ausente, CSV ausente etc.
        print(f"Aviso: modelo indisponível ({type(e).__name__}: {e}). Usando apenas regras.")
        return None
    print(f"Modelo supervisionado treinado com: {fonte}")
    if demo:
        print("MODO DEMONSTRAÇÃO: dados sintéticos; o desempenho do modelo NÃO vale como evidência clínica.")
    return modelo


def main() -> None:
    ap = argparse.ArgumentParser(description="Triagem clínica (sistema especialista híbrido)")
    ap.add_argument("--queixa", help="id do fluxograma (modo direto)")
    ap.add_argument("--fatos", help="fatos em JSON (modo direto)")
    ap.add_argument("--sem-ml", action="store_true", help="usa apenas as regras, sem o modelo")
    ap.add_argument("--auditar", metavar="ARQUIVO.csv", help="registra só timestamp e cor")
    ap.add_argument("--resumo", action="store_true", help="oculta a trilha de inferência")
    args = ap.parse_args()

    motor = MotorInferencia()
    print(AVISO)
    modelo = carregar_modelo_ou_none(args.sem_ml)

    if args.queixa:
        fatos = json.loads(args.fatos or "{}")
        queixa = args.queixa
    else:
        queixa = escolher_queixa(motor)
        fatos = {}
        print("\nDiscriminadores gerais:")
        for p in motor.perguntas_gerais:
            fatos[p["fato"]] = perguntar(p)
        print(f"\nPerguntas do fluxograma '{motor.fluxogramas[queixa]['nome']}':")
        for p in motor.fluxogramas[queixa]["perguntas"]:
            fatos[p["fato"]] = perguntar(p)

    h = classificar(motor, modelo, queixa, fatos)
    exibir(h, explicar=not args.resumo)
    if args.auditar:
        auditar(args.auditar, h.final.cor)
    # 'fatos' sai de escopo aqui: nada clínico é gravado (arquitetura stateless / LGPD)


if __name__ == "__main__":
    main()

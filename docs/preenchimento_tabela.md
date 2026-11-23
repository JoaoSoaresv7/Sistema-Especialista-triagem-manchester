# Texto para preencher a tabela (Grupo 10)

**Nome completo dos integrantes**
1. [SEU NOME COMPLETO]
(2 a 5: deixe em branco, trabalho individual)

**Link do repositório do Projeto:** [COLE AQUI O LINK DO SEU REPOSITÓRIO]

---

## Descrição do SE (0,5 pt)

Sistema Especialista de Triagem Clínica inspirado no Protocolo de Manchester, voltado ao apoio à decisão na classificação de risco em urgência e emergência. Não faz diagnóstico: avalia a gravidade de sintomas agudos e define a cor de prioridade (Vermelho: imediato; Laranja: até 10 min; Amarelo: até 60 min; Verde: até 120 min; Azul: até 240 min). O profissional escolhe uma queixa principal (dor torácica, febre, dor abdominal, cefaleia ou ferida/curativo) e responde perguntas objetivas sobre discriminadores clínicos e sinais vitais. É um sistema híbrido: um modelo de Regressão Logística estima o nível de risco e sua previsão entra no motor como um fato, mas só pode escalonar a prioridade, nunca rebaixar nem decidir Vermelho. A saída traz a cor, o tempo-alvo, a regra que decidiu e a trilha de inferência. Segue a LGPD (stateless, sem dados pessoais, auditoria anônima opcional). Protótipo acadêmico em Python; a decisão final é do profissional de saúde.

## Base de Conhecimento e Regras (0,5 pt)

A base fica separada do código, em `base_conhecimento.json`. Contém as 5 classificações (cor, rótulo, tempo-alvo e saliência), 7 discriminadores gerais, 5 fluxogramas com suas perguntas e 28 regras de produção SE-ENTÃO: 7 de derivação de fatos (ex.: dor_escala >= 8 deduz dor_nivel = intensa), 2 gerais de alto risco (Vermelho: vias aéreas, choque ou convulsão; Laranja: alteração súbita de consciência ou hemorragia maior), 15 regras específicas por fluxograma, 1 de queixa crônica (Azul), 1 de segurança conservadora (padrão Amarelo se nada casar) e 2 do componente híbrido (HY1 e HY2, que escalonam a partir do risco estimado pelo modelo). Cada regra tem saliência alinhada à gravidade (Vermelho 100, Laranja 80, Amarelo 60, Verde 40, Azul 20) e justificativa textual. Fontes: Sistema Manchester (GBCR) como referência e materiais públicos como o Caderno de Atenção Básica nº 28 (Ministério da Saúde). As regras estão exportadas em `dados/regras.csv` para conferência.

## Motor de Inferência (0,5 pt)

Implementado em `motor_inferencia.py`, independente de qualquer regra clínica. Usa encadeamento para frente (forward chaining): parte dos fatos informados, monta a agenda de regras satisfeitas e dispara repetidamente, deduzindo novos fatos até chegar à classificação. Os conflitos são resolvidos por saliência (a regra de maior gravidade dispara primeiro) e há parada antecipada (short-circuit) ao disparar uma regra terminal. A memória de trabalho é local à requisição e descartada ao final (stateless). A integração com o modelo supervisionado (`ml_risco.py` e `triagem_hibrida.py`) roda o motor sem e com a previsão, para comparar e mostrar quando houve escalonamento. A regra de segurança garante que o resultado nunca seja indefinido. Validado por 43 testes automatizados (pytest) e por um notebook executado (`triagem_hibrida.ipynb`) com avaliação do modelo, validação cruzada e verificação de que o modelo nunca rebaixa a classificação.

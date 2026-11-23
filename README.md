# Triagem Clínica Híbrida: regras + aprendizado supervisionado

Sistema especialista de triagem (inspirado no Protocolo de Manchester) com aprendizado supervisionado (Regressão Logística) + regras SE-ENTÃO, para as queixas dor torácica, febre, dor abdominal, cefaleia e ferida/curativo. Classifica o risco em cinco cores (Vermelho 0 min, Laranja 10, Amarelo 60, Verde 120, Azul 240). Protótipo acadêmico de IA e LFA (UDF): **não diagnostica e não substitui o profissional de saúde.**

**Fonte única das regras e dos tempos-alvo:** especificação técnica do projeto e materiais públicos (Caderno de Atenção Básica nº 28, Ministério da Saúde). O Protocolo de Manchester completo (52 fluxogramas) é licenciado e **não** está reproduzido; os limiares clínicos não foram validados por profissional de saúde. Fora do escopo: diagnóstico, prescrição, pediatria, gestantes.

## Como rodar

```bash
pip install -r requirements.txt
python main.py                 # interativo (regras + modelo); use --sem-ml para só regras
python -m pytest -q            # 43 testes
jupyter notebook triagem_hibrida.ipynb
```

Rode a partir desta pasta (lê `base_conhecimento.json` e `dados/` por caminho relativo). Sem scikit-learn, o sistema funciona só com regras.

## Como funciona

O modelo estima o risco (`baixo`, `medio`, `alto`) a partir dos fatos informados. Se a confiança for ≥ 0,60, a previsão entra no motor como o fato `risco_ml`. As regras HY1/HY2 só **escalonam** a prioridade (Laranja/Amarelo): o modelo **nunca rebaixa** a classificação e **nunca decide Vermelho** (reservado aos discriminadores críticos). Em dúvida, a regra F1 escala para Amarelo. Motor: forward chaining, saliência e short-circuit.

## O que falta para o relatório

1. Colocar **casos reais rotulados por profissional** em `dados/casos_rotulados_ouro.csv` (mesmas colunas de `casos_sinteticos.csv`), com **pelo menos 30**. Sem isso o sistema roda em **modo demonstração** e avisa que as métricas não valem.
2. Validar os limiares clínicos das regras com um especialista.
3. Revisar com suas palavras as análises marcadas como *Preencha* no notebook (seção 9).

## Arquivos

- `triagem_hibrida.ipynb`: notebook completo, já executado (avaliação, validação cruzada, limiar de confiança, integração e verificação de segurança).
- `base_conhecimento.json`: classificações, fluxogramas, perguntas e 28 regras. `motor_inferencia.py` é o motor; `ml_risco.py` o modelo; `triagem_hibrida.py` a integração; `main.py` a interface.
- `dados/casos_sinteticos.csv`: 600 casos **FICTÍCIOS** para treinar e testar o código. `regras.csv` e `tabela_classificacao.csv` são conferências exportadas do JSON.
- `scripts/`: gera os casos sintéticos (`gerar_casos_sinteticos.py`) e exporta as regras (`exportar_regras.py`).
- `tests/`: regras e motor (`test_triagem.py`), modelo e integração (`test_hibrido.py`).

## Conferência

43 testes automatizados cobrem todas as cores, short-circuit, ausência de estado, integridade da base, "o modelo nunca rebaixa" e "o modelo sozinho nunca gera Vermelho". Nos dados sintéticos a acurácia fica em torno de 80% a 84%; o rótulo vem de uma pontuação com ruído definida no gerador, não das regras do motor. Mesmo assim os dados são inventados, e as métricas demonstram o código, não validade clínica.

## Limitações conhecidas

O modelo foi treinado com dados sintéticos e sua previsão é um sinal fraco, tratado como sugestão. Sistemas especialistas lidam mal com ambiguidade e exigem atualização manual das regras. Casos atípicos podem escapar das regras; nesses casos vale a regra de segurança (Amarelo) e a avaliação humana. Fatos não informados são tratados como ausentes.


# CronoSaúde Analytics

Análise de dados e aprendizado de máquina para o acompanhamento de pacientes crônicos
em uma operadora de saúde suplementar (Unimed), com dashboard web.

Projeto Integrador em Computação IV (PJI410), Universidade Virtual do Estado de São Paulo (UNIVESP), 2026.

Vídeo de apresentação: https://youtu.be/IDzpbCfHJkQ

## O que o projeto faz

- **Anonimiza** a base de exames da operadora: retira nome, CPF e data de nascimento,
  identifica cada paciente por um código CRONO sorteado e generaliza idade e data (LGPD).
- **Classifica** cada exame em uma de seis categorias crônicas (diabetes, tireoide,
  dislipidemia, renal, hepática e cardiovascular).
- **Segmenta** os pacientes em perfis de utilização com K-Means.
- **Estima** a probabilidade de cada paciente ter exames de duas ou mais condições
  crônicas (Random Forest comparado com Regressão Logística).
- **Simula** a sinistralidade dos exames (despesa ÷ receita) com parâmetros ajustáveis.
- **Mostra** tudo num dashboard Streamlit com acesso restrito.

## Estrutura

```
src/
  config.py                  caminhos e constantes
  codigos_beneficiarios.py   gera a tabela-chave com os códigos CRONO (fica com a operadora)
  anonimizacao.py            anonimiza a base e monta a tabela por paciente
  modelo.py                  K-Means e classificador de multicondição
  banco.py                   monta o banco SQLite do dashboard (exames agregados)
  figuras.py                 figuras e números do relatório
  validacao_temporal.py      treino com 2024, teste com 2025
  sinistralidade.py          simulação com os parâmetros padrão do dashboard
app/
  app.py                     dashboard Streamlit
deploy/                      Docker e nginx para publicação
```

## Como rodar

Requer Python 3.11.

```bash
pip install -r requirements.txt

# 1. tabela-chave (contém dados pessoais: guardar fora do projeto)
python src/codigos_beneficiarios.py CRONICOS.xlsx /pasta/segura/tabela_chave.xlsx

# 2. pipeline
python src/anonimizacao.py CRONICOS.xlsx /pasta/segura/tabela_chave.xlsx
python src/modelo.py
python src/banco.py            # precisa de dados/referencia/tabela_tuss_valores.csv

# 3. dashboard
streamlit run app/app.py
```

Opcionais: `src/figuras.py`, `src/validacao_temporal.py` e `src/sinistralidade.py`.
As saídas ficam em `dados/processados/`, `modelos/` e `resultados/`.

## Dados e privacidade

Os dados da operadora **não fazem parte deste repositório**. A base original, a tabela-chave
e a tabela de valores de procedimentos ficam só com a operadora. O servidor recebe apenas o
banco do dashboard, com pacientes identificados pelo código CRONO e exames agregados por mês,
faixa etária, perfil e procedimento.

## Resultados

Base: 226.422 registros de exames de janeiro de 2024 a junho de 2025, dos quais 209.435 são
de 6.847 pacientes identificáveis.

| Modelo | AUC | Precisão | Recall | F1 |
|---|---|---|---|---|
| Random Forest (escolhido) | 0,969 | 0,919 | 0,933 | 0,926 |
| Regressão Logística | 0,969 | 0,934 | 0,893 | 0,913 |
| Referência: só o total de exames | 0,963 | – | – | – |

- O K-Means separou três perfis (silhueta 0,384): monitoramento pontual, investigação
  concentrada e acompanhamento intensivo.
- Validação temporal (variáveis de 2024, desfecho em 2025): AUC 0,77 e recall 0,81 para
  pacientes que passaram a ter exames de uma condição nova.
- Simulação de sinistralidade com valores de teste (50% da tabela e mensalidades de exemplo):
  os exames consomem 21,5% da receita estimada.

Como o total de exames sozinho já atinge AUC 0,963, o classificador é usado como ordenador
de prioridade, e não como diagnóstico.

## Equipe

Cátia Simone Pereira, Fernando Henrique Bortoti, Leonardo Medeiros Naffek Martinelli de Oliveira,
Rafael Braatz Moreira, Saulo Moreira e Simone Corrêa de Souza.

Orientador: Prof. Me. Marco Antonio de Mello Teixeira.

"""Caminhos e constantes usados em todo o pipeline."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados"
PROCESSADOS = DADOS / "processados"
REFERENCIA = DADOS / "referencia"
MODELOS = RAIZ / "modelos"
RESULTADOS = RAIZ / "resultados"

EXAMES_CSV = PROCESSADOS / "exames_anonimizados.csv"
PACIENTES_CSV = PROCESSADOS / "pacientes_features.csv"
SCORES_CSV = PROCESSADOS / "pacientes_scored.csv"
BANCO = PROCESSADOS / "cronosaude.db"
TABELA_VALORES = REFERENCIA / "tabela_tuss_valores.csv"
METRICAS_JSON = RESULTADOS / "metricas_modelo.json"

FAIXAS = ["0-17", "18-29", "30-39", "40-49", "50-59", "60-69", "70+"]
FAIXA_ORDEM = {f: i for i, f in enumerate(FAIXAS)}

# Faixas etárias da ANS (RN 63/2003), usadas na precificação dos planos.
FAIXAS_ANS = ["0-18", "19-23", "24-28", "29-33", "34-38", "39-43", "44-48", "49-53", "54-58", "59+"]

CATEGORIAS = ["Diabetes", "Tireoide", "Dislipidemia", "Renal", "Hepatico", "Cardiovascular"]

# Variáveis de utilização usadas pelo classificador (não incluem as condições,
# que definem o próprio alvo).
FEATURES = ["faixa_etaria_ord", "n_exames_total", "media_exames_mes", "meses_acompanhamento"]

SEMENTE = 42


def preparar_pastas():
    for pasta in (PROCESSADOS, MODELOS, RESULTADOS):
        pasta.mkdir(parents=True, exist_ok=True)

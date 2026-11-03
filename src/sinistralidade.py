"""
Simulação de sinistralidade dos exames com os parâmetros padrão do dashboard
(mesma conta da aba "Sinistralidade"): despesa = percentual pago x valor de
tabela; receita = mensalidade da faixa ANS x meses do período.

Uso (depois de banco.py):
    python src/sinistralidade.py
"""

import json
import sqlite3

import pandas as pd

import config

PCT_PAGO = 0.50
# Valores de exemplo: tabela pública da Unimed Natal (Unimed Ser Empresarial, enfermaria).
MENSALIDADE = dict(zip(config.FAIXAS_ANS,
                       [118.82, 133.09, 149.05, 171.43, 197.12, 236.55, 298.05, 396.41, 531.20, 711.82]))


def indice(df: pd.DataFrame, por: str, ordem=None) -> dict:
    g = df.groupby(por)[["despesa", "receita"]].sum()
    if ordem:
        g = g.reindex(ordem)
    return (g["despesa"] / g["receita"]).round(4).to_dict()


def main():
    with sqlite3.connect(config.BANCO) as con:
        pac = pd.read_sql("SELECT * FROM pacientes", con)
        meses = pd.read_sql("SELECT DISTINCT ano_ref, mes_ref FROM exames_agregados", con).shape[0]

    pac["receita"] = pac["faixa_ans"].map(MENSALIDADE) * meses
    pac["despesa"] = pac["valor_tabela_total"] * PCT_PAGO
    despesa, receita = pac["despesa"].sum(), pac["receita"].sum()

    saida = {
        "pct_pago": PCT_PAGO,
        "meses": meses,
        "despesa": round(despesa, 2),
        "receita": round(receita, 2),
        "indice": round(despesa / receita, 4),
        "despesa_media_paciente": round(despesa / len(pac), 2),
        "por_perfil": indice(pac, "perfil_utilizacao"),
        "por_faixa_ans": indice(pac, "faixa_ans", config.FAIXAS_ANS),
        "mensalidades": MENSALIDADE,
    }
    config.RESULTADOS.mkdir(exist_ok=True)
    (config.RESULTADOS / "simulacao_sinistralidade.json").write_text(
        json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(saida, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

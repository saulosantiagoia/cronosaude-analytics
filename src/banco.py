"""
Monta o banco SQLite lido pelo dashboard (app/app.py).

Só entram dados anonimizados, e os exames entram agregados por mês, faixa
etária, perfil, categoria e procedimento: nenhuma linha individual de exame
vai para o servidor.

Uso (depois de anonimizacao.py e modelo.py):
    python src/banco.py
"""

import json
import sqlite3

import pandas as pd

import config


def main():
    pac = pd.read_csv(config.SCORES_CSV)
    pac = pac.merge(pd.read_csv(config.PACIENTES_CSV, usecols=["paciente_id_anon", "faixa_ans"]),
                    on="paciente_id_anon", how="left")

    ex = pd.read_csv(config.EXAMES_CSV)
    ex = ex[~ex["e_placeholder_nao_individual"]]
    ex = ex.merge(pac[["paciente_id_anon", "perfil_utilizacao"]], on="paciente_id_anon", how="left")

    # valor de referência por procedimento (tabela da operadora); 0 ou 1 = sem preço
    valores = pd.read_csv(config.TABELA_VALORES, usecols=["cod_proc", "valor"])
    ex["cod_proc"] = pd.to_numeric(ex["cod_proc"], errors="coerce")
    ex = ex.merge(valores, on="cod_proc", how="left")
    ex["valor_tabela"] = ex["valor"].where(ex["valor"] > 1, 0.0).fillna(0.0)
    cobertura = float((ex["valor_tabela"] > 0).mean())

    custo = ex.groupby("paciente_id_anon")["valor_tabela"].sum().rename("valor_tabela_total")
    pac = pac.merge(custo, on="paciente_id_anon", how="left").fillna({"valor_tabela_total": 0.0})

    agregado = (ex.groupby(["ano_ref", "mes_ref", "faixa_etaria", "perfil_utilizacao", "categoria_condicao",
                            "des_procedimento"])
                  .agg(n=("cod_proc", "size"), valor_tabela=("valor_tabela", "sum")).reset_index())
    agregado["categoria_condicao"] = agregado["categoria_condicao"].replace(
        {"Hepatico": "Hepática", "Outros/Rotina": "Outros/rotina"})

    metricas = json.loads(config.METRICAS_JSON.read_text(encoding="utf-8"))
    metricas["cobertura_tuss"] = cobertura

    config.BANCO.unlink(missing_ok=True)
    with sqlite3.connect(config.BANCO) as con:
        pac.to_sql("pacientes", con, index=False)
        agregado.to_sql("exames_agregados", con, index=False)
        pd.DataFrame([{"chave": "metricas", "valor": json.dumps(metricas, ensure_ascii=False)}]).to_sql(
            "info", con, index=False)
        con.execute("CREATE INDEX ix_pac_perfil ON pacientes(perfil_utilizacao, faixa_etaria)")
        con.execute("CREATE INDEX ix_ex_periodo ON exames_agregados(ano_ref, mes_ref)")

    print(f"{config.BANCO}: {len(pac)} pacientes, {agregado['n'].sum()} exames em {len(agregado)} linhas agregadas")
    print(f"exames com valor de tabela: {cobertura:.1%}")


if __name__ == "__main__":
    main()

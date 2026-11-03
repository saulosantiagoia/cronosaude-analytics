"""
Anonimiza a base de exames da operadora (planilha CRONICOS.xlsx, aba DADOS).

- remove nome, CPF e data de nascimento;
- troca cada paciente pelo código CRONO da tabela-chave (codigos_beneficiarios.py);
- generaliza idade em faixa etária e a data em mês/ano;
- classifica cada exame em uma categoria de condição crônica;
- monta a tabela com uma linha por paciente, usada pelo modelo;
- confere o k-anonimato dos quase-identificadores que restam.

A planilha original e a tabela-chave ficam fora do repositório.

Uso:
    python src/anonimizacao.py <CRONICOS.xlsx> <tabela_chave.xlsx>
"""

import sys
import unicodedata

import pandas as pd

import config

# Registro genérico da operadora, que não corresponde a uma pessoa.
NAO_INDIVIDUAL = {"USUARIO EVENTUAL"}

PALAVRAS_CHAVE = {
    "Diabetes": ["GLICOSE", "HEMOGLOBINA GLICADA", "INSULINA", "CURVA GLICEMICA"],
    "Tireoide": ["TIREOESTIMULANTE", "TSH", "T4 LIVRE", "T3", "TIREOGLOBULINA",
                 "ANTI-TPO", "ANTIMICROSSOMAL", "ANTI-TRAB"],
    "Dislipidemia": ["COLESTEROL", "TRIGLICERIDEOS", "LIPOPROTEINA", "APOLIPOPROTEINA"],
    "Renal": ["CREATININA", "UREIA", "MICROALBUMINURIA", "CLEARANCE", "POTASSIO",
              "SODIO", "ACIDO URICO"],
    "Hepatico": ["TRANSAMINASE", "GAMA-GLUTAMIL", "FOSFATASE ALCALINA", "BILIRRUBINA"],
    "Cardiovascular": ["ECG", "ECODOPPLER", "ERGOMETRICO", "DOPPLER VASCULAR",
                       "TROPONINA", "BNP", "CREATINO FOSFOQUINASE"],
}


def normalizar(texto) -> str:
    texto = str(texto).strip().upper()
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")


def chave_beneficiario(nome, nascimento) -> str:
    return normalizar(nome) + "|" + str(nascimento).strip()


def faixa_etaria(idade: int) -> str:
    for faixa, limite in zip(config.FAIXAS, [18, 30, 40, 50, 60, 70]):
        if idade < limite:
            return faixa
    return "70+"


def faixa_ans(idade: int) -> str:
    for faixa, limite in zip(config.FAIXAS_ANS, [18, 23, 28, 33, 38, 43, 48, 53, 58]):
        if idade <= limite:
            return faixa
    return "59+"


def categoria(descricao: str) -> str:
    desc = normalizar(descricao)
    for nome, palavras in PALAVRAS_CHAVE.items():
        if any(normalizar(p) in desc for p in palavras):
            return nome
    return "Outros/Rotina"


def anonimizar(caminho_base: str, caminho_chave: str) -> pd.DataFrame:
    df = pd.read_excel(caminho_base, sheet_name="DADOS")
    chave = pd.read_excel(caminho_chave, dtype=str)
    codigos = dict(zip(chave["chave"], chave["codigo"]))

    df["e_placeholder_nao_individual"] = df["USUARIO"].map(normalizar).isin(NAO_INDIVIDUAL)
    df["paciente_id_anon"] = [codigos.get(chave_beneficiario(n, d))
                              for n, d in zip(df["USUARIO"], df["DT_NASCIMENTO"])]
    df.loc[df["e_placeholder_nao_individual"], "paciente_id_anon"] = "USUARIO-EVENTUAL"
    sem_codigo = df["paciente_id_anon"].isna().sum()
    if sem_codigo:
        sys.exit(f"{sem_codigo} exames sem código: atualize a tabela-chave com codigos_beneficiarios.py")

    df["faixa_etaria"] = df["IDADE"].map(faixa_etaria)
    df["faixa_ans"] = df["IDADE"].map(faixa_ans)
    df["categoria_condicao"] = df["DES_PROCEDIMENTO"].map(categoria)
    df = df.rename(columns={"ANO_REF": "ano_ref", "MES_REF": "mes_ref",
                            "COD_PROC": "cod_proc", "DES_PROCEDIMENTO": "des_procedimento"})
    return df[["paciente_id_anon", "e_placeholder_nao_individual", "faixa_etaria", "faixa_ans",
               "ano_ref", "mes_ref", "cod_proc", "des_procedimento", "categoria_condicao"]]


def montar_pacientes(exames: pd.DataFrame) -> pd.DataFrame:
    base = exames[~exames["e_placeholder_nao_individual"]].copy()
    # mês contínuo, para não confundir jan/2024 com jan/2025
    base["periodo"] = base["ano_ref"] * 12 + base["mes_ref"]

    pac = base.groupby("paciente_id_anon").agg(
        faixa_etaria=("faixa_etaria", "first"),
        faixa_ans=("faixa_ans", lambda s: max(s, key=config.FAIXAS_ANS.index)),
        n_exames_total=("cod_proc", "count"),
        n_condicoes_distintas=("categoria_condicao", lambda s: s[s != "Outros/Rotina"].nunique()),
        periodo_inicio=("periodo", "min"),
        periodo_fim=("periodo", "max"),
    ).reset_index()
    pac["meses_acompanhamento"] = pac["periodo_fim"] - pac["periodo_inicio"] + 1
    pac["media_exames_mes"] = pac["n_exames_total"] / pac["meses_acompanhamento"]

    for cat in config.CATEGORIAS:
        tem = base[base["categoria_condicao"] == cat].groupby("paciente_id_anon").size().gt(0)
        pac[f"flag_{cat.lower()}"] = pac["paciente_id_anon"].map(tem).fillna(False)

    pac["paciente_multicondicao"] = pac["n_condicoes_distintas"] >= 2
    return pac


def k_anonimato(exames: pd.DataFrame, k: int = 5) -> str:
    colunas = ["faixa_etaria", "ano_ref", "mes_ref"]
    grupos = exames.groupby(colunas).size()
    abaixo = int((grupos < k).sum())
    return (f"k-anonimato (k={k}) sobre {colunas}: {len(grupos)} grupos, "
            f"{abaixo} com menos de {k} registros ({abaixo / len(grupos):.1%})")


def main(caminho_base: str, caminho_chave: str):
    config.preparar_pastas()
    exames = anonimizar(caminho_base, caminho_chave)
    pacientes = montar_pacientes(exames)
    exames.to_csv(config.EXAMES_CSV, index=False, encoding="utf-8")
    pacientes.to_csv(config.PACIENTES_CSV, index=False, encoding="utf-8")

    linhas = [
        f"Exames na base original: {len(exames)}",
        f"Exames do registro não individual (excluídos): {int(exames['e_placeholder_nao_individual'].sum())}",
        f"Pacientes: {len(pacientes)}",
        f"Multicondição (2+ categorias): {int(pacientes['paciente_multicondicao'].sum())} "
        f"({pacientes['paciente_multicondicao'].mean():.1%})",
        k_anonimato(exames),
        "Pacientes por categoria:",
    ] + [f"  {cat}: {int(pacientes[f'flag_{cat.lower()}'].sum())}" for cat in config.CATEGORIAS]
    texto = "\n".join(linhas)
    (config.RESULTADOS / "relatorio_anonimizacao.txt").write_text(texto, encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])

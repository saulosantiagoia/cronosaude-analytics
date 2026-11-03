"""
Gera ou atualiza a tabela-chave que liga cada beneficiário a um código
CRONO-NNNNNN sorteado.

A tabela-chave tem nome, CPF e nascimento: fica só com a operadora, fora do
repositório e do servidor. Ao rodar com uma extração nova, os códigos já
atribuídos são mantidos e só os beneficiários novos recebem código.

Uso:
    python src/codigos_beneficiarios.py <CRONICOS.xlsx> <tabela_chave.xlsx>
"""

import secrets
import sys
from pathlib import Path

import pandas as pd

from anonimizacao import NAO_INDIVIDUAL, chave_beneficiario, normalizar

DIGITOS = 6


def main(caminho_base: str, caminho_chave: str):
    df = pd.read_excel(caminho_base, sheet_name="DADOS", usecols=["USUARIO", "CPF", "DT_NASCIMENTO"])
    df = df[~df["USUARIO"].map(normalizar).isin(NAO_INDIVIDUAL)]
    df["chave"] = [chave_beneficiario(n, d) for n, d in zip(df["USUARIO"], df["DT_NASCIMENTO"])]
    df["CPF"] = df["CPF"].astype(str).str.strip().replace({"": None, "0": None, "nan": None})
    pessoas = (df.sort_values("CPF", na_position="last")
                 .groupby("chave", as_index=False)
                 .agg(nome=("USUARIO", "first"), cpf=("CPF", "first"), dt_nascimento=("DT_NASCIMENTO", "first")))

    destino = Path(caminho_chave)
    anterior = pd.read_excel(destino, dtype=str) if destino.exists() else None
    codigos = dict(zip(anterior["chave"], anterior["codigo"])) if anterior is not None else {}

    novos = [c for c in pessoas["chave"] if c not in codigos]
    usados = set(codigos.values())
    livres = [n for n in range(1, 10 ** DIGITOS) if f"CRONO-{n:0{DIGITOS}d}" not in usados]
    for chave, numero in zip(novos, secrets.SystemRandom().sample(livres, len(novos))):
        codigos[chave] = f"CRONO-{numero:0{DIGITOS}d}"

    pessoas["codigo"] = pessoas["chave"].map(codigos)
    if anterior is not None:
        # quem não aparece na extração nova continua na tabela, para consultas ao histórico
        fora = anterior[~anterior["chave"].isin(pessoas["chave"])]
        pessoas = pd.concat([pessoas, fora[pessoas.columns]], ignore_index=True)

    destino.parent.mkdir(parents=True, exist_ok=True)
    pessoas.sort_values("codigo")[["codigo", "nome", "cpf", "dt_nascimento", "chave"]].to_excel(destino, index=False)
    print(f"{destino}: {len(pessoas)} beneficiários, {len(novos)} códigos novos")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])

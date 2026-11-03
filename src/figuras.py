"""
Gera as figuras e os números usados no relatório, a partir das saídas do
pipeline. Grava PNG (300 dpi) e numeros_resultados.json em resultados/.

Uso (depois de modelo.py):
    python src/figuras.py
"""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

import config

AZUL = "#2a78d6"
LARANJA = "#eb6834"
CINZA = "#52514e"
TEXTO = "#0b0b0b"
GRADE = "#e4e3df"
MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
NOME_CATEGORIA = {"Hepatico": "Hepática"}


def estilo():
    plt.switch_backend("Agg")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.edgecolor": GRADE, "axes.labelcolor": CINZA,
        "xtick.color": CINZA, "ytick.color": CINZA,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def br(n, casas=0) -> str:
    """Número no formato brasileiro (1.234,5)."""
    return f"{n:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def salvar(fig, nome):
    fig.savefig(config.RESULTADOS / nome)
    plt.close(fig)


def fig_fluxo():
    etapas = [
        ("Base original", "exames com\nnome e CPF"),
        ("Anonimização", "código CRONO,\nfaixa etária, mês/ano"),
        ("Preparação", "exclusão do registro\nnão individual; 1 linha\npor paciente"),
        ("Segmentação", "K-Means (k=3)\npor silhueta"),
        ("Classificação", "Regressão Logística ×\nRandom Forest"),
        ("Dashboard", "indicadores e lista\nde prioridade"),
    ]
    passo, largura = 2.3, 2.0
    fig, ax = plt.subplots(figsize=(12, 2.4))
    ax.set_xlim(0, len(etapas) * passo)
    ax.set_ylim(0, 2.1)
    ax.axis("off")
    for i, (titulo, detalhe) in enumerate(etapas):
        x = i * passo + 0.05
        ax.add_patch(FancyBboxPatch((x, 0.2), largura, 1.7, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    linewidth=1.2, edgecolor=AZUL, facecolor="#eef4fc"))
        ax.text(x + largura / 2, 1.55, titulo, ha="center", va="center", fontsize=9, fontweight="bold", color=TEXTO)
        ax.text(x + largura / 2, 0.85, detalhe, ha="center", va="center", fontsize=7.5, color=CINZA, linespacing=1.3)
        if i < len(etapas) - 1:
            ax.annotate("", xy=(x + passo - 0.02, 1.05), xytext=(x + largura + 0.04, 1.05),
                        arrowprops=dict(arrowstyle="-|>", color=CINZA, lw=1.2))
    salvar(fig, "fig1_fluxo_pipeline.png")


def fig_exames_mes(ind, numeros):
    serie = ind.groupby(["ano_ref", "mes_ref"]).size().reset_index(name="n")
    rotulos = [f"{MESES[m - 1]}/{str(a)[2:]}" for a, m in zip(serie.ano_ref, serie.mes_ref)]
    x = np.arange(len(serie))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(x, serie.n, color=AZUL, linewidth=2)
    ax.scatter(x, serie.n, color=AZUL, s=22, zorder=3, edgecolor="white", linewidth=1)
    ax.set_xticks(x, rotulos, rotation=45, ha="right", rotation_mode="anchor", fontsize=8.5)
    ax.set_ylim(0, serie.n.max() * 1.15)
    ax.yaxis.set_major_formatter(lambda v, _: br(v))
    ax.set_ylabel("Exames realizados")
    ax.grid(axis="y", color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    for i in (serie.n.idxmin(), serie.n.idxmax()):
        ax.annotate(br(serie.n[i]), (i, serie.n[i]), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8.5, color=TEXTO)
    salvar(fig, "fig2_exames_por_mes.png")
    numeros["exames_mes"] = {r: int(n) for r, n in zip(rotulos, serie.n)}


def fig_condicoes(pa, numeros):
    vals = pd.Series({NOME_CATEGORIA.get(c, c): int(pa[f"flag_{c.lower()}"].sum())
                      for c in config.CATEGORIAS}).sort_values()
    total = len(pa)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.barh(vals.index, vals.values, color=AZUL, height=0.6)
    for i, v in enumerate(vals.values):
        ax.text(v + total * 0.008, i, f"{br(v)} ({br(v / total * 100, 1)}%)", va="center", fontsize=8.5, color=TEXTO)
    ax.set_xlim(0, vals.max() * 1.25)
    ax.xaxis.set_major_formatter(lambda v, _: br(v))
    ax.set_xlabel(f"Pacientes com ao menos um exame da categoria (total: {br(total)})")
    ax.grid(axis="x", color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    salvar(fig, "fig3_pacientes_por_condicao.png")
    numeros["pacientes_por_condicao"] = vals.to_dict()


def fig_n_condicoes(pa, numeros):
    dist = pa["n_condicoes_distintas"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(dist.index.astype(str), dist.values, color=[CINZA if k < 2 else AZUL for k in dist.index], width=0.65)
    for k, v in dist.items():
        ax.text(str(k), v + 20, br(v), ha="center", fontsize=8.5, color=TEXTO)
    ax.set_ylim(0, dist.max() * 1.15)
    ax.set_xlabel("Número de categorias crônicas distintas por paciente")
    ax.set_ylabel("Pacientes")
    ax.yaxis.set_major_formatter(lambda v, _: br(v))
    ax.grid(axis="y", color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.text(0.99, 0.95, "azul = multicondição (2 ou mais)", transform=ax.transAxes, ha="right",
            fontsize=8.5, color=CINZA)
    salvar(fig, "fig4_condicoes_por_paciente.png")
    numeros["n_condicoes"] = {int(k): int(v) for k, v in dist.items()}


def fig_modelo(pa, metricas):
    """Curva ROC do modelo contra a referência que usa só o total de exames."""
    df = pa.assign(faixa_etaria_ord=pa["faixa_etaria"].map(config.FAIXA_ORDEM))
    X, y = df[config.FEATURES].values, df["paciente_multicondicao"].astype(int).values
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.25, random_state=config.SEMENTE, stratify=y)
    m = joblib.load(config.MODELOS / "classificador_multicondicao.joblib")
    proba = m["modelo"].predict_proba(m["scaler"].transform(X_te))[:, 1]

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=GRADE, linewidth=1, linestyle="--")
    for valores, cor, texto, y_txt in [(X_te[:, 1], LARANJA, "Só o total de exames", 0.42),
                                       (proba, AZUL, "Random Forest", 0.62)]:
        fpr, tpr, _ = roc_curve(y_te, valores)
        ax.plot(fpr, tpr, color=cor, linewidth=2)
        ax.plot([0.33, 0.38], [y_txt + 0.04] * 2, color=cor, linewidth=2)
        ax.text(0.40, y_txt, f"{texto}\nAUC = {br(roc_auc_score(y_te, valores), 3)}", color=TEXTO, fontsize=8.5)
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos (recall)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.xaxis.set_major_formatter(lambda v, _: br(v, 1))
    ax.yaxis.set_major_formatter(lambda v, _: br(v, 1))
    ax.grid(color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    salvar(fig, "fig5_curva_roc.png")

    imp = pd.Series(metricas["importancias"])
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    ax.barh(imp.index, imp.values, color=AZUL, height=0.55)
    for i, v in enumerate(imp.values):
        ax.text(v + 0.01, i, f"{br(v * 100, 1)}%", va="center", fontsize=8.5, color=TEXTO)
    ax.set_xlim(0, imp.max() * 1.2)
    ax.xaxis.set_major_formatter(lambda v, _: f"{br(v * 100)}%")
    ax.set_xlabel("Importância relativa no Random Forest")
    ax.grid(axis="x", color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    salvar(fig, "fig6_importancia_variaveis.png")


def resumo(ex, ind, pa, numeros):
    periodo = ex["ano_ref"] * 100 + ex["mes_ref"]
    numeros["base"] = {
        "linhas_originais": int(len(ex)),
        "exames_placeholder": int(ex["e_placeholder_nao_individual"].sum()),
        "exames_individuais": int(len(ind)),
        "pacientes": int(len(pa)),
        "codigos_procedimento": int(ex["cod_proc"].nunique()),
        "periodo_inicio": int(periodo.min()),
        "periodo_fim": int(periodo.max()),
        "meses": int(periodo.nunique()),
        "multicondicao": int(pa["paciente_multicondicao"].sum()),
    }
    numeros["exames_por_categoria"] = ind["categoria_condicao"].value_counts().to_dict()
    numeros["faixa_etaria"] = pa["faixa_etaria"].value_counts().reindex(config.FAIXAS).to_dict()
    numeros["multi_por_faixa"] = (pa.groupby("faixa_etaria")["paciente_multicondicao"].mean()
                                    .reindex(config.FAIXAS).round(4).to_dict())
    numeros["perfis"] = pa.groupby("perfil_utilizacao").agg(
        pacientes=("paciente_id_anon", "count"),
        exames_media=("n_exames_total", "mean"),
        exames_mediana=("n_exames_total", "median"),
        exames_mes=("media_exames_mes", "mean"),
        meses=("meses_acompanhamento", "mean"),
        condicoes=("n_condicoes_distintas", "mean"),
        multicondicao=("paciente_multicondicao", "mean"),
        prob_media=("probabilidade_multicondicao", "mean"),
    ).round(3).to_dict(orient="index")


def main():
    config.preparar_pastas()
    estilo()
    ex = pd.read_csv(config.EXAMES_CSV)
    ind = ex[~ex["e_placeholder_nao_individual"]]
    pa = pd.read_csv(config.SCORES_CSV)
    metricas = json.loads(config.METRICAS_JSON.read_text(encoding="utf-8"))

    numeros = {"metricas": {k: metricas[k] for k in ("regressao_logistica", "random_forest")},
               "silhueta": metricas["silhueta"], "matriz_confusao": metricas["matriz_confusao"],
               "auc_so_n_exames": metricas["auc_so_n_exames"], "importancias": metricas["importancias"]}
    fig_fluxo()
    fig_exames_mes(ind, numeros)
    fig_condicoes(pa, numeros)
    fig_n_condicoes(pa, numeros)
    fig_modelo(pa, metricas)
    resumo(ex, ind, pa, numeros)
    (config.RESULTADOS / "numeros_resultados.json").write_text(
        json.dumps(numeros, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("Figuras e números gravados em", config.RESULTADOS)


if __name__ == "__main__":
    main()

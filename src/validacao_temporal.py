"""
Validação temporal: usa só os exames de 2024 para prever quais pacientes terão,
em 2025, exames de uma categoria crônica que não apareceu em 2024. Aqui o alvo
está no futuro em relação às variáveis, o que testa a capacidade de antecipação.

Uso (depois de anonimizacao.py):
    python src/validacao_temporal.py
"""

import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config
from figuras import AZUL, GRADE, LARANJA, br, estilo

FEATURES = ["faixa_ord", "n_exames_2024", "media_mes_2024", "meses_2024", "n_categorias_2024"]


def categorias_por_paciente(exames: pd.DataFrame) -> pd.Series:
    cronicos = exames[exames["categoria_condicao"].isin(config.CATEGORIAS)]
    return cronicos.groupby("paciente_id_anon")["categoria_condicao"].agg(set)


def montar_base() -> pd.DataFrame:
    ex = pd.read_csv(config.EXAMES_CSV)
    ex = ex[~ex["e_placeholder_nao_individual"]]
    e24, e25 = ex[ex["ano_ref"] == 2024], ex[ex["ano_ref"] == 2025]
    cat24, cat25 = categorias_por_paciente(e24), categorias_por_paciente(e25)

    g = e24.groupby("paciente_id_anon").agg(faixa=("faixa_etaria", "first"), n_exames_2024=("cod_proc", "size"),
                                            mes_min=("mes_ref", "min"), mes_max=("mes_ref", "max"))
    g["meses_2024"] = g["mes_max"] - g["mes_min"] + 1
    g["media_mes_2024"] = g["n_exames_2024"] / g["meses_2024"]
    g["faixa_ord"] = g["faixa"].map(config.FAIXA_ORDEM)
    g["n_categorias_2024"] = g.index.map(lambda p: len(cat24.get(p, set())))

    # só pacientes que voltam em 2025, para os quais se sabe o desfecho
    g = g[g.index.isin(set(e25["paciente_id_anon"]))].copy()
    g["nova_categoria_2025"] = [int(bool(cat25.get(p, set()) - cat24.get(p, set()))) for p in g.index]
    return g.sort_values(FEATURES + ["nova_categoria_2025"], kind="mergesort")


def main():
    g = montar_base()
    X, y = g[FEATURES].values, g["nova_categoria_2025"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=config.SEMENTE, stratify=y)
    sc = StandardScaler().fit(X_tr)

    modelos = {
        "regressao_logistica": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced",
                                                random_state=config.SEMENTE),
    }
    resultado, probas = {}, {}
    for nome, m in modelos.items():
        m.fit(sc.transform(X_tr), y_tr)
        probas[nome] = m.predict_proba(sc.transform(X_te))[:, 1]
        prec, rec, f1, _ = precision_recall_fscore_support(y_te, (probas[nome] >= 0.5).astype(int), average="binary")
        resultado[nome] = {"auc": round(roc_auc_score(y_te, probas[nome]), 3), "precisao": round(prec, 3),
                           "recall": round(rec, 3), "f1": round(f1, 3)}
    resultado["so_n_exames_2024"] = {"auc": round(roc_auc_score(y_te, X_te[:, 1]), 3)}

    saida = {"pacientes_2024_e_2025": int(len(g)), "com_categoria_nova_2025": int(y.sum()),
             "prevalencia": round(float(y.mean()), 4), "teste": int(len(y_te)), "modelos": resultado}
    config.RESULTADOS.mkdir(exist_ok=True)
    (config.RESULTADOS / "validacao_temporal.json").write_text(json.dumps(saida, ensure_ascii=False, indent=2),
                                                               encoding="utf-8")

    estilo()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=GRADE, linestyle="--", linewidth=1)
    for nome, cor, rotulo in [("random_forest", AZUL, "Random Forest"),
                              ("regressao_logistica", LARANJA, "Regressão Logística")]:
        fpr, tpr, _ = roc_curve(y_te, probas[nome])
        ax.plot(fpr, tpr, color=cor, linewidth=2, label=f"{rotulo} (AUC = {br(resultado[nome]['auc'], 3)})")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos (recall)")
    ax.xaxis.set_major_formatter(lambda v, _: br(v, 1))
    ax.yaxis.set_major_formatter(lambda v, _: br(v, 1))
    ax.grid(color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    fig.savefig(config.RESULTADOS / "fig10_validacao_temporal.png")
    print(json.dumps(saida, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

"""
Treina os dois modelos do projeto sobre a tabela de pacientes:

1. Segmentação (K-Means): agrupa os pacientes pelo padrão de utilização;
   o número de grupos é escolhido pelo coeficiente de silhueta.
2. Classificação: estima a probabilidade de o paciente ter exames de duas ou
   mais condições crônicas (multicondição), usando só idade e padrão de uso.
   Compara Regressão Logística e Random Forest.

Saídas: dados/processados/pacientes_scored.csv, modelos/*.joblib,
resultados/metricas_modelo.json e resultados/relatorio_modelo.txt

Uso:
    python src/modelo.py
"""

import json

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config

FEATURES_CLUSTER = config.FEATURES + ["n_condicoes_distintas"]
NOMES_PERFIL = ["Monitoramento pontual", "Investigação concentrada", "Acompanhamento intensivo",
                "Alta prioridade - multicondição"]
NOME_FEATURE = {
    "faixa_etaria_ord": "Faixa etária",
    "n_exames_total": "Total de exames no período",
    "media_exames_mes": "Média de exames por mês",
    "meses_acompanhamento": "Meses de acompanhamento",
}


def carregar() -> pd.DataFrame:
    df = pd.read_csv(config.PACIENTES_CSV)
    df["faixa_etaria_ord"] = df["faixa_etaria"].map(config.FAIXA_ORDEM)
    # Ordena pelo conteúdo, não pelo código do paciente: assim a divisão treino/teste
    # e o K-Means não mudam se os códigos forem trocados.
    ordem = FEATURES_CLUSTER + ["paciente_multicondicao"] + [c for c in df.columns if c.startswith("flag_")]
    return df.sort_values(ordem, kind="mergesort").reset_index(drop=True)


def segmentar(df: pd.DataFrame, k_min=3, k_max=6):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES_CLUSTER].values)

    silhueta, modelos = {}, {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=config.SEMENTE, n_init=10)
        silhueta[k] = silhouette_score(X, km.fit_predict(X))
        modelos[k] = km
    melhor_k = max(silhueta, key=silhueta.get)
    df["cluster"] = modelos[melhor_k].predict(X)

    # nomes em ordem crescente de intensidade (condições, depois exames por mês)
    resumo = df.groupby("cluster")[["n_exames_total", "media_exames_mes", "n_condicoes_distintas"]].mean()
    ordem = resumo.sort_values(["n_condicoes_distintas", "media_exames_mes"]).index
    nomes = {c: NOMES_PERFIL[min(i, len(NOMES_PERFIL) - 1)] for i, c in enumerate(ordem)}
    df["perfil_utilizacao"] = df["cluster"].map(nomes)

    joblib.dump({"scaler": scaler, "kmeans": modelos[melhor_k], "nomes": nomes, "features": FEATURES_CLUSTER},
                config.MODELOS / "kmeans_segmentacao.joblib")
    texto = "\n".join([f"k={k}: silhueta={s:.3f}" for k, s in silhueta.items()]
                      + [f"Escolhido: k={melhor_k}", "", resumo.round(2).to_string()])
    return {str(k): round(s, 3) for k, s in silhueta.items()}, texto


def classificar(df: pd.DataFrame):
    X = df[config.FEATURES].values
    y = df["paciente_multicondicao"].astype(int).values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=config.SEMENTE, stratify=y)
    scaler = StandardScaler().fit(X_tr)

    candidatos = {
        "regressao_logistica": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced",
                                                random_state=config.SEMENTE),
    }
    metricas, texto = {}, [f"Treino: {len(X_tr)} pacientes | teste: {len(X_te)} | "
                           f"multicondição no teste: {y_te.mean():.1%}", ""]
    for nome, modelo in candidatos.items():
        modelo.fit(scaler.transform(X_tr), y_tr)
        proba = modelo.predict_proba(scaler.transform(X_te))[:, 1]
        pred = modelo.predict(scaler.transform(X_te))
        prec, rec, f1, _ = precision_recall_fscore_support(y_te, pred, average="binary")
        metricas[nome] = {"auc": roc_auc_score(y_te, proba), "precisao": prec, "recall": rec, "f1": f1,
                          "acuracia": accuracy_score(y_te, pred)}
        texto += [f"[{nome}] " + " | ".join(f"{k}={v:.3f}" for k, v in metricas[nome].items()),
                  classification_report(y_te, pred, digits=3)]

    # Entre os modelos com AUC a até 0,005 da melhor, fica o de maior recall:
    # deixar de sinalizar quem precisa de acompanhamento custa mais que um alerta falso.
    melhor_auc = max(m["auc"] for m in metricas.values())
    escolhido = max((n for n in metricas if melhor_auc - metricas[n]["auc"] <= 0.005),
                    key=lambda n: metricas[n]["recall"])
    metricas = {n: {k: round(float(v), 3) for k, v in m.items()} for n, m in metricas.items()}
    modelo = candidatos[escolhido]
    texto.append(f"Modelo escolhido: {escolhido}")
    joblib.dump({"scaler": scaler, "modelo": modelo, "nome": escolhido, "features": config.FEATURES},
                config.MODELOS / "classificador_multicondicao.joblib")

    proba_teste = modelo.predict_proba(scaler.transform(X_te))[:, 1]
    vn, fp, fn, vp = confusion_matrix(y_te, (proba_teste >= 0.5).astype(int)).ravel()
    extras = {
        "modelo_escolhido": escolhido,
        "pacientes_teste": int(len(X_te)),
        "matriz_confusao": {"VN": int(vn), "FP": int(fp), "FN": int(fn), "VP": int(vp)},
        # referência: ordenar só pelo total de exames
        "auc_so_n_exames": round(roc_auc_score(y_te, X_te[:, 1]), 3),
    }
    if hasattr(modelo, "feature_importances_"):
        imp = pd.Series(modelo.feature_importances_, index=[NOME_FEATURE[f] for f in config.FEATURES]).sort_values()
        extras["importancias"] = imp.round(3).to_dict()

    probabilidade = modelo.predict_proba(scaler.transform(X))[:, 1]
    return probabilidade, metricas | extras, "\n".join(texto)


def main():
    config.preparar_pastas()
    df = carregar()
    silhueta, texto_cluster = segmentar(df)
    df["probabilidade_multicondicao"], metricas, texto_classif = classificar(df)
    metricas["silhueta"] = silhueta

    colunas = ["paciente_id_anon", "faixa_etaria", "n_exames_total", "media_exames_mes", "meses_acompanhamento",
               "n_condicoes_distintas", "paciente_multicondicao", "cluster", "perfil_utilizacao",
               "probabilidade_multicondicao"] + [f"flag_{c.lower()}" for c in config.CATEGORIAS]
    df[colunas].to_csv(config.SCORES_CSV, index=False, encoding="utf-8")
    config.METRICAS_JSON.write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")

    texto = f"SEGMENTAÇÃO (K-Means)\n{texto_cluster}\n\nCLASSIFICAÇÃO (multicondição)\n{texto_classif}"
    (config.RESULTADOS / "relatorio_modelo.txt").write_text(texto, encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    main()

"""
CronoSaúde Analytics: dashboard em Streamlit.

Lê o banco SQLite gerado por src/banco.py (só dados anonimizados, com os
exames agregados). Para rodar localmente:
    streamlit run app/app.py
"""

import json
import os
import sqlite3

import altair as alt
import pandas as pd
import streamlit as st

DB = os.environ.get("CRONOSAUDE_DB", "dados/processados/cronosaude.db")

FAIXAS = ["0-17", "18-29", "30-39", "40-49", "50-59", "60-69", "70+"]
PERFIS = ["Monitoramento pontual", "Investigação concentrada", "Acompanhamento intensivo"]
# cores fixas por perfil (a cor segue o perfil, não a posição)
COR_PERFIL = {"Monitoramento pontual": "#2a78d6", "Investigação concentrada": "#eb6834", "Acompanhamento intensivo": "#1baf7a"}
COR = "#2a78d6"
FAIXAS_ANS = ["0-18", "19-23", "24-28", "29-33", "34-38", "39-43", "44-48", "49-53", "54-58", "59+"]
# Valores de EXEMPLO (tabela pública da Unimed Natal, Unimed Ser Empresarial, enfermaria).
# A operadora substitui pelos próprios valores na tela.
MENSALIDADE_EXEMPLO = [118.82, 133.09, 149.05, 171.43, 197.12, 236.55, 298.05, 396.41, 531.20, 711.82]
CATEGORIAS = ["Diabetes", "Tireoide", "Dislipidemia", "Renal", "Hepatico", "Cardiovascular"]
NOME_CAT = {"Hepatico": "Hepática"}
MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]

st.set_page_config(page_title="CronoSaúde Analytics", page_icon="🩺", layout="wide")


def br(v, casas=0):
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def reais(v):
    return "R$ " + br(v, 0 if abs(v) >= 1000 else 2)


NUM_BR = "replace(replace(replace(format(datum.value, ',.0f'), ',', '#'), '.', ','), '#', '.')"


def barras(df, cat, val, titulo_x, ordem=None, cores=None, pct=False, altura_barra=22, tooltip=None):
    """Barras horizontais com rótulo direto e números no padrão brasileiro."""
    df = df.copy()
    df["_rotulo"] = df[val].map(lambda v: f"{br(v * 100, 1)}%" if pct else br(v))
    cor = (alt.Color(f"{cat}:N", scale=alt.Scale(domain=list(cores), range=list(cores.values())), legend=None)
           if cores else alt.value(COR))
    eixo_x = alt.Axis(format="%") if pct else alt.Axis(labelExpr=NUM_BR)
    base = alt.Chart(df).encode(
        y=alt.Y(f"{cat}:N", sort=ordem or "-x", title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X(f"{val}:Q", title=titulo_x, axis=eixo_x, scale=alt.Scale(domainMax=float(df[val].max()) * 1.15)),
        tooltip=tooltip or [alt.Tooltip(f"{cat}:N", title=cat.capitalize()), alt.Tooltip("_rotulo:N", title=titulo_x)],
    )
    grafico = (base.mark_bar(cornerRadiusEnd=4, height=altura_barra).encode(color=cor)
               + base.mark_text(align="left", dx=6, color="gray").encode(text="_rotulo:N"))
    return grafico.properties(height=len(df) * (altura_barra + 14) + 30)


def mostrar(grafico):
    altura = grafico.height if isinstance(grafico.height, int) else "content"
    st.altair_chart(grafico, width="stretch", height=altura)


@st.cache_data
def carregar():
    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as con:
        pac = pd.read_sql("SELECT * FROM pacientes", con)
        exa = pd.read_sql("SELECT * FROM exames_agregados", con)
        info = json.loads(pd.read_sql("SELECT valor FROM info WHERE chave='metricas'", con).iloc[0, 0])
    exa["periodo"] = exa["ano_ref"] * 100 + exa["mes_ref"]
    return pac, exa, info


pac, exa, info = carregar()
periodos = sorted(exa["periodo"].unique())
rotulo = lambda p: f"{MESES[p % 100 - 1]}/{str(p // 100)[2:]}"

# ---------------------------------------------------------------- filtros
with st.sidebar:
    st.header("Filtros")
    f_faixa = st.multiselect("Faixa etária", FAIXAS, default=FAIXAS)
    f_perfil = st.multiselect("Perfil de utilização", PERFIS, default=PERFIS)
    p_ini, p_fim = st.select_slider("Período dos exames", options=periodos, value=(periodos[0], periodos[-1]),
                                    format_func=rotulo)
    st.caption("Os indicadores de pacientes consideram todo o período (jan/24 a jun/25). "
               "O filtro de período vale para os gráficos de exames.")
    st.divider()
    st.caption("Dados anonimizados: pacientes identificados só pelo código CRONO (LGPD). "
               "Fonte: Unimed. PI IV — UNIVESP, 2026.")

pf = pac[pac["faixa_etaria"].isin(f_faixa) & pac["perfil_utilizacao"].isin(f_perfil)]
ef = exa[exa["faixa_etaria"].isin(f_faixa) & exa["perfil_utilizacao"].isin(f_perfil)
         & exa["periodo"].between(p_ini, p_fim)]

st.title("CronoSaúde Analytics")
st.caption("Acompanhamento de pacientes crônicos — Unimed · base anonimizada, jan/2024 a jun/2025")

if pf.empty:
    st.warning("Nenhum paciente com esses filtros.")
    st.stop()

aba_geral, aba_prior, aba_sin, aba_modelo = st.tabs(
    ["Visão geral", "Pacientes prioritários", "Sinistralidade (simulação)", "Sobre o modelo"])

# ----------------------------------------------------------- visão geral
with aba_geral:
    n = len(pf)
    multi = int(pf["paciente_multicondicao"].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pacientes", br(n), help="Beneficiários com exames no período, após excluir o registro genérico 'usuário eventual'.")
    c2.metric("Multicondição", f"{br(multi)} ({br(multi / n * 100, 1)}%)",
              help="Exames de 2 ou mais categorias crônicas diferentes. Indica padrão de acompanhamento, não diagnóstico.")
    c3.metric("Exames no período", br(ef["n"].sum()))
    c4.metric("Exames por paciente", br(pf["n_exames_total"].mean(), 1), help="Média no período completo.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Pacientes por perfil")
        dp = pf["perfil_utilizacao"].value_counts().reindex(PERFIS).dropna().reset_index()
        dp.columns = ["perfil", "pacientes"]
        dp["_pct"] = dp["pacientes"].map(lambda v: f"{br(v / n * 100, 1)}%")
        mostrar(barras(dp, "perfil", "pacientes", "Pacientes", ordem=PERFIS, cores=COR_PERFIL, altura_barra=28,
                       tooltip=[alt.Tooltip("perfil:N", title="Perfil"), alt.Tooltip("_rotulo:N", title="Pacientes"),
                                alt.Tooltip("_pct:N", title="% do total")]))
        st.caption("Pontual: poucos exames, maioria sem multicondição. Concentrada: menos de 2 meses de acompanhamento "
                   "com muitos exames por mês. Intensivo: acompanhamento por mais de um ano, quase todos multicondição.")
    with col_b:
        st.subheader("Pacientes por condição monitorada")
        dc = pd.DataFrame({"categoria": [NOME_CAT.get(c, c) for c in CATEGORIAS],
                           "pacientes": [int(pf[f"flag_{c.lower()}"].sum()) for c in CATEGORIAS]})
        dc["_pct"] = dc["pacientes"].map(lambda v: f"{br(v / n * 100, 1)}%")
        mostrar(barras(dc, "categoria", "pacientes", "Pacientes com ao menos um exame da categoria",
                       tooltip=[alt.Tooltip("categoria:N", title="Categoria"), alt.Tooltip("_rotulo:N", title="Pacientes"),
                                alt.Tooltip("_pct:N", title="% dos pacientes")]))

    st.subheader("Exames por mês")
    dm = ef.groupby("periodo")["n"].sum().reindex([p for p in periodos if p_ini <= p <= p_fim], fill_value=0).reset_index()
    dm.columns = ["periodo", "exames"]
    dm["mes"] = dm["periodo"].map(rotulo)
    dm["_txt"] = dm["exames"].map(br)
    ordem_meses = dm["mes"].tolist()
    linha = alt.Chart(dm).encode(
        x=alt.X("mes:N", sort=ordem_meses, title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("exames:Q", title="Exames realizados", axis=alt.Axis(labelExpr=NUM_BR)),
        tooltip=[alt.Tooltip("mes:N", title="Mês"), alt.Tooltip("_txt:N", title="Exames")],
    )
    mostrar(linha.mark_line(color=COR, strokeWidth=2) + linha.mark_point(color=COR, filled=True, size=60))

    col_c, col_d = st.columns([3, 2])
    with col_c:
        st.subheader("Exames mais solicitados")
        dt = ef.groupby("des_procedimento")["n"].sum().nlargest(10).reset_index()
        dt.columns = ["procedimento", "exames"]
        dt["curto"] = dt["procedimento"].str.split(r" - | \(|,", regex=True).str[0].str.strip().str.capitalize()
        mostrar(barras(dt, "curto", "exames", "Exames", altura_barra=16,
                       tooltip=[alt.Tooltip("procedimento:N", title="Procedimento"), alt.Tooltip("_rotulo:N", title="Exames")]))
    with col_d:
        st.subheader("Exames por categoria")
        dg = ef.groupby("categoria_condicao")["n"].sum().sort_values(ascending=False).reset_index()
        dg.columns = ["Categoria", "Exames"]
        dg["%"] = (dg["Exames"] / dg["Exames"].sum() * 100).map(lambda v: f"{br(v, 1)}%")
        dg["Exames"] = dg["Exames"].map(br)
        st.dataframe(dg, hide_index=True, width="stretch")
        st.caption("“Outros/rotina” são exames sem relação com as seis condições (ex.: hemograma, urina).")

# ------------------------------------------------------ pacientes prioritários
with aba_prior:
    st.subheader("Pacientes ordenados pela probabilidade de multicondição")
    st.caption("Estimativa estatística do modelo Random Forest, para apoiar a priorização do acompanhamento. "
               "Não é diagnóstico: o modelo erra em cerca de 9% dos casos e a decisão é da equipe de saúde.")
    limite = st.slider("Probabilidade mínima", 0, 100, 80, step=5, format="%d%%")
    lista = pf[pf["probabilidade_multicondicao"] * 100 >= limite].sort_values(
        ["probabilidade_multicondicao", "n_exames_total"], ascending=False)
    st.write(f"**{br(len(lista))}** pacientes com probabilidade de {limite}% ou mais.")
    tabela = pd.DataFrame({
        "Código": lista["paciente_id_anon"],
        "Faixa etária": lista["faixa_etaria"],
        "Perfil": lista["perfil_utilizacao"],
        "Exames": lista["n_exames_total"],
        "Meses": lista["meses_acompanhamento"],
        "Categorias": lista["n_condicoes_distintas"],
        "Probabilidade": lista["probabilidade_multicondicao"] * 100,
    })
    st.dataframe(tabela.head(500), hide_index=True, width="stretch", height=520,
                 column_config={"Probabilidade": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)})
    if len(tabela) > 500:
        st.caption("Mostrando os 500 primeiros. Use os filtros para refinar.")

# ------------------------------------------------- sinistralidade (simulação)
with aba_sin:
    st.subheader("Simulação de sinistralidade dos exames")
    st.warning("**Simulação com parâmetros ajustáveis.** A despesa considera **só os exames** desta base "
               "(sem consultas, internações e terapias), a partir da tabela de valores por procedimento da "
               "operadora. A receita usa a tabela de mensalidades informada abaixo. O índice **não é** a "
               "sinistralidade da operadora: mostra quanto da receita é consumido pelos exames.")
    col_p1, col_p2 = st.columns([1, 1])
    with col_p1:
        pct_pago = st.slider("Percentual pago sobre o valor de tabela", 10, 100, 50, step=5, format="%d%%",
                             help="Quanto a operadora efetivamente paga ao prestador em relação à tabela.") / 100
        modo_meses = st.radio("Meses de receita por paciente",
                              ["Todos os meses do período selecionado", "Só os meses entre o 1º e o último exame"],
                              help="A base só mostra quem fez exames; ela não informa por quanto tempo cada um esteve no plano.")
    with col_p2:
        st.markdown("**Mensalidade por faixa etária (ANS)**")
        tab_mens = st.data_editor(
            pd.DataFrame({"Faixa": FAIXAS_ANS, "Mensalidade (R$)": MENSALIDADE_EXEMPLO}),
            hide_index=True, width="stretch", disabled=["Faixa"], key="mensalidades",
            column_config={"Mensalidade (R$)": st.column_config.NumberColumn(min_value=0.0, step=0.01, format="R$ %.2f")})
        st.caption("Valores de exemplo: tabela pública da Unimed Natal (Unimed Ser Empresarial, enfermaria). "
                   "Edite com os valores da Unimed.")

    mens = dict(zip(tab_mens["Faixa"], tab_mens["Mensalidade (R$)"].fillna(0)))
    meses_periodo = len([p for p in periodos if p_ini <= p <= p_fim])
    ps = pf.copy()
    ps["mensalidade"] = ps["faixa_ans"].map(mens).fillna(0)
    ps["meses_receita"] = meses_periodo if modo_meses.startswith("Todos") else ps["meses_acompanhamento"].clip(upper=meses_periodo)
    ps["receita"] = ps["mensalidade"] * ps["meses_receita"]

    custo = ef["valor_tabela"].sum() * pct_pago
    receita = ps["receita"].sum()
    indice = custo / receita if receita else 0
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Despesa estimada com exames", reais(custo))
    k2.metric("Receita estimada", reais(receita))
    k3.metric("Índice (despesa ÷ receita)", f"{br(indice * 100, 1)}%")
    k4.metric("Despesa média por paciente", reais(custo / len(ps)))
    st.caption(f"Período: {rotulo(p_ini)} a {rotulo(p_fim)} ({meses_periodo} meses) · "
               f"{br(info.get('cobertura_tuss', 0) * 100, 1)}% dos exames têm valor na tabela de procedimentos.")

    # índice por perfil e por faixa ANS (custo do período completo por paciente)
    fator_periodo = ef["valor_tabela"].sum() / max(pf["valor_tabela_total"].sum(), 1)
    ps["despesa"] = ps["valor_tabela_total"] * pct_pago * fator_periodo
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("**Índice por perfil**")
        g = ps.groupby("perfil_utilizacao")[["despesa", "receita"]].sum().reindex(PERFIS).dropna().reset_index()
        g["indice"] = g["despesa"] / g["receita"]
        mostrar(barras(g.rename(columns={"perfil_utilizacao": "perfil"}), "perfil", "indice", "Despesa ÷ receita",
                       ordem=PERFIS, cores=COR_PERFIL, pct=True, altura_barra=28))
    with col_s2:
        st.markdown("**Índice por faixa etária (ANS)**")
        g2 = ps.groupby("faixa_ans")[["despesa", "receita"]].sum().reindex(FAIXAS_ANS).dropna().reset_index()
        g2["indice"] = g2["despesa"] / g2["receita"].replace(0, float("nan"))
        mostrar(barras(g2.dropna(), "faixa_ans", "indice", "Despesa ÷ receita", ordem=FAIXAS_ANS, pct=True, altura_barra=16))

    st.markdown("**Índice mês a mês**")
    receita_mes = ps["mensalidade"].sum()
    dm2 = ef.groupby("periodo")["valor_tabela"].sum().reindex(
        [p for p in periodos if p_ini <= p <= p_fim], fill_value=0).reset_index()
    dm2.columns = ["periodo", "valor"]
    dm2["indice"] = dm2["valor"] * pct_pago / receita_mes if receita_mes else 0
    dm2["mes"] = dm2["periodo"].map(rotulo)
    dm2["_txt"] = dm2["indice"].map(lambda v: f"{br(v * 100, 1)}%")
    dm2["_desp"] = (dm2["valor"] * pct_pago).map(reais)
    lin = alt.Chart(dm2).encode(
        x=alt.X("mes:N", sort=dm2["mes"].tolist(), title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("indice:Q", title="Despesa com exames ÷ receita do mês", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("mes:N", title="Mês"), alt.Tooltip("_txt:N", title="Índice"), alt.Tooltip("_desp:N", title="Despesa")],
    )
    mostrar(lin.mark_line(color=COR, strokeWidth=2) + lin.mark_point(color=COR, filled=True, size=60))
    st.caption("No mês a mês, a receita considera todos os pacientes filtrados como ativos em todos os meses.")

    st.markdown("**Pacientes com maior despesa estimada**")
    top = ps.sort_values("despesa", ascending=False).head(200)
    st.dataframe(pd.DataFrame({
        "Código": top["paciente_id_anon"], "Faixa ANS": top["faixa_ans"], "Perfil": top["perfil_utilizacao"],
        "Exames": top["n_exames_total"], "Despesa estimada": top["despesa"].map(reais),
        "Receita estimada": top["receita"].map(reais),
        "Índice": (top["despesa"] / top["receita"].replace(0, float("nan"))).map(lambda v: f"{br(v * 100, 0)}%" if v == v else "—"),
    }), hide_index=True, width="stretch", height=420)

# ---------------------------------------------------------- sobre o modelo
with aba_modelo:
    rf, lr, mc = info["random_forest"], info["regressao_logistica"], info["matriz_confusao"]
    st.subheader(f"Desempenho no conjunto de teste ({br(info['pacientes_teste'])} pacientes não usados no treino)")
    f3 = lambda v: "—" if v is None else br(v, 3)
    st.dataframe(pd.DataFrame([
        {"Modelo": "Regressão Logística", "Precisão": f3(lr["precisao"]), "Recall": f3(lr["recall"]), "F1": f3(lr["f1"]), "AUC": f3(lr["auc"])},
        {"Modelo": "Random Forest (escolhido)", "Precisão": f3(rf["precisao"]), "Recall": f3(rf["recall"]), "F1": f3(rf["f1"]), "AUC": f3(rf["auc"])},
        {"Modelo": "Referência: só o total de exames", "Precisão": "—", "Recall": "—", "F1": "—", "AUC": f3(info["auc_so_n_exames"])},
    ]), hide_index=True, width="stretch")

    col_e, col_f = st.columns(2)
    with col_e:
        st.markdown("**Matriz de confusão (Random Forest)**")
        st.dataframe(pd.DataFrame({
            "": ["Real: não multicondição", "Real: multicondição"],
            "Previsto: não": [mc["VN"], mc["FN"]],
            "Previsto: sim": [mc["FP"], mc["VP"]],
        }), hide_index=True, width="stretch")
        erros = mc["FP"] + mc["FN"]
        total = erros + mc["VN"] + mc["VP"]
        st.caption(f"{erros} erros em {br(total)} pacientes ({br(erros / total * 100, 1)}%): "
                   f"{mc['FN']} multicondição não encontrados e {mc['FP']} alertas falsos.")
    with col_f:
        st.markdown("**Importância das variáveis**")
        di = pd.DataFrame({"variavel": list(info["importancias"]), "importancia": list(info["importancias"].values())})
        mostrar(barras(di, "variavel", "importancia", "Importância relativa", pct=True, altura_barra=18,
                       tooltip=[alt.Tooltip("variavel:N", title="Variável"), alt.Tooltip("_rotulo:N", title="Importância")]))

    st.markdown(
        "**Como ler.** O modelo estima a chance de o paciente ter exames de duas ou mais condições crônicas usando só o "
        "padrão de utilização (idade, total de exames, exames por mês e meses de acompanhamento). Como quem faz mais "
        "exames tende a fazer exames de mais condições, o total de exames sozinho já separa bem os grupos "
        f"(AUC {br(info['auc_so_n_exames'], 3)}). Por isso o modelo funciona como ordenador de prioridade. "
        "Na validação temporal (treino com 2024, teste com 2025), ele identificou 81% dos pacientes que passaram "
        "a ter exames de uma condição nova em 2025 (AUC 0,77)."
    )
    sil = info["silhueta"]
    st.caption("Segmentação K-Means: coeficiente de silhueta por número de grupos — "
               + " · ".join(f"k={k}: {br(v, 3)}" for k, v in sil.items()) + ". Escolhido k=3.")

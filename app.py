import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="ML Saúde - Câncer de Mama",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title(" Sistema Inteligente de Diagnóstico - Câncer de Mama")

st.markdown("""
###  Contexto Médico
O câncer de mama é o segundo tipo de câncer mais comum entre as mulheres no mundo, representando cerca de 25% de todos os casos de câncer feminino. 
O diagnóstico precoce é fundamental, pois pode aumentar as chances de cura em até 95% quando detectado em estágios iniciais.

###  Nossa Solução
Este sistema utiliza Inteligência Artificial para:
- Analisar características de tumores mamários
- Predizer diagnósticos com alta precisão
- Agrupar pacientes com perfis similares para tratamento personalizado
- Auxiliar médicos na tomada de decisões clínicas

###  Sobre os Dados
Utilizamos o dataset Wisconsin Breast Cancer, que contém:
- 569 casos de pacientes reais
- 30 características morfológicas dos tumores
- Diagnósticos confirmados por biópsia (Maligno/Benigno)
""")

@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['Diagnóstico'] = data.target
    df['Diagnóstico_Label'] = df['Diagnóstico'].map({0: 'Maligno', 1: 'Benigno'})
    
    # Dicionário de tradução das características
    traducao_colunas = {
        'mean radius': 'Raio Médio (mm)',
        'mean texture': 'Textura Média',
        'mean perimeter': 'Perímetro Médio (mm)',
        'mean area': 'Área Média (mm²)',
        'mean smoothness': 'Suavidade Média',
        'mean compactness': 'Compacidade Média',
        'mean concavity': 'Concavidade Média',
        'mean concave points': 'Pontos Côncavos Médios',
        'mean symmetry': 'Simetria Média',
        'mean fractal dimension': 'Dimensão Fractal Média',
        'radius error': 'Erro do Raio',
        'texture error': 'Erro da Textura',
        'perimeter error': 'Erro do Perímetro',
        'area error': 'Erro da Área',
        'smoothness error': 'Erro da Suavidade',
        'compactness error': 'Erro da Compacidade',
        'concavity error': 'Erro da Concavidade',
        'concave points error': 'Erro dos Pontos Côncavos',
        'symmetry error': 'Erro da Simetria',
        'fractal dimension error': 'Erro da Dimensão Fractal',
        'worst radius': 'Pior Raio (mm)',
        'worst texture': 'Pior Textura',
        'worst perimeter': 'Pior Perímetro (mm)',
        'worst area': 'Pior Área (mm²)',
        'worst smoothness': 'Pior Suavidade',
        'worst compactness': 'Pior Compacidade',
        'worst concavity': 'Pior Concavidade',
        'worst concave points': 'Piores Pontos Côncavos',
        'worst symmetry': 'Pior Simetria',
        'worst fractal dimension': 'Pior Dimensão Fractal'
    }
    
    df_traduzido = df.copy()
    df_traduzido = df_traduzido.rename(columns=traducao_colunas)
    
    return df, data, df_traduzido, traducao_colunas

df, data, df_traduzido, traducao_colunas = load_data()

st.markdown("---")
st.header(" 1. Visão Geral dos Dados")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total de Pacientes", len(df))
with col2:
    st.metric("Casos Benignos", len(df[df['Diagnóstico'] == 1]))
with col3:
    st.metric("Casos Malignos", len(df[df['Diagnóstico'] == 0]))
with col4:
    st.metric("Características", len(df.columns) - 2)

st.subheader(" Amostra dos Dados")

col1, col2 = st.columns([3, 1])
with col2:
    mostrar_traducao = st.checkbox("Mostrar em Português", value=True)

if mostrar_traducao:
    st.dataframe(df_traduzido.head(10))
    
else:
    st.dataframe(df.head(10))
    st.info("Ative 'Mostrar em Português' para ver os dados traduzidos e o glossário médico.")

st.subheader(" Distribuição de Diagnósticos")
fig_dist = px.pie(
    values=df['Diagnóstico_Label'].value_counts().values,
    names=df['Diagnóstico_Label'].value_counts().index,
    title="Distribuição de Diagnósticos",
    color_discrete_map={'Benigno': '#2E8B57', 'Maligno': '#DC143C'}
)
st.plotly_chart(fig_dist, use_container_width=True)

st.markdown("---")
st.header("2. Análise Exploratória dos Dados (EDA)")

st.subheader("Matriz de Correlação")
features_principais = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
corr_matrix = df[features_principais + ['Diagnóstico']].corr()
fig_corr = px.imshow(
    corr_matrix,
    text_auto=True,
    aspect="auto",
    title="Correlação entre Características Principais",
    color_continuous_scale='RdBu_r'
)
st.plotly_chart(fig_corr, use_container_width=True)

st.info("""
Insights da Correlação:
- Características de tamanho (radius, perimeter, area) são altamente correlacionadas
- Tumores malignos tendem a ter maior correlação com características de tamanho
- A textura mostra correlação moderada com o diagnóstico
""")

st.subheader("Controles Interativos")
features_traduzidas = [traducao_colunas[f] for f in features_principais]

feature_traduzida = st.selectbox("Escolha uma característica para analisar:", features_traduzidas)
feature_selecionada = [k for k, v in traducao_colunas.items() if v == feature_traduzida][0]

st.subheader(f"Distribuição: {feature_traduzida}")

explicacoes_clinicas = {
    'Raio Médio (mm)': 'Mede o tamanho do tumor. Tumores maiores podem indicar maior agressividade.',
    'Textura Média': 'Avalia a rugosidade da superfície. Texturas irregulares são mais comuns em tumores malignos.',
    'Perímetro Médio (mm)': 'Contorno do tumor. Perímetros maiores geralmente indicam tumores maiores.',
    'Área Média (mm²)': 'Espaço ocupado pelo tumor. Áreas maiores podem sugerir tumores mais avançados.',
    'Suavidade Média': 'Regularidade da superfície. Tumores malignos tendem a ser menos suaves.'
}

if feature_traduzida in explicacoes_clinicas:
    st.info(f"**O que isso significa:** {explicacoes_clinicas[feature_traduzida]}")

col1, col2 = st.columns(2)

with col1:
    fig_hist = px.histogram(
        df, x=feature_selecionada, color='Diagnóstico_Label',
        title=f"Distribuição - {feature_traduzida}",
        nbins=30,
        color_discrete_map={'Benigno': '#2E8B57', 'Maligno': '#DC143C'}
    )
    fig_hist.update_xaxes(title=feature_traduzida)
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    fig_box = px.box(
        df, x='Diagnóstico_Label', y=feature_selecionada,
        title=f"Comparação por Diagnóstico - {feature_traduzida}",
        color='Diagnóstico_Label',
        color_discrete_map={'Benigno': '#2E8B57', 'Maligno': '#DC143C'}
    )
    fig_box.update_yaxes(title=feature_traduzida)
    st.plotly_chart(fig_box, use_container_width=True)

st.subheader("Resumo dos Números por Diagnóstico")

st.info("""
**Como ler esta tabela:**
- **count**: Quantos pacientes temos de cada tipo
- **mean**: Valor médio (número típico)
- **std**: O quanto os valores variam (se é muito diferente entre pacientes)
- **min**: O menor valor encontrado
- **25%**: 25% dos pacientes têm valores menores que este
- **50%**: Valor do meio (metade acima, metade abaixo)
- **75%**: 75% dos pacientes têm valores menores que este
- **max**: O maior valor encontrado
""")

stats_comparison = df.groupby('Diagnóstico_Label')[feature_selecionada].describe()
st.dataframe(stats_comparison)

st.subheader("Casos Atípicos")
Q1 = df[feature_selecionada].quantile(0.25)
Q3 = df[feature_selecionada].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df[feature_selecionada] < Q1 - 1.5*IQR) | (df[feature_selecionada] > Q3 + 1.5*IQR)]

col1, col2 = st.columns(2)
with col1:
    st.metric("Casos Atípicos Encontrados", len(outliers))
with col2:
    st.metric("% de Casos Atípicos", f"{len(outliers)/len(df)*100:.1f}%")

st.success("""
**Principais Descobertas:**
- Tumores malignos são geralmente maiores
- Tumores malignos têm superfície mais irregular
- Há casos que se misturam, por isso precisamos de inteligência artificial
- Alguns casos são muito diferentes da maioria e merecem atenção especial
""")

st.markdown("---")
st.header("3. Modelos de Inteligência Artificial")

st.subheader("Configurações dos Modelos")
test_size = st.slider("Tamanho do conjunto de teste", 0.1, 0.4, 0.2, 0.05)

X = df.drop(['Diagnóstico', 'Diagnóstico_Label'], axis=1)
y = df['Diagnóstico']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

st.markdown("""
### Objetivo
Criar sistemas inteligentes que ajudem médicos a identificar se um tumor é perigoso (maligno) ou não (benigno), 
analisando as características do tumor.
""")

st.subheader("Treinamento e Comparação dos Sistemas Inteligentes")

dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

def calcular_metricas(y_true, y_pred, model_name):
    return {
        'Modelo': model_name,
        'Acurácia': accuracy_score(y_true, y_pred),
        'Precisão': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred)
    }

metricas_dt = calcular_metricas(y_test, dt_pred, 'Árvore de Decisão')
metricas_rf = calcular_metricas(y_test, rf_pred, 'Random Forest')

df_metricas = pd.DataFrame([metricas_dt, metricas_rf])

st.subheader("Qual Sistema é Melhor?")

col1, col2 = st.columns(2)

with col1:
    st.dataframe(df_metricas.round(3))

with col2:
    fig_comp = px.bar(
        df_metricas.melt(id_vars='Modelo', var_name='Métrica', value_name='Valor'),
        x='Métrica', y='Valor', color='Modelo',
        title="Comparação de Métricas",
        barmode='group'
    )
    st.plotly_chart(fig_comp, use_container_width=True)

st.subheader("Vantagens de Cada Sistema")

col1, col2 = st.columns(2)

with col1:
    st.info("""
    **Sistema de Decisão Simples**
    - **Fácil de entender**: Como uma árvore de perguntas
    - **Rápido**: Funciona muito rápido
    - **Clínico**: Médicos conseguem seguir o raciocínio
    - **Problema**: Pode "decorar" demais os exemplos
    """)

with col2:
    st.success("""
    **Sistema Inteligente Avançado**
    - **Mais confiável**: Não "decora" tanto os exemplos
    - **Mais preciso**: Acerta mais casos
    - **Mais estável**: Não se confunde com casos estranhos
    - **Problema**: Mais difícil de entender como funciona
    """)

melhor_modelo = 'Random Forest' if metricas_rf['Acurácia'] > metricas_dt['Acurácia'] else 'Árvore de Decisão'
modelo_final = rf_model if melhor_modelo == 'Random Forest' else dt_model
pred_final = rf_pred if melhor_modelo == 'Random Forest' else dt_pred

st.subheader(f"Sistema Escolhido: {melhor_modelo}")

cm = confusion_matrix(y_test, pred_final)
fig_cm = px.imshow(
    cm, text_auto=True,
    labels=dict(x="Predito", y="Real", color="Quantidade"),
    x=['Maligno', 'Benigno'], y=['Maligno', 'Benigno'],
    title=f"Acertos e Erros - {melhor_modelo}",
    color_continuous_scale='Blues'
)
st.plotly_chart(fig_cm, use_container_width=True)

st.subheader("Detalhes dos Resultados")
report = classification_report(y_test, pred_final, target_names=['Maligno', 'Benigno'], output_dict=True)
df_report = pd.DataFrame(report).transpose()
st.dataframe(df_report.round(3))

st.subheader("Teste de Confiabilidade")
cv_scores = cross_val_score(modelo_final, X, y, cv=5, scoring='accuracy')

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Precisão Média", f"{cv_scores.mean():.3f}")
with col2:
    st.metric("Variação", f"{cv_scores.std():.3f}")
with col3:
    st.metric("Faixa de Confiança", f"{cv_scores.mean():.3f} ± {1.96*cv_scores.std():.3f}")

if melhor_modelo == 'Random Forest':
    st.subheader("Quais Características São Mais Importantes")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importância': modelo_final.feature_importances_
    }).sort_values('Importância', ascending=False).head(10)
    
    fig_imp = px.bar(
        feature_importance, x='Importância', y='Feature',
        title="10 Características Mais Importantes para o Diagnóstico",
        orientation='h'
    )
    st.plotly_chart(fig_imp, use_container_width=True)

st.success(f"""
**Resumo dos Resultados:**
- O sistema {melhor_modelo} acertou {metricas_rf['Acurácia'] if melhor_modelo == 'Random Forest' else metricas_dt['Acurácia']:.1%} dos casos
- É importante não perder nenhum caso de câncer
- Também é importante não assustar pacientes sem necessidade
- Este sistema pode ajudar médicos a fazer diagnósticos mais rápidos
""")

st.markdown("---")
st.header("4. Grupos de Pacientes Similares")

st.subheader("Configurações dos Grupos")
n_clusters = st.slider("Número de Grupos", 2, 6, 3)

st.markdown("""
### Para que serve agrupar pacientes?
Encontrar grupos de pacientes parecidos para:
- Dar tratamentos mais adequados para cada grupo
- Descobrir tipos de tumores que não conhecíamos
- Organizar melhor o atendimento médico
""")

X = df.drop(['Diagnóstico', 'Diagnóstico_Label'], axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.subheader("Encontrando o Número Ideal de Grupos")

inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_scaled)
    inertias.append(kmeans_temp.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans_temp.labels_))

col1, col2 = st.columns(2)

with col1:
    fig_elbow = px.line(
        x=list(k_range), y=inertias,
        title="Qual o melhor número de grupos?",
        labels={'x': 'Número de Grupos', 'y': 'Qualidade do Agrupamento'}
    )
    fig_elbow.add_vline(x=3, line_dash="dash", line_color="red", annotation_text="K Ótimo")
    st.plotly_chart(fig_elbow, use_container_width=True)

with col2:
    fig_sil = px.line(
        x=list(k_range), y=silhouette_scores,
        title="Qualidade dos Grupos",
        labels={'x': 'Número de Grupos', 'y': 'Nota da Qualidade'}
    )
    best_k = k_range[np.argmax(silhouette_scores)]
    fig_sil.add_vline(x=best_k, line_dash="dash", line_color="green", annotation_text=f"Melhor K={best_k}")
    st.plotly_chart(fig_sil, use_container_width=True)

st.info(f"""
**Por que escolhemos {n_clusters} grupos?**
- O gráfico mostra que 3 é um bom número
- A qualidade é melhor com {best_k} grupos (nota: {max(silhouette_scores):.3f})
- Na medicina, faz sentido ter grupos diferentes de pacientes
""")

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df_clustered = df.copy()
df_clustered['Cluster'] = clusters
df_clustered['Cluster_Label'] = df_clustered['Cluster'].map({i: f'Grupo {i+1}' for i in range(n_clusters)})

silhouette_avg = silhouette_score(X_scaled, clusters)
inertia = kmeans.inertia_

st.subheader("Qualidade dos Grupos Criados")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Nota da Qualidade", f"{silhouette_avg:.3f}")
with col2:
    st.metric("Organização", f"{inertia:.0f}")
with col3:
    st.metric("Número de Grupos", n_clusters)

st.subheader("Como os Grupos Ficaram Organizados")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'Cluster': df_clustered['Cluster_Label'],
    'Diagnóstico': df_clustered['Diagnóstico_Label']
})

col1, col2 = st.columns(2)

with col1:
    fig_pca_cluster = px.scatter(
        df_pca, x='PC1', y='PC2', color='Cluster',
        title="Grupos de Pacientes",
        labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} da variância)',
               'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} da variância)'}
    )
    st.plotly_chart(fig_pca_cluster, use_container_width=True)

with col2:
    fig_pca_diag = px.scatter(
        df_pca, x='PC1', y='PC2', color='Diagnóstico',
        title="Diagnósticos Reais dos Pacientes",
        color_discrete_map={'Benigno': '#2E8B57', 'Maligno': '#DC143C'},
        labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} da variância)',
               'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} da variância)'}
    )
    st.plotly_chart(fig_pca_diag, use_container_width=True)

st.info("""
**O que cada grupo representa:**

**Grupo 1**: Pacientes com tumores pequenos e regulares - Menor risco
**Grupo 2**: Pacientes com tumores médios - Risco moderado  
**Grupo 3**: Pacientes com tumores grandes e irregulares - Maior risco

**Como isso ajuda na medicina:**
- Cada grupo pode ter um tipo de acompanhamento diferente
- Focar mais recursos nos grupos de maior risco
- Identificar rapidamente pacientes que precisam de mais atenção
- Personalizar o tratamento para cada tipo de paciente
""")

st.subheader("Quantos Pacientes em Cada Grupo")

cluster_counts = df_clustered['Cluster_Label'].value_counts()
fig_dist = px.pie(
    values=cluster_counts.values,
    names=cluster_counts.index,
    title="Número de Pacientes por Grupo"
)
st.plotly_chart(fig_dist, use_container_width=True)

st.subheader("Características de Cada Grupo")

features_analise = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
cluster_profiles = df_clustered.groupby('Cluster_Label')[features_analise].mean()

fig_heatmap = px.imshow(
    cluster_profiles.T,
    labels=dict(x="Cluster", y="Característica", color="Valor Médio"),
    title="Características Médias de Cada Grupo",
    color_continuous_scale='Viridis'
)
st.plotly_chart(fig_heatmap, use_container_width=True)

st.dataframe(cluster_profiles.round(3))

st.subheader("Diagnósticos em Cada Grupo")

crosstab = pd.crosstab(df_clustered['Cluster_Label'], df_clustered['Diagnóstico_Label'], normalize='index')
fig_cross = px.bar(
    crosstab.reset_index().melt(id_vars='Cluster_Label'),
    x='Cluster_Label', y='value', color='Diagnóstico_Label',
    title="Porcentagem de Casos Malignos e Benignos por Grupo",
    labels={'value': 'Porcentagem', 'Cluster_Label': 'Grupo'},
    color_discrete_map={'Benigno': '#2E8B57', 'Maligno': '#DC143C'}
)
st.plotly_chart(fig_cross, use_container_width=True)


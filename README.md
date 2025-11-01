# ML Breast Cancer Diagnosis

Sistema de Machine Learning para classificação de câncer de mama usando o dataset Wisconsin Breast Cancer.

## Tecnologias

- Python 3.8+
- Streamlit
- Scikit-learn
- Pandas, NumPy
- Plotly

## Instalação

```bash
git clone https://github.com/binnahv/Projeto-saude.git
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

- **569 amostras** do Wisconsin Breast Cancer Dataset
- **30 features** morfológicas de tumores
- **2 classes**: Maligno (0) / Benigno (1)

## Funcionalidades

### 1. Análise Exploratória
- Correlação entre features
- Distribuições por diagnóstico
- Detecção de outliers

### 2. Classificação Supervisionada
- Decision Tree vs Random Forest
- Métricas: Accuracy, Precision, Recall, F1-Score
- Cross-validation
- Feature importance

### 3. Clustering Não-Supervisionado
- K-Means clustering
- Elbow method + Silhouette Score
- Visualização PCA

## Resultados

- **Accuracy**: ~95% (Random Forest)
- **Silhouette Score**: >0.5
- **3 clusters** identificados com perfis de risco distintos

## Limitações

⚠️ **Ferramenta de apoio** - não substitui diagnóstico médico profissional.

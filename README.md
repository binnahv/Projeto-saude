# Sistema Inteligente de Diagnóstico - Câncer de Mama

Sistema de Machine Learning para auxiliar no diagnóstico de câncer de mama utilizando o dataset Wisconsin Breast Cancer, desenvolvido com Streamlit.

## 📋 Sobre o Projeto

O câncer de mama é o segundo tipo de câncer mais comum entre as mulheres no mundo, representando cerca de 25% de todos os casos de câncer feminino. O diagnóstico precoce é fundamental, pois pode aumentar as chances de cura em até 95% quando detectado em estágios iniciais.

Este sistema utiliza Inteligência Artificial para:
- Analisar características de tumores mamários
- Predizer diagnósticos com alta precisão
- Agrupar pacientes com perfis similares para tratamento personalizado
- Auxiliar médicos na tomada de decisões clínicas

## 🎯 Funcionalidades

### 1. Visão Geral dos Dados
- Métricas gerais do dataset
- Distribuição de diagnósticos
- Amostra dos dados clínicos

### 2. Análise Exploratória (EDA)
- Matriz de correlação interativa
- Gráficos comparativos por diagnóstico
- Análise de outliers
- Insights clínicos relevantes

### 3. Modelos Preditivos (Aprendizagem Supervisionada)
- Comparação entre Árvore de Decisão e Random Forest
- Métricas completas: Acurácia, Precisão, Recall, F1-Score
- Validação cruzada
- Matriz de confusão
- Importância das características

### 4. Agrupamento de Pacientes (Aprendizagem Não Supervisionada)
- Clustering com K-Means
- Método do cotovelo para determinação do K ótimo
- Análise com Silhouette Score
- Visualização PCA dos clusters
- Interpretação clínica dos grupos

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+**
- **Streamlit** - Interface web interativa
- **Pandas** - Manipulação de dados
- **NumPy** - Computação numérica
- **Scikit-learn** - Machine Learning
- **Plotly** - Visualizações interativas
- **Matplotlib/Seaborn** - Gráficos estatísticos

## 📦 Instalação e Execução

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/projeto-ml-saude.git
cd projeto-ml-saude
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Execute a aplicação
```bash
streamlit run app.py
```

### 4. Acesse no navegador
```
http://localhost:8501
```

## 📊 Dataset

O projeto utiliza o **Wisconsin Breast Cancer Dataset**, que contém:
- **569 casos** de pacientes reais
- **30 características** morfológicas dos tumores
- **Diagnósticos confirmados** por biópsia (Maligno/Benigno)

### Características principais analisadas:
- Raio médio do tumor
- Textura média
- Perímetro médio
- Área média
- Suavidade média
- E outras 25 características derivadas

## 🔬 Metodologia

### Análise Exploratória
1. **Correlação entre variáveis** - Identificação de características mais discriminativas
2. **Distribuições por diagnóstico** - Análise de padrões entre casos malignos e benignos
3. **Detecção de outliers** - Identificação de casos atípicos

### Modelos de Machine Learning
1. **Árvore de Decisão**
   - Interpretabilidade alta
   - Regras clínicas claras
   - Rápido treinamento

2. **Random Forest**
   - Maior robustez
   - Menos propenso ao overfitting
   - Melhor precisão geral

### Clustering
1. **Preparação dos dados** - Padronização com StandardScaler
2. **Determinação do K** - Método do cotovelo + Silhouette Score
3. **Interpretação clínica** - Perfis de risco por cluster

## 📈 Resultados

### Modelos Supervisionados
- **Acurácia**: ~95% (Random Forest)
- **Precisão**: Alta para ambas as classes
- **Recall**: Otimizado para não perder casos malignos
- **F1-Score**: Balanceado entre precisão e recall

### Clustering
- **Silhouette Score**: >0.5 (boa separação)
- **Grupos identificados**: 3 perfis de risco distintos
- **Interpretação**: Baixo, moderado e alto risco

## 🏥 Aplicações Clínicas

### Para Médicos
- **Triagem inicial** de casos suspeitos
- **Segunda opinião** automatizada
- **Identificação de padrões** não óbvios

### Para Hospitais
- **Otimização de recursos** baseada em perfis de risco
- **Protocolos personalizados** por cluster de pacientes
- **Agilização do processo** diagnóstico

## ⚠️ Limitações e Considerações

- Este sistema é uma **ferramenta de apoio** e não substitui o julgamento clínico profissional
- Os resultados devem sempre ser **validados por especialistas**
- O modelo foi treinado em um dataset específico e pode não generalizar para todas as populações
- **Não deve ser usado** como única fonte para decisões médicas

## 🤝 Contribuições

Contribuições são bem-vindas! Para contribuir:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 👨‍💻 Autor

**Seu Nome**
- GitHub: [@seu-usuario](https://github.com/seu-usuario)
- LinkedIn: [Seu Perfil](https://linkedin.com/in/seu-perfil)
- Email: seu.email@exemplo.com

## 🙏 Agradecimentos

- Dataset fornecido pela **University of Wisconsin**
- Comunidade **Streamlit** pelas ferramentas incríveis
- Biblioteca **Scikit-learn** pela implementação robusta de ML
- **Plotly** pelas visualizações interativas

---

**Importante**: Este projeto foi desenvolvido para fins educacionais e de pesquisa. Sempre consulte profissionais de saúde qualificados para decisões médicas.

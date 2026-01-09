# 🛡️ Stats Weapons: Toolkit de Estatística Aplicada e Ciência de Dados

> *Um arsenal de funções Python para facilitar a ponte entre a teoria estatística e a prática de análise de dados.*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Educational%20%2F%20WIP-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Sobre o Projeto

Este repositório contém o módulo `stats_weapons.py`, uma biblioteca de utilitários que desenvolvi para auxiliar nas aulas de **Estatística Aplicada** que ministrei na minha empresa.

O objetivo era criar uma ferramenta "agóstica" que permitisse aos alunos (engenheiros e analistas) focar na interpretação dos testes estatísticos e na limpeza de dados, abstraindo a complexidade da sintaxe de bibliotecas como `scipy`, `statsmodels` e `scikit-learn` em funções reutilizáveis.

### 🧠 Contexto Educacional
O código foi utilizado para demonstrar conceitos práticos de:
* Análise Descritiva e EDA.
* Detecção e Tratamento de Outliers.
* Testes de Normalidade.
* Testes de Hipóteses (Paramétricos e Não-Paramétricos).

## 🛠️ Funcionalidades Implementadas

O toolkit atua como um wrapper para facilitar as seguintes tarefas:

### 1. ETL Robusto (`extract_data`)
Carregamento unificado de dados de diversas fontes, tratando automaticamente as extensões:
* Arquivos Flat: `.csv`, `.txt`, `.json`, `.xml`
* Planilhas: `.xlsx`, `.xls`
* Bancos de Dados: `.sql`, `.db`, `.sqlite`, `.parquet`
* Outros: `.h5`, `.pkl`

### 2. Limpeza e Qualidade de Dados
* **Outlier Detection:** Implementações de métodos estatísticos (Z-Score, IQR) e Machine Learning (Isolation Forest, LOF, OneClassSVM, DBSCAN).
* **Missing Values:** Tratamento automático de nulos e duplicatas.

### 3. Inferência Estatística
Automação de testes comuns com interpretação simplificada dos resultados (P-Valor):
* **Normalidade:** Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling.
* **Correlação:** Pearson, Spearman, Kendall.
* **Comparação de Médias/Medianas:** T-Test, ANOVA, Mann-Whitney, Kruskal-Wallis.

## 🚀 Como Usar

1. Clone o repositório:
   ```bash
   git clone [https://github.com/seu-usuario/stats-weapons.git](https://github.com/seu-usuario/stats-weapons.git)

## 🚧 Roadmap (Em Desenvolvimento)
Como este projeto é fruto de uma iniciativa de ensino contínuo, as seguintes áreas estão mapeadas para implementação futura (atualmente como placeholders no código):

[ ] Séries Temporais: Decomposição STL, Testes de Estacionariedade (ADF), Modelos ARIMA/Smoothing.

[ ] Análise de Confiabilidade: Curvas de sobrevivência, Weibull, MTBF/MTTR.

[ ] Engenharia de Atributos: Pipelines automáticos de transformação.

## 🤝 Contribuição
Sugestões e Pull Requests são bem-vindos! Se você é estudante de Data Science ou Engenharia, sinta-se à vontade para usar este código como base para seus estudos.

## ✍️ Autor
Renery Carvalho, Engenheiro Mecânico & Pós-Graduando em Estatística para Ciência de Dados

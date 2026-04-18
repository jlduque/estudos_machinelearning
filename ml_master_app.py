import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="🧠 ML Master - Estudo de Algoritmos", page_icon="🧠", layout="wide")

ALGORITHMS = [
    "Regressão Linear",
    "Regressão Logística",
    "Árvore de Decisão",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "KNN",
    "SVM",
    "Naive Bayes",
    "Redes Neurais",
    "K-Means",
    "DBSCAN",
    "PCA"
]

ALGORITHM_DATA = {
    "Regressão Linear": {
        "tipo": "Supervisionado",
        "tarefa": "Regressão",
        "parametrico": True,
        "instace_based": False,
        "escalas": True,
        "sensivel_outliers": "Alta",
        "interpretabilidade": "Alta",
        "performance": "Média",
        "complexidade_treino": "O(n)",
        "complexidade_inferencia": "O(1)",
        "proposito": "Prever valores contínuos com base em relações lineares entre variáveis",
        "estrutura": "Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε",
        "funcao_perda": "MSE (Mean Squared Error)",
        "regularizacao": ["L1 (Lasso)", "L2 (Ridge)"],
        "como_aprende": """
O algoritmo encontra os coeficientes (β) que minimizam a soma dos quadrados dos resíduos:
1. Inicializa coeficientes com valores zeros ou aleatórios
2. Calcula previsões usando a equação linear
3. Calcula o erro (diferença entre previsto e real)
4. Usa gradiente descendente para ajustar coeficientes
5. Repete até convergência ou número máximo de iterações
""",
        "como_prevê": "Multiplica cada feature pelo seu coeficiente e soma (produto escalar) + intercepto",
        "pipeline": """
1. Entrada: Features X e target y
2. Pré-processamento: 
   - Remover outliers extremos
   - Verificar linearidade
   - Padronizar features (se usar regularização)
3. Treinamento:
   - Escolher método (OLS, Gradiente Descendente, etc.)
   - Minimizar MSE
4. Ajuste de hiperparâmetros:
   - α (taxa de aprendizado)
   - λ (parâmetro de regularização)
5. Predição: y = X_test @ β + β₀
""",
        "overfitting": """
- Causas: Muitas features, multicolinearidade, poucos dados
- Soluções:
  * Regularização L1 (Lasso) → zera coeficientes irrelevantes
  * Regularização L2 (Ridge) → reduz coeficientes sem zerar
  * Seleção de features
  * Cross-validation
  * Diminuir complexidade do modelo
""",
        "pre_processamento": """
- Normalização/Padronização: NECESSÁRIA se usar regularização ou gradiente descendente
- Outliers: ALTA sensibilidade - remover ou tratar
- Variáveis categóricas: Codificar (One-Hot, Label Encoding)
- Multicolinearidade: Verificar VIF, remover features correlacionadas
- Linearidade: Verificar relação linear entre X e y
""",
        "metricas_regressao": """
- MSE (Mean Squared Error): Penaliza mais erros grandes
- RMSE: Raiz quadrada do MSE (mesma escala do target)
- MAE (Mean Absolute Error): Menos sensível a outliers
- R²: Proporção da variância explicada pelo modelo
""",
        "vies_variancia": """
- Alto viés: Underfitting (modelo muito simples)
- Alta variância: Overfitting (modelo muito complexo)
- Trade-off: Regularização equilibra viés-variância
- Gráfico: Curva em U onde viés diminui e variância aumenta com complexidade
""",
        "vantagens": [
            "Simples e interpretável",
            "Rápido treinamento e inferência",
            "Boa baseline para comparação",
            "Funciona bem com relações lineares",
            "Pouco risco de overfitting com regularização"
        ],
        "desvantagens": [
            "Assume linearidade",
            "Sensível a outliers",
            "Não captura relações não-lineares",
            "Requer feature engineering"
        ],
        "aplicacoes": [
            "Previsão de preços (imóveis, ações)",
            "Análise de tendência",
            "Baseline para modelos complexos",
            "Econometria e finanças"
        ]
    },
    
    "Regressão Logística": {
        "tipo": "Supervisionado",
        "tarefa": "Classificação",
        "parametrico": True,
        "instace_based": False,
        "escalas": True,
        "sensivel_outliers": "Média",
        "interpretabilidade": "Alta",
        "performance": "Média",
        "complexidade_treino": "O(n)",
        "complexidade_inferencia": "O(1)",
        "proposito": "Classificar dados em categorias (principalmente binária) usando função sigmoide",
        "estrutura": "P(y=1) = 1 / (1 + e^(-z)) onde z = β₀ + β₁X₁ + ...",
        "funcao_perda": "Log Loss (Binary Cross-Entropy)",
        "regularizacao": ["L1 (Lasso)", "L2 (Ridge)"],
        "como_aprende": """
1. Calcula z = X @ β + β₀ (combinação linear)
2. Aplica função sigmoide: σ(z) = 1/(1+e^(-z))
3. Interpreta resultado como probabilidade
4. Calcula Log Loss: -(1/n) * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
5. Usa gradiente descendente para minimizar Log Loss
6. Ajusta coeficientes iterativamente
""",
        "como_prevê": """
1. Calcula z = X @ β + β₀
2. Aplica sigmoide → probabilidade entre 0 e 1
3. Se P > limiar (default 0.5): classe 1, senão: classe 0
""",
        "pipeline": """
1. Entrada: Features X e target binário y
2. Pré-processamento:
   - Codificar variáveis categóricas
   - Padronizar features numéricas
   - Tratar outliers moderadamente
3. Treinamento:
   - Gradiente descendente
   - Minimizar Log Loss
4. Ajuste de hiperparâmetros:
   - α (taxa de aprendizado)
   - λ (regularização)
   - Número de iterações
5. Predição: Classes ou probabilidades
""",
        "overfitting": """
- Causas: Muitas features, classes desbalanceadas
- Soluções:
  * Regularização L1/L2
  * Reduzir número de features
  * Class-weight para desbalanceamento
  * Cross-validation
""",
        "pre_processamento": """
- Escalonamento: NECESSÁRIO para convergência rápida
- Outliers: Sensibilidade MÉDIA
- Variáveis categóricas: One-Hot Encoding
- Features: Sem correlação forte entre si
- Classes: Verificar balanceamento
""",
        "metricas_classificacao": """
- Accuracy: Proporção de acertos
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1-Score: Harmônica de Precision e Recall
- ROC-AUC: Área sob curva ROC
- Log Loss: Probabilidades bem calibradas
""",
        "vies_variancia": """
- Alto viés: Fronteira de decisão muito simples
- Alta variância: Overfitting a ruídos
- Regularização controla complexidade
""",
        "vantagens": [
            "Interpretável (coeficientes indicam importância)",
            "Probabilidades de saída",
            "Rápido e eficiente",
            "Boa baseline",
            "Regularizável"
        ],
        "desvantagens": [
            "Apenas fronteiras lineares",
            "Sensível a classes desbalanceadas",
            "Não captura interações automaticamente",
            "Limitações com dados complexos"
        ],
        "aplicacoes": [
            "Detecção de spam",
            "Aprovação de crédito",
            "Diagnóstico médico binário",
            "Churn prediction"
        ]
    },
    
    "Árvore de Decisão": {
        "tipo": "Supervisionado",
        "tarefa": "Classificação e Regressão",
        "parametrico": False,
        "instace_based": False,
        "escalas": False,
        "sensivel_outliers": "Baixa",
        "interpretabilidade": "Alta",
        "performance": "Média",
        "complexidade_treino": "O(n log n)",
        "complexidade_inferencia": "O(log n)",
        "proposito": "Criar modelo de decisões em formato de árvore, dividindo dados por regras de decisão",
        "estrutura": "Nós de decisão conectados em estrutura hierárquica árvore",
        "funcao_perda": "Gini (classificação) ou MSE (regressão)",
        "regularizacao": ["Poda", "max_depth", "min_samples_leaf"],
        "como_aprende": """
Algoritmo ID3/CART usando Divisão Gulosa (Greedy):
1. Começa com todos os dados no nó raiz
2. Para cada feature, calcula impureza (Gini ou Entropia)
3. Escolhe a divisão que maximiza o ganho de informação
4. Cria dois (ou mais) nós filhos
5. Repete recursivamente para cada nó filho
6. Para quando critério de parada é atingido
""",
        "como_prevê": """
1. Começa na raiz
2. Para cada nó:
   - Verifica a condição de divisão
   - Segue para o nó filho correspondente
3. Quando chega numa folha:
   - Retorna a classe majoritária (classificação)
   - Retorna a média dos valores (regressão)
""",
        "pipeline": """
1. Entrada: Features X e target y
2. Pré-processamento:
   - Não requer escalonamento
   - Não é afetado por outliers
   - Codificar categóricas (ordinal ou one-hot)
3. Treinamento:
   - Divisão recursiva greedy
   - Critério: Gini, Entropia ou MSE
4. Ajuste de hiperparâmetros:
   - max_depth: profundidade máxima
   - min_samples_split: mínimo amostras para dividir
   - min_samples_leaf: mínimo amostras por folha
   - criterion: gini, entropy ou mse
5. Predição: Percorre árvore até folha
""",
        "overfitting": """
- Causas: Árvore muito profunda, poucos dados por folha
- Soluções:
  * Poda (pre ou post-pruning)
  * Limitar profundidade (max_depth)
  * Mínimo de amostras por folha
  * Mínimo de amostras para dividir
  * Cross-validation para escolher hiperparâmetros
""",
        "pre_processamento": """
- Escalonamento: NÃO NECESSÁRIO
- Outliers: Baixa sensibilidade (divisões por ranges)
- Categóricas: Suporta nativamente (ordinal encoding)
- Missing values: Suporta com estratégias específicas
- Features: Importantes para qualidade das divisões
""",
        "metricas_classificacao": """
- Gini Impurity: 1 - Σ(pᵢ)²
- Entropia: -Σ(pᵢ * log₂(pᵢ))
- Ganho de Informação: Entropia(pai) - Σ(wᵢ * Entropia(filho))
""",
        "vies_variancia": """
- Alto viés: Árvore muito rasa
- Alta variância: Árvore muito profunda
- Variance alto: Sensível a pequenas mudanças nos dados
- Ensemble (Random Forest) reduz variância
""",
        "vantagens": [
            "Muito interpretável (visualização clara)",
            "Não precisa escalonamento",
            "Lida com categóricas nativamente",
            "Pouco sensível a outliers",
            "Captura relações não-lineares"
        ],
        "desvantagens": [
            "Alta variância (instável)",
            "Overfitting fácil",
            "Divisão greedy pode não ser ótima globalmente",
            "Fronteiras de decisão axis-aligned"
        ],
        "aplicacoes": [
            "Sistemas de recomendação",
            "Diagnóstico médico",
            "Aprovação de crédito",
            "Classificação de documentos"
        ]
    },
    
    "Random Forest": {
        "tipo": "Supervisionado",
        "tarefa": "Classificação e Regressão",
        "parametrico": False,
        "instace_based": False,
        "escalas": False,
        "sensivel_outliers": "Baixa",
        "interpretabilidade": "Média",
        "performance": "Alta",
        "complexidade_treino": "O(k * n log n)",
        "complexidade_inferencia": "O(k * log n)",
        "proposito": "Ensemble de árvores de decisão que usa bagging e randomização de features para reduzir variância",
        "estrutura": "k árvores independentes, cada uma treinada com Bootstrap + Random Subspace",
        "funcao_perda": "Gini ou MSE (por árvore)",
        "regularizacao": ["max_depth", "min_samples", "n_estimators"],
        "como_aprende": """
Bagging + Random Subspace:
1. Para cada uma das k árvores:
   a) Bootstrap sampling: seleciona n amostras com reposição
   b) Random subspace: seleciona m features aleatórias
   c) Treina árvore de decisão com as m features
2. Agrega previsões:
   - Classificação: Votação majoritária
   - Regressão: Média das previsões
""",
        "como_prevê": """
1. Cada árvore faz sua previsão
2. Classificação: Votação majoritária (moda)
3. Regressão: Média aritmética
""",
        "pipeline": """
1. Entrada: Features X e target y
2. Pré-processamento:
   - Não requer escalonamento
   - Trata outliers automaticamente
   - Codificar categóricas
3. Treinamento:
   - Bootstrap em cada árvore
   - Random subspace de features
   - Treinar k árvores independentes
4. Ajuste de hiperparâmetros:
   - n_estimators: número de árvores
   - max_features: features por split
   - max_depth: profundidade máxima
   - min_samples_split/leaf
5. Predição: Ensemble de árvores
""",
        "overfitting": """
- Overfitting REDUZIDO comparado a árvore única
- Ainda pode overfitar com:
  * Muitas árvores com alta profundidade
  * Dados muito ruidosos
- Soluções:
  * Limitar max_depth
  * Aumentar min_samples_leaf
  * Cross-validation
  * Regularização por hiperparâmetros
""",
        "pre_processamento": """
- Escalonamento: NÃO NECESSÁRIO
- Outliers: Baixa sensibilidade
- Categóricas: Suporta (nativas ou encoded)
- Features: m = √p geralmente funciona bem
- Importância: Calcula feature importance
""",
        "metricas": """
- Importância de Features: Gini importance ou Permutation importance
- OOB Error: Erro em amostras fora do bootstrap
- Out-of-Bag Score para validação
""",
        "vies_variancia": """
- Variância: REDUZIDA (benefício do bagging)
- Viés: Similar a árvore única
- Resultado: Melhor trade-off viés-variância
- Decomposição: Var(Y) = ρ * σ² + (1-ρ)/k * σ²
""",
        "vantagens": [
            "Reduz variância significativamente",
            "Paralelizável",
            "Lida com missing values",
            "Menos overfitting que árvores individuais",
            "Importância de features integrada",
            "Robusto a outliers"
        ],
        "desvantagens": [
            "Menos interpretável que árvore única",
            "Mais computacional que árvore única",
            "Pode ser lento com muitas árvores",
            "Ensemble complexo"
        ],
        "aplicacoes": [
            "Kaggle competitions",
            "Detecção de fraude",
            "Classificação de imagens",
            "Feature selection"
        ]
    },
    
    "Gradient Boosting": {
        "tipo": "Supervisionado",
        "tarefa": "Classificação e Regressão",
        "parametrico": False,
        "instace_based": False,
        "escalas": False,
        "sensivel_outliers": "Média",
        "interpretabilidade": "Baixa",
        "performance": "Muito alta",
        "complexidade_treino": "O(k * n log n)",
        "complexidade_inferencia": "O(k)",
        "proposito": "Ensemble sequencial que treina árvores para corrigir erros das anteriores (boosting)",
        "estrutura": "Sequência de árvores, cada uma corrigindo resíduos das anteriores",
        "funcao_perda": "Log Loss (classificação) ou MSE (regressão)",
        "regularizacao": [" learning_rate", "n_estimators", "max_depth"],
        "como_aprende": """
Boosting Sequencial:
1. Inicializa com predição média (ou constante)
2. Para cada iteração t = 1, 2, ..., k:
   a) Calcula pseudo-resíduos: -∂L/∂ŷ
   b) Treina árvore para predizer pseudo-resíduos
   c) Calcula taxa de aprendizado (shrinkage)
   d) Atualiza modelo: F_t(x) = F_{t-1}(x) + η * h_t(x)
3. Resultado: Soma ponderada de árvores
""",
        "como_prevê": """
1. Soma as previsões de todas as árvores:
   ŷ = F_k(x) = F₀(x) + η * Σ h_t(x)
2. Para classificação: aplica link function (sigmoid ou softmax)
""",
        "pipeline": """
1. Entrada: Features X e target y
2. Pré-processamento:
   - Não requer escalonamento obrigatório
   - Lidar com outliers moderadamente
   - Codificar categóricas
3. Treinamento:
   - Função de perda escolhe tarefa
   - Gradiente descendente funcional
   - Cada árvore corrige erros anteriores
4. Ajuste de hiperparâmetros:
   - n_estimators: número de árvores
   - learning_rate: shrinkage (0.01-0.3)
   - max_depth: profundidade das árvores
   - subsample: amostragem de dados
5. Predição: Ensemble ponderado
""",
        "overfitting": """
- Risco MODERADO de overfitting
- Causas: Muitas árvores, alta complexidade
- Soluções:
  * Learning rate baixo + mais árvores
  * Limitar max_depth
  * Subsampling (Stochastic GB)
  * Early stopping
  * Regularização nos leaf nodes
  * Gradient-based one-side sampling (GOSS)
""",
        "pre_processamento": """
- Escalonamento: NÃO NECESSÁRIO
- Outliers: Sensibilidade MÉDIA (especialmente regressão)
- Categóricas: Encode necessário
- Features: Mistas numéricas e categóricas
- Missing: Suporta internamente
""",
        "metricas": """
- Regressão: MSE, MAE, Huber loss
- Classificação: Log loss, AUC
- Early stopping: Monitora em conjunto de validação
""",
        "vies_variancia": """
- Viés: REDUZIDO (modelo forte)
- Variância: Moderada (pode aumentar com complexidade)
- Boosting: Reduz viés iterativamente
- Early stopping previne overfitting
""",
        "vantagens": [
            "Performance state-of-the-art",
            "Lida com relações complexas",
            "Feature importance nativa",
            "Versátil (classificação/regressão)",
            "Poucos hiperparâmetros críticos"
        ],
        "desvantagens": [
            "Menos interpretável",
            "Treinamento sequencial (lento)",
            " حساسة excessiva a outliers",
            "Hyperparameter tuning importante",
            "Memória intensiva"
        ],
        "aplicacoes": [
            "Kaggle (vencedor de muitas competições)",
            "Ranking (learning to rank)",
            "Detecção de anomalias",
            "Previsão de séries temporais"
        ]
    },
    
    "XGBoost": {
        "tipo": "Supervisionado",
        "tarefa": "Classificação e Regressão",
        "parametrico": False,
        "instace_based": False,
        "escalas": False,
        "sensivel_outliers": "Média",
        "interpretabilidade": "Baixa",
        "performance": "Muito alta",
        "complexidade_treino": "O(k * n log n)",
        "complexidade_inferencia": "O(k)",
        "proposito": "Implementação otimizada de Gradient Boosting com regularização integrada e eficiência computacional",
        "estrutura": "Árvores com regularização (L1+L2) e second-order gradient",
        "funcao_perda": "Log Loss ou MSE com regularização",
        "regularizacao": ["L1", "L2", "lambda", "alpha"],
        "como_aprende": """
XGBoost Otimizado:
1. Usa second-order Taylor expansion (Hessiana)
2. Regularização integrada na função objetivo:
   Obj = Σ L(yᵢ, ŷᵢ) + Σ Ω(fₜ)
   onde Ω(f) = γT + ½λΣw² + α|w|
3. Tree pruning baseado em gain negativo
4. Divisão paralela por feature
5. Cache-aware access
""",
        "como_prevê": "Mesma lógica de Gradient Boosting: soma ponderada de árvores com regularização",
        "pipeline": """
1. Entrada: Features X e target y
2. Pré-processamento:
   - Pode trabalhar com dados esparsos
   - Criação de features (DMatrix)
   - Missing values tratados automaticamente
3. Treinamento:
   - Regularização dual (L1 + L2)
   - Second-order gradient
   - Column subsampling por árvore
   - Row subsampling (Stochastic GB)
4. Ajuste de hiperparâmetros:
   - learning_rate, n_estimators
   - max_depth, min_child_weight
   - subsample, colsample_bytree
   - gamma, reg_alpha, reg_lambda
5. Predição: Ensemble regularizado
""",
        "overfitting": """
- Controles de regularização:
  * lambda (L2) e alpha (L1)
  * max_depth: controla complexidade
  * min_child_weight: previne overfitting local
  * subsample, colsample: redução de variância
  * Early stopping com eval_set
""",
        "pre_processamento": """
- Escalonamento: NÃO NECESSÁRIO
- Outliers: Melhor tratamento via regularização
- Missing: Automático
- Categóricas: One-hot encoding ou categorias nativas
- Sparsity: Suporta dados esparsos eficientemente
""",
        "metricas": """
- Log loss para classificação
- RMSE/MSE para regressão
- AUC para ranking
- Custom objectives permitidos
""",
        "vies_variancia": """
- Viés: Muito baixo
- Variância: Controlada por regularização
- Trade-off otimizado com regularização interna
""",
        "vantagens": [
            "Performance líder em muitos benchmarks",
            "Regularização nativa",
            "Handles missing values",
            "Paralelização eficiente",
            "Cross-validation nativo",
            "Early stopping automático"
        ],
        "desvantagens": [
            "Menos interpretável",
            "Mais parâmetros para tunerar",
            "Pode overfitar se mal configurado",
            "Memória e CPU intensos"
        ],
        "aplicacoes": [
            "Competições Kaggle (muito popular)",
            "Sistemas de recomendação",
            "Detecção de fraude",
            "Previsão de churn"
        ]
    },
    
    "LightGBM": {
        "tipo": "Supervisionado",
        "tarefa": "Classificação e Regressão",
        "parametrico": False,
        "instace_based": False,
        "escalas": False,
        "sensivel_outliers": "Média",
        "interpretabilidade": "Baixa",
        "performance": "Muito alta",
        "complexidade_treino": "O(k * n)",
        "complexidade_inferencia": "O(k)",
        "proposito": "Gradient Boosting mais rápido usando Leaf-wise growth e histogram-based splitting",
        "estrutura": " leaf-wise trees com histogram-based splitting",
        "funcao_perda": "Log Loss, MSE, Huber, etc.",
        "regularizacao": ["L1", "L2", "num_leaves", "min_data"],
        "como_aprende": """
LightGBM Innovation:
1. Leaf-wise (Level-wise vs Leaf-wise):
   - Cresce folha com maior loss reduction
   - Pode gerar árvores mais profundas mais rápido
   
2. Histogram-based splitting:
   - Discretiza valores contínuos em bins
   - Reduz complexidade de O(n) para O(bins)
   
3. Gradient-based One-Side Sampling (GOSS):
   - Mantém instâncias com gradiente alto
   - Amostra aleatoriamente instâncias com gradiente baixo
""",
        "como_prevê": "Mesma estrutura de boosted trees com inferência rápida",
        "pipeline": """
1. Entrada: Features X e target y
2. Pré-processamento:
   - Altamente otimizado para categorical features
   - Missing values automático
   - Dados podem ser muito grandes
3. Treinamento:
   - Leaf-wise tree growth
   - Histogram-based binning
   - GOSS para datasets imbalance
   - Exclusive Feature Bundling (EFB)
4. Ajuste de hiperparâmetros:
   - num_leaves: 31-1024 (mais que max_depth)
   - learning_rate, n_estimators
   - min_data_in_leaf
   - feature_fraction, bagging_fraction
5. Predição: Rápida, otimizada para memória
""",
        "overfitting": """
- Controles:
  * num_leaves: 31-1024 (limita complexidade)
  * min_data_in_leaf: mínimo samples por folha
  * max_depth: limite de profundidade
  * lambda_l1, lambda_l2: regularização
  * feature_fraction: randomização
""",
        "pre_processamento": """
- Escalonamento: NÃO NECESSÁRIO
- Outliers: Sensibilidade média
- Categóricas: Suporte NATIVO a categorias
- Velocidade: Mais rápido que XGBoost
- Memória: Menor consumo
""",
        "metricas": """
- Same as XGBoost
- Suporta métricas customizadas
- Early stopping nativo
""",
        "vies_variancia": """
- Viés: Muito baixo (leaf-wise pode ser mais complexo)
- Variância: Controlada por regularização
- Leaf-wise pode gerar mais overfitting que level-wise
""",
        "vantagens": [
            "Mais rápido que XGBoost",
            "Menos memória",
            "Suporte nativo a categóricas",
            "Melhor em datasets unbalanced",
            "Leaf-wise: melhor accuracy em shallow trees"
        ],
        "desvantagens": [
            "Menos popular que XGBoost",
            "Documentação menos extensa",
            "Pode overfitar em datasets pequenos",
            "Leaf-wise menos estável"
        ],
        "aplicacoes": [
            "Datasets muito grandes",
            "Baixa latência requerida",
            "Feature engineering intensive",
            "Ranking problems"
        ]
    },
    
    "CatBoost": {
        "tipo": "Supervisionado",
        "tarefa": "Classificação e Regressão",
        "parametrico": False,
        "instace_based": False,
        "escalas": False,
        "sensivel_outliers": "Baixa",
        "interpretabilidade": "Média",
        "performance": "Muito alta",
        "complexidade_treino": "O(k * n log n)",
        "complexidade_inferencia": "O(k)",
        "proposito": "Gradient Boosting otimizado para lidar com variáveis categóricas sem preprocessing",
        "estrutura": "Symmetric trees com ordered boosting",
        "funcao_perda": "Log Loss, RMSE, MultiClass",
        "regularizacao": ["L2", "ordered_boosting"],
        "como_aprende": """
CatBoost Innovations:
1. Ordered Boosting:
   - Evita target leakage
   - Usa permutações para calcular estatísticas
   - Reduces overfitting em dados pequenos
   
2. Target Statistics:
   - Para categóricas: calcula estatísticas do target
   - Usa prior com random permutations
   - Evita overfitting por encoding circular
   
3. Symmetric Trees:
   - Árvores balanceadas (symmetric)
   - Mais rápidas para inferência
   - Menos overfitting
""",
        "como_prevê": "Ensemble de symmetric trees com cálculo automático de categorias",
        "pipeline": """
1. Entrada: Features X e target y
2. Pré-processamento:
   - Categóricas: NÃO PRECISAM de encoding!
   - Especificar cat_features no construtor
   - Missing values automático
3. Treinamento:
   - Ordered boosting para evitar leakage
   - Target encoding interno
   - Symmetric trees
4. Ajuste de hiperparâmetros:
   - iterations, learning_rate
   - depth: 1-10 (symmetric trees)
   - l2_leaf_reg: regularização
   - border_count: bins para numéricas
5. Predição: Rápida com symmetric trees
""",
        "overfitting": """
- Controles:
  * Ordered boosting: reduz overfitting naturalmente
  * depth: limite de profundidade
  * l2_leaf_reg: regularização L2
  * random_strength: randomização
  * bagging_temperature: Bayesian bootstrap
""",
        "pre_processamento": """
- Escalonamento: NÃO NECESSÁRIO
- Outliers: BAIXA sensibilidade
- Categóricas: Suporte EXCELENTE nativo
  - Sem necessidade de encoding manual
  - Target encoding automático
  - Lida com alta cardinalidade
- Missing: Automático e robusto
""",
        "metricas": """
- MultiClass para múltiplas classes
- Logloss para classificação binária
- RMSE/MQE para regressão
- Early stopping automático
""",
        "vies_variancia": """
- Viés: Baixo (target encoding preciso)
- Variância: Reduzida por ordered boosting
- Symmetric trees: mais estável
""",
        "vantagens": [
            "Melhor para categóricas (sem preprocessing)",
            "Menos overfitting",
            "Handles missing values",
            "Symmetric trees = inferência rápida",
            "Multi-class otimizado"
        ],
        "desvantagens": [
            "Mais lento que LightGBM",
            "Menos flexível que XGBoost",
            "Documentação menos extensa"
        ],
        "aplicacoes": [
            "Dados com muitas categóricas",
            "Datasets pequenos (ordered boosting)",
            "Problemas multi-class",
            "Quando preprocessing é limitado"
        ]
    },
    
    "KNN": {
        "tipo": "Supervisionado",
        "tarefa": "Classificação e Regressão",
        "parametrico": False,
        "instace_based": True,
        "escalas": True,
        "sensivel_outliers": "Alta",
        "interpretabilidade": "Média",
        "performance": "Baixa",
        "complexidade_treino": "O(1)",
        "complexidade_inferencia": "O(n * d)",
        "proposito": "Classificar/regredir baseado nos k vizinhos mais próximos no espaço de features",
        "estrutura": "Não treina modelo, apenas armazena dados de treinamento",
        "funcao_perda": "Sem função de perda explícita",
        "regularizacao": ["k value", "distance metric"],
        "como_aprende": """
Lazy Learning (Instance-based):
1. Não há treinamento!
2. Apenas armazena os dados de treino
3. Armazena as labels correspondentes
4. Não há ajuste de pesos ou parâmetros
""",
        "como_prevê": """
1. Recebe nova amostra x
2. Calcula distância para TODAS as amostras de treino:
   - Euclidiana: √Σ(xᵢ - yᵢ)²
   - Manhattan: Σ|xᵢ - yᵢ|
   - Minkowski: (Σ|xᵢ - yᵢ|^p)^(1/p)
   - Cosseno: 1 - cos(θ)
3. Seleciona os k vizinhos mais próximos
4. Classificação: Votação majoritária
5. Regressão: Média/mediana dos k vizinhos
""",
        "pipeline": """
1. Entrada: Features X e target y
2. Pré-processamento:
   - ESCALONAMENTO OBRIGATÓRIO
   - Tratar outliers
   - Codificar categóricas
   - Reduzir dimensionalidade (se muitas features)
3. Treinamento: Armazenar X e y (zero custo)
4. Ajuste de hiperparâmetros:
   - k: número de vizinhos (ímpar para binária)
   - metric: distância euclidiana, manhattan, etc.
   - weights: uniform ou distance-weighted
5. Predição: Calcular distâncias para todos
""",
        "overfitting": """
- Overfitting: k muito pequeno (k=1)
  * Sensível a ruído
  * Fronteira de decisão irregular
- Underfitting: k muito grande
  * Fronteira muito suave
  * Ignora estrutura local
- Solução: Cross-validation para escolher k
""",
        "pre_processamento": """
- Escalonamento: CRÍTICO (distâncias sensíveis a escala)
- Outliers: ALTA sensibilidade (vizinhanca afetada)
- Categóricas: Encode necessário (ou distância especial)
- Dimensionalidade: Curse of dimensionality
  * Reduzir com PCA ou feature selection
- Normalização: MinMax ou Z-score
""",
        "metricas": """
- Distância: Euclidiana, Manhattan, Minkowski, Cosseno
- Weighted KNN: weight = 1/d² (mais próximo = mais peso)
- Validação: Cross-validation para k
""",
        "vies_variancia": """
- Viés: Baixo (modelo flexível)
- Variância: Alta (sensível a dados específicos)
- k pequeno = alta variância, baixo viés
- k grande = baixa variância, alto viés
- Curse of dimensionality: variância aumenta com d
""",
        "vantagens": [
            "Simples de entender e implementar",
            "Não tem fase de treinamento",
            "Lida com múltiplas classes naturalmente",
            "Poucos hiperparâmetros (k e métrica)",
            "Bom para problemas com fronteiras complexas"
        ],
        "desvantagens": [
            "Lento para inferência (busca linear)",
            "Sensível a dimensionalidade",
            "Escalonamento obrigatório",
            "Sensível a outliers",
            "Não funciona bem com muitas features"
        ],
        "aplicacoes": [
            "Sistemas de recomendação",
            "Busca por similaridade",
            "Classificação de imagens simples",
            "Previsão de preferências"
        ]
    },
    
    "SVM": {
        "tipo": "Supervisionado",
        "tarefa": "Classificação e Regressão",
        "parametrico": False,
        "instace_based": False,
        "escalas": True,
        "sensivel_outliers": "Alta",
        "interpretabilidade": "Baixa",
        "performance": "Alta",
        "complexidade_treino": "O(n²) a O(n³)",
        "complexidade_inferencia": "O(d)",
        "proposito": "Encontrar hiperplano ótimo que maximiza a margem entre classes",
        "estrutura": "Hiperplano: w·x + b = 0 com support vectors nas margens",
        "funcao_perda": "Hinge Loss",
        "regularizacao": ["C (soft margin)", "kernel"],
        "como_aprende": """
Maximum Margin Classifier:
1. Encontrar hiperplano que maximiza margem:
   - Margem = distância entre support vectors
   - w·x + b = +1 e w·x + b = -1 são as margens
   
2. Problema de otimização:
   min: ||w||²/2
   s.t.: yᵢ(w·xᵢ + b) ≥ 1
   
3. Soft Margin (com C):
   min: ||w||²/2 + C Σξᵢ
   s.t.: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ
   
4. KKT conditions definem support vectors
""",
        "como_prevê": """
1. Calcula f(x) = w·x + b
2. Classificação: sign(f(x))
3. Probabilidade: usando Platt scaling ou sigmoid
""",
        "pipeline": """
1. Entrada: Features X e target y
2. Pré-processamento:
   - ESCALONAMENTO CRÍTICO
   - Tratar outliers (sensibilidade alta)
   - Codificar categóricas
3. Treinamento:
   - Escolher kernel:
     * Linear: K(x,z) = x·z
     * RBF: K(x,z) = exp(-γ||x-z||²)
     * Polynomial: K(x,z) = (γx·z + r)^d
     * Sigmoid: K(x,z) = tanh(γx·z + r)
   - Parâmetros: C, gamma, kernel
4. Ajuste de hiperparâmetros:
   - C: trade-off margem vs erro
   - gamma: influência de cada sample (RBF)
   - kernel: tipo de kernel
5. Predição: Baseada em support vectors
""",
        "overfitting": """
- Causas:
  * C muito alto: overfitting
  * gamma muito alto (RBF): overfitting
  * Outliers mal tratados
- Soluções:
  * Grid search para C e gamma
  * Cross-validation
  * Normalizar dados
  * Usar soft margin adequado
""",
        "pre_processamento": """
- Escalonamento: CRÍTICO e obrigatório
- Outliers: ALTA sensibilidade (afeta support vectors)
- Categóricas: Encode necessário
- Dimensão: Funciona bem em alta dimensão
- Normalização: Z-score ou MinMax
""",
        "metricas": """
- Margin: 2/||w||
- Hinge Loss: max(0, 1 - y·f(x))
- Cross-validation para validação
""",
        "vies_variancia": """
- Linear SVM: Alto viés, baixa variância
- RBF com gamma alto: Baixo viés, alta variância
- C alto: Menos regularization, mais overfitting
- Trade-off controlado por C e gamma
""",
        "vantagens": [
            "Fronteiras complexas com kernels",
            "Bom em alta dimensionalidade",
            "Memória eficiente (suport vectors)",
            "Risco moderado de overfitting",
            "Versátil com diferentes kernels"
        ],
        "desvantagens": [
            "Lento para treinar (O(n²)-O(n³))",
            "Sensível a escala",
            "Não escala bem para n grande",
            "Parâmetros difíceis de tunerar",
            "Não dá probabilidades diretamente"
        ],
        "aplicacoes": [
            "Classificação de texto",
            "Reconhecimento de padrões",
            "Bioinformática",
            "Detecção de anomalias"
        ]
    },
    
    "Naive Bayes": {
        "tipo": "Supervisionado",
        "tarefa": "Classificação",
        "parametrico": True,
        "instace_based": False,
        "escalas": False,
        "sensivel_outliers": "Baixa",
        "interpretabilidade": "Alta",
        "performance": "Média",
        "complexidade_treino": "O(n * d)",
        "complexidade_inferencia": "O(d * k)",
        "proposito": "Classificador probabilístico baseado no Teorema de Bayes com independência condicional",
        "estrutura": "P(y|X) ∝ P(y) * ∏ P(xᵢ|y)",
        "funcao_perda": "Log loss (usado para otimização implícita)",
        "regularizacao": [" Laplace smoothing", "alpha"],
        "como_aprende": """
Treinamento (Estimação de parâmetros):
1. Calcular prior P(y) para cada classe:
   P(y) = nᵧ / n

2. Calcular likelihood P(xᵢ|y):
   - Gaussiano: Para contínuas
   - Multinomial: Para contagens (NLP)
   - Bernoulli: Para binárias
   
3. Variantes principais:
   - Gaussian NB: Assume normal distribution
   - Multinomial NB: Para features de contagem
   - Bernoulli NB: Para features binárias
""",
        "como_prevê": """
Teorema de Bayes:
P(y|X) = P(y) * P(X|y) / P(X)

Simplificação Naive:
P(y|X) ∝ P(y) * ∏ᵢ P(xᵢ|y)

Para cada classe:
1. Calcula prior P(y)
2. Multiplica por ∏ P(xᵢ|y) para cada feature
3. Seleciona classe com maior probabilidade
""",
        "pipeline": """
1. Entrada: Features X e target y
2. Pré-processamento:
   - Variante escolhe o encoding:
     * Gaussian: dados contínuos
     * Multinomial: TF-IDF, contagens
     * Bernoulli: dados binários
   - Tratamento de zero-probabilities
3. Treinamento:
   - Estimar P(y) (priors)
   - Estimar P(xᵢ|y) (likelihoods)
   - Laplace smoothing: (count + α) / (total + α*n_classes)
4. Ajuste de hiperparâmetros:
   - alpha: smoothing parameter
   - var_smoothing (Gaussian NB)
5. Predição: Multiplicação de probabilidades
""",
        "overfitting": """
- Overfitting: Pode ocorrer com:
  * Muitas features esparsas
  * Poucas observações
  * Zeros em probabilidade
- Soluções:
  * Laplace smoothing (α > 0)
  * Feature selection
  * Independence assumption relaxada
""",
        "pre_processamento": """
- Escalonamento: NÃO NECESSÁRIO
- Outliers: Baixa sensibilidade
- Categóricas: FREQUENTISTA (counts)
- Contagens: TF-IDF para NLP
- Sparsity: Lida bem com dados esparsos
""",
        "metricas": """
- Probabilidades de classe
- Log-loss para avaliar calibração
- Accuracy, Precision, Recall
-works well despite naive assumptions
""",
        "vies_variancia": """
- Viés: ALTO (independência raramente verdadeira)
- Variância: BAIXA (estável com novos dados)
- Trade-off: Subestimação de incertezas
- Modelo simples = difícil overfit
""",
        "vantagens": [
            "Extremamente rápido",
            "Poucos hiperparâmetros",
            "Lida bem com alta dimensionalidade",
            "Funciona com poucos dados",
            "Probabilidades de saída",
            "Boa baseline para NLP"
        ],
        "desvantagens": [
            "Independência raramente verdadeira",
            "Probabilidades mal calibradas",
            "Performance inferior a modelos complexos",
            "Zeros destroem probabilidades"
        ],
        "aplicacoes": [
            "Classificação de spam (NLP)",
            "Análise de sentimentos",
            "Categorização de documentos",
            "Sistemas de recomendação simples"
        ]
    },
    
    "Redes Neurais": {
        "tipo": "Supervisionado",
        "tarefa": "Classificação e Regressão",
        "parametrico": False,
        "instace_based": False,
        "escalas": True,
        "sensivel_outliers": "Média",
        "interpretabilidade": "Muito baixa",
        "performance": "Muito alta",
        "complexidade_treino": "O(epochs * n * d * h)",
        "complexidade_inferencia": "O(d * h + h * o)",
        "proposito": "Aprender representações complexas através de camadas de neurônios com funções de ativação não-lineares",
        "estrutura": "Input → Hidden layers → Output, com pesos sinápticos w e biases b",
        "funcao_perda": "Cross-Entropy, MSE, Huber, etc.",
        "regularizacao": ["Dropout", "L2", "Batch Norm", "Early Stopping"],
        "como_aprende": """
Backpropagation + Gradient Descent:
1. Forward pass:
   - z¹ = x · W¹ + b¹
   - a¹ = σ(z¹) (ativação)
   - Repetir para cada camada até output
   
2. Calcular loss: L(y, ŷ)

3. Backward pass:
   - δᴸ = ∂L/∂aᴸ * σ'(zᴸ) (camada final)
   - δˡ = (Wˡ⁺¹)ᵀ · δˡ⁺¹ * σ'(zˡ) (camadas ocultas)
   - ∂L/∂Wˡ = aˡ⁻¹ · δˡ
   - ∂L/∂bˡ = δˡ

4. Atualizar pesos:
   - W ← W - η * ∂L/∂W
""",
        "como_prevê": """
1. Forward pass pela rede:
   - Cada neurônio: z = Σ(w·x) + b
   - Aplicar função de ativação
2. Camada final:
   - Sigmoid: probabilities (binária)
   - Softmax: probabilities (multi-classe)
   - Linear: valores contínuos (regressão)
""",
        "pipeline": """
1. Entrada: Features X e target y
2. Pré-processamento:
   - ESCALONAMENTO CRÍTICO (0-1 ou z-score)
   - Tratar outliers
   - Codificar categóricas (one-hot)
   - Normalizar imagens (se CNN)
3. Treinamento:
   - Inicializar pesos
   - Mini-batch gradient descent
   - Backpropagation
   - Otimizadores (Adam, SGD, etc.)
4. Ajuste de hiperparâmetros:
   - learning_rate: 0.001-0.1
   - batch_size: 32-256
   - epochs: early stopping
   - layers, neurons
   - dropout rate
   - activation functions
5. Predição: Forward pass
""",
        "overfitting": """
- Causas:
  * Muitas camadas/neurônios
  * Poucos dados
  * Muitas epochs
  * Sem regularização
- Soluções:
  * Dropout (0.2-0.5)
  * L2 regularization
  * Batch normalization
  * Early stopping
  * Data augmentation
  * Reduce complexity
""",
        "pre_processamento": """
- Escalonamento: CRÍTICO e obrigatório
- Outliers: Sensibilidade média
- Categóricas: One-hot encoding
- Imagens: Normalizar pixels [0,1]
- Textos: Embeddings
- Dimensionalidade: Pode ser alta
""",
        "metricas": """
- Classificação: Cross-entropy, Accuracy
- Regressão: MSE, MAE
- Perplexity (NLP)
- Custom losses para problemas específicos
""",
        "vies_variancia": """
- Viés: Reduzido com profundidade
- Variância: Alta (muito flexível)
- Deep networks: podem ter vanishing gradients
- Regularização: Essencial para controlar
- Underfitting: rede muito rasa
""",
        "vantagens": [
            "Aprende representações automáticas",
            "Captura relações não-lineares complexas",
            "Versátil (CNN, RNN, Transformers)",
            "Scale para grandes volumes",
            "Transfer learning"
        ],
        "desvantagens": [
            "Black box (sem interpretabilidade)",
            "Muitos hiperparâmetros",
            "Precisa muitos dados",
            "Computacionalmente intensivo",
            "Sensible a inicialização"
        ],
        "aplicacoes": [
            "Visão computacional (CNN)",
            "NLP (Transformers, RNN)",
            "Reconhecimento de voz",
            "Jogos (Deep RL)",
            "Geração de conteúdo (GANs)"
        ]
    },
    
    "K-Means": {
        "tipo": "Não supervisionado",
        "tarefa": "Clustering",
        "parametrico": False,
        "instace_based": False,
        "escalas": True,
        "sensivel_outliers": "Alta",
        "interpretabilidade": "Média",
        "performance": "Média",
        "complexidade_treino": "O(k * n * i)",
        "complexidade_inferencia": "O(k)",
        "proposito": "Segmentar dados em k clusters baseados em similaridade (distância)",
        "estrutura": "k centróides que representam clusters",
        "funcao_perda": "Inércia (Within-cluster Sum of Squares)",
        "regularizacao": ["k value", "inicialização"],
        "como_aprende": """
Algoritmo Lloyd's:
1. Inicializar k centróides (random ou k-means++)
2. Repetir até convergência:
   a) Atribuir cada ponto ao cluster mais próximo:
      clusterᵢ = argminⱼ ||xᵢ - μⱼ||²
   b) Recalcular centróides:
      μⱼ = mean(X[clusterᵢ == j])

3. Convergência: centróides estáveis ou mudança mínima
4. Output: k clusters e k centróides
""",
        "como_prevê": """
1. Recebe novo ponto x
2. Calcula distância para cada centróide
3. Atribui ao cluster do centróide mais próximo
4. Output: label do cluster (0 a k-1)
""",
        "pipeline": """
1. Entrada: Features X (sem labels)
2. Pré-processamento:
   - ESCALONAMENTO CRÍTICO
   - Tratar outliers (sensibilidade alta)
   - Reduzir dimensionalidade (PCA) se necessário
   - Verificar suposições de clusters esféricos
3. Treinamento:
   - Escolher k ( elbow, silhouette, gap statistic)
   - Inicialização (k-means++ recomendado)
   - Iterar até convergência
4. Ajuste de hiperparâmetros:
   - k: número de clusters
   - n_init: número de inicializações
   - max_iter: máximo de iterações
   - tol: tolerância de convergência
5. Predição: Atribuição a centróide mais próximo
""",
        "overfitting": """
- Overfitting: k muito alto
  * Cada ponto vira cluster próprio
  * Inércia continua diminuindo
- Underfitting: k muito baixo
  * Clusters agregam grupos distintos
- Como evitar:
  * Elbow method
  * Silhouette score
  * Gap statistic
  * Conhecimento de domínio
""",
        "pre_processamento": """
- Escalonamento: CRÍTICO e obrigatório
- Outliers: ALTA sensibilidade (distorce centróides)
  * Remover outliers antes ou usar K-Medoids
- Categóricas: Não funciona diretamente
  * Alternative: K-Modes ou mixed metrics
- Dimensionalidade: Reduzir com PCA se d > 10
- Normalização: MinMax ou Z-score
""",
        "metricas": """
- Inércia: Σ ||x - μ||² (WCSS)
- Silhouette Score: (-1 a 1)
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Elbow method: plot k vs Inércia
""",
        "vies_variancia": """
- Viés: Assumir clusters esféricos e iguais
- Variância: Sensível a inicialização
  * k-means++ reduz variância
  * Múltiplas inicializações
- Locais óptimos: k-means pode ficar preso em mínimos locais
""",
        "vantagens": [
            "Simples e interpretável",
            "Rápido para k pequeno",
            "Escalável para grandes datasets",
            "Fácil de implementar",
            "Bom para clusters esféricos"
        ],
        "desvantagens": [
            "Assume clusters esféricos",
            "Assume clusters de tamanho similar",
            "Sensível a outliers",
            "Sensível a inicialização",
            "Precisa definir k a priori"
        ],
        "aplicacoes": [
            "Segmentação de clientes",
            "Compressão de imagens",
            "Detecção de anomalias",
            "Organização de documentos",
            "Análise de comportamento"
        ]
    },
    
    "DBSCAN": {
        "tipo": "Não supervisionado",
        "tarefa": "Clustering",
        "parametrico": False,
        "instace_based": False,
        "escalas": True,
        "sensivel_outliers": "Baixa",
        "interpretabilidade": "Média",
        "performance": "Média",
        "complexidade_treino": "O(n²) sem índice, O(n log n) com índice",
        "complexidade_inferencia": "O(n)",
        "proposito": "Identificar clusters de forma arbitrária e detectar ruído/outliers",
        "estrutura": "Pontos classificados como Core, Border ou Noise",
        "funcao_perda": "Sem função de perda",
        "regularizacao": ["eps", "min_samples"],
        "como_aprende": """
Density-based Clustering:
1. Conceitos:
   - ε (eps): raio da vizinhança
   - MinPts: mínimo pontos para ser core
   - ε-neighborhood: N_ε(p) = {q | dist(p,q) ≤ ε}
   
2. Core point: |N_ε(p)| ≥ MinPts
3. Border point: Pertence a vizinhança de core point
4. Noise point: Não é core nem border

5. Algoritmo:
   Para cada ponto não visitado:
   - Se é core point: expandir cluster
   - Senão: marcar como noise
   - Expandir: DFS/BFS para todos Density-reachable
""",
        "como_prevê": """
1. Recebe novo ponto x
2. Verifica se está na ε-vizinhança de algum core point
3. Se sim: atribui ao cluster correspondente
4. Se não: marca como noise/outlier
""",
        "pipeline": """
1. Entrada: Features X (sem labels)
2. Pré-processamento:
   - ESCALONAMENTO CRÍTICO (distâncias)
   - Outliers NÃO afetam tanto (são marcados como noise)
   - Reduzir dimensionalidade se necessário
   - Escolher métrica de distância apropriada
3. Treinamento:
   - Estimar ε (eps) usando k-distance graph
   - Definir MinPts (regra: 2 * d)
   - Rodar DBSCAN
4. Ajuste de hiperparâmetros:
   - eps: raio da vizinhança
   - min_samples: densidade mínima
   - metric: euclidean, cosine, etc.
5. Predição: Verificar proximidade de core points
""",
        "overfitting": """
- Overfitting:
  * eps muito pequeno: muitos clusters pequenos ou noise
  * MinPts muito alto: poucos pontos são core
- Underfitting:
  * eps muito grande: tudo vira um cluster
  * MinPts muito baixo: clusters espúrios
- Tuning:
  * Usar k-distance graph para eps
  * Silhouette score para avaliação
""",
        "pre_processamento": """
- Escalonamento: CRÍTICO e obrigatório
- Outliers: Baixa sensibilidade (são identificados como noise)
- Categóricas: Não funciona diretamente
  * Usar Gower distance ou HDBSCAN
- Dimensionalidade: Curse of dimensionality
  * Reduzir dimensões (PCA)
  * Usar métricas apropriadas
""",
        "metricas": """
- Sem métrica de loss tradicional
- Silhouette Score para avaliação
- Número de clusters encontrados
- Percentual de noise (outliers)
- DBCV (Density-Based Clustering Validation)
""",
        "vies_variancia": """
- Viés: Assumir clusters de densidade similar
- Variância: Reduzida (baseado em densidade)
- Sensibilidade: Alta a eps
- Não assume número de clusters a priori
""",
        "vantagens": [
            "Não precisa de k (clusters arbitrários)",
            "Identifica outliers automaticamente",
            "Lida com formas não-esféricas",
            "Robusto a outliers (os marca como noise)",
            "Determinístico (a menos de ruído)"
        ],
        "desvantagens": [
            "Sensível aeps (difícil de tunerar)",
            "Não funciona bem com densidades variadas",
            "Lento para n grande sem índice espacial",
            "Não atribui todos os pontos a clusters",
            "Não funciona com dados categóricos"
        ],
        "aplicacoes": [
            "Detecção de anomalias",
            "Segmentação de imagens",
            "Análise de logs",
            "Geospatial clustering",
            "Identificação de padrões de compra"
        ]
    },
    
    "PCA": {
        "tipo": "Não supervisionado",
        "tarefa": "Redução de Dimensionalidade",
        "parametrico": True,
        "instace_based": False,
        "escalas": True,
        "sensivel_outliers": "Alta",
        "interpretabilidade": "Média",
        "performance": "Alta",
        "complexidade_treino": "O(d²n) + O(d³)",
        "complexidade_inferencia": "O(d' * n)",
        "proposito": "Reduzir dimensionalidade preservando máxima variância dos dados",
        "estrutura": "k componentes principais (combinações lineares de features originais)",
        "funcao_perda": "Reconstrução de erro (MSE)",
        "regularizacao": ["n_components", "standardization"],
        "como_aprende": """
Principal Component Analysis:
1. Centralizar dados (subtrair média)
2. Calcular matriz de covariância Σ
3. Decomposição de autovetores:
   Σ = VΛVᵀ
   
4. Ordenar autovetores por autovalores decrescentes:
   λ₁ ≥ λ₂ ≥ ... ≥ λₐ
   
5. Selecionar top k autovetores (componentes)
6. Transformar: Z = X · V_k

Os autovetores definem as direções de máxima variância
""",
        "como_prevê": """
1. Centroide: X_centered = X - μ
2. Projeção: Z = X_centered · V_k
3. Reconstrução (opcional): X_reconstructed = Z · V_kᵀ + μ
4. Output: Dados em k dimensões
""",
        "pipeline": """
1. Entrada: Features X (sem labels necessários)
2. Pré-processamento:
   - ESCALONAMENTO CRÍTICO (Z-score)
   - Tratar outliers (sensibilidade alta)
   - Verificar linearidade (assunção)
   - Correlação entre features é importante
3. Treinamento:
   - Calcular média e desvio padrão
   - Padronizar dados
   - Calcular matriz de covariância
   - Extrair autovetores/valores
   - Ordenar por importância
4. Ajuste de hiperparâmetros:
   - n_components: k dimensões
     * Variância acumulada (0.95+)
     * Scree plot (elbow)
     * Kaiser criterion (λ > 1)
5. Predição: Transformação linear
""",
        "overfitting": """
- Overfitting: Pode não ser problema direto
  * Redução de dimensões reduz overfitting
  * Mas escolha de k muito baixo perde informação
- Como evitar:
  * Validar com downstream task
  * Usar cumulative variance plot
  * Scree plot para escolher k
  * Cross-validation
""",
        "pre_processamento": """
- Escalonamento: CRÍTICO e obrigatório
- Outliers: ALTA sensibilidade (distorcem covariância)
  * Remover outliers ou usar robust PCA
  * Winsorização
- Categóricas: Não funciona diretamente
  * Usar MCA para categóricas
  * Encode categóricas primeiro
- Linearidade: Assumida (baseado em covariância)
- Missing: Tratar antes
""",
        "metricas": """
- Explained variance ratio: λᵢ / Σλ
- Cumulative variance: % acumulada
- Reconstruction error: ||X - X_reconstructed||²
- Scree plot: λ vs componente
- Kaiser criterion: λ > 1 (para dados padronizados)
""",
        "vies_variancia": """
- Viés: Não-linearidades perdidas
- Variância: Maximizada nos componentes
- Trade-off: k pequeno = mais viés, menos variância
- k grande = menos viés, mais ruído
""",
        "vantagens": [
            "Reduz dimensionalidade",
            "Remove redundância (features correlacionadas)",
            "Ajuda a visualizar dados em 2D/3D",
            "Reduz overfitting",
            "Não-supervisionado",
            "Interpretação dos componentes"
        ],
        "desvantagens": [
            "Assume linearidade",
            "Sensível a outliers",
            "Perde interpretabilidade das features",
            "Pode perder informação importante",
            "Não preserva distâncias"
        ],
        "aplicacoes": [
            "Pré-processamento para ML",
            "Visualização de dados",
            "Compressão de imagens",
            "Extração de features",
            "Noise reduction",
            "Análise exploratória"
        ]
    }
}

def render_fundamentals(data):
    st.markdown("### 📌 Fundamentos do Algoritmo")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**🎯 Propósito:**")
        st.info(data['proposito'])
        
        st.markdown(f"**📊 Tipo:**")
        st.success(data['tipo'])
        
        st.markdown(f"**📝 Tarefa:**")
        st.info(data['tarefa'])
    
    with col2:
        st.markdown(f"**🔣 Estrutura:**")
        st.code(data['estrutura'])
        
        st.markdown(f"**⚖️ Função de Perda:**")
        st.warning(data['funcao_perda'])
    
    with st.expander("📐 Representação Matemática"):
        st.latex(data['estrutura'])

def render_functioning(data):
    st.markdown("### 📌 Funcionamento Detalhado")
    
    with st.expander("🧠 Como o Algoritmo Aprende"):
        st.text(data['como_aprende'])
    
    with st.expander("🔮 Como Realiza Previsões"):
        st.text(data['como_prevê'])
    
    with st.expander("🔄 Pipeline Completo"):
        st.code(data['pipeline'])

def render_generalization(data):
    st.markdown("### 📌 Generalização e Overfitting")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**⚠️ Overfitting:**")
        st.text(data['overfitting'][:500])
    
    with col2:
        st.markdown("**🛡️ Regularização:**")
        if isinstance(data['regularizacao'], list):
            for r in data['regularizacao']:
                st.write(f"- {r}")
        else:
            st.write(data['regularizacao'])

def render_technical(data):
    st.markdown("### 📌 Características Técnicas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Paramétrico", "✅ Sim" if data['parametrico'] else "❌ Não")
    
    with col2:
        st.metric("Instance-based", "✅ Sim" if data['instace_based'] else "❌ Não")
    
    with col3:
        st.metric("Escalas Necessário", "✅ Sim" if data['escalas'] else "❌ Não")

def render_preprocessing(data):
    st.markdown("### 📌 Dados e Pré-processamento")
    st.text(data['pre_processamento'])

def render_metrics(data):
    st.markdown("### 📌 Métricas e Função de Perda")
    
    if 'metricas_regressao' in data:
        st.markdown("**📏 Métricas de Regressão:**")
        st.text(data['metricas_regressao'])
    elif 'metricas_classificacao' in data:
        st.markdown("**📊 Métricas de Classificação:**")
        st.text(data['metricas_classificacao'])
    elif 'metricas' in data:
        st.markdown("**📊 Métricas:**")
        st.text(data['metricas'])

def render_performance(data):
    st.markdown("### 📌 Performance e Complexidade")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**⏱️ Complexidade:**")
        st.write(f"- Treino: `{data['complexidade_treino']}`")
        st.write(f"- Inferência: `{data['complexidade_inferencia']}`")
    
    with col2:
        st.markdown("**📈 Sensibilidade:**")
        st.write(f"- Outliers: `{data['sensivel_outliers']}`")
        st.write(f"- Interpretabilidade: `{data['interpretabilidade']}`")
        st.write(f"- Performance: `{data['performance']}`")

def render_pros_cons(data):
    st.markdown("### 📌 Vantagens e Desvantagens")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**✅ Vantagens:**")
        for v in data['vantagens']:
            st.write(f"- {v}")
    
    with col2:
        st.markdown("**❌ Desvantagens:**")
        for d in data['desvantagens']:
            st.write(f"- {d}")

def render_applications(data):
    st.markdown("### 📌 Aplicações Reais")
    
    cols = st.columns(len(data['aplicacoes']))
    for i, app in enumerate(data['aplicacoes']):
        with cols[i]:
            st.success(app)

def render_bias_variance():
    st.markdown("### 📌 Vies vs Variância")
    st.text("""
**Definições:**

🔹 VIÉS (Bias):
- Erro por assumir premissas simplificadas
- Modelo ignora padrões reais
- Underfitting: viés alto

🔹 VARIÂNCIA (Variance):
- Sensibilidade a pequenas mudanças nos dados
- Modelo captura ruído como padrão
- Overfitting: variância alta

**Relação com Complexidade:**

        Alto
         |         /  ✓
Viés     |        /
         |       /
         |      /
         |     /
         |    /
         |   /  -----
         |  /       
         | /        
         |/         
         +------------------> Complexidade
              \      ____
               \___/    Variância
                  Baixa
                  
**Trade-off:**
- Viés ↓ quando complexidade ↑
- Variância ↑ quando complexidade ↑
- Objetivo: ponto ótimo de complexidade

**Como Encontrar Equilíbrio:**
1. Cross-validation
2. Validação com dados separados
3. Learning curves
4. Regularização adequada
    """)

def render_ensemble_learning():
    st.markdown("## 🚀 Técnicas Avançadas")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Ensemble Learning", "Bagging", "Boosting", "Grid Search"])
    
    with tab1:
        st.markdown("### 🔹 Ensemble Learning")
        st.text("""
**O que é?**
Combinar múltiplos modelos para obter melhor performance que qualquer modelo individual.

**Por que usar?**
- Reduz variância (Bagging)
- Reduz viés (Boosting)
- Melhora robustez
- Combina forças de diferentes modelos

**Tipos principais:**
1. Bagging (Bootstrap Aggregating)
2. Boosting (Sequential improvement)
3. Stacking (Meta-learning)
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success("Bagging: Paralelo, reduz variância")
        with col2:
            st.success("Boosting: Sequencial, reduz viés")
        with col3:
            st.success("Stacking: Meta-modelo")
    
    with tab2:
        st.markdown("### 🔹 Bagging")
        st.text("""
**Conceito:**
1. Bootstrap: Amostras com reposição
2. Aggregate: Combinar previsões (votação/média)

**Random Forest:**
- Bagging + Random Subspace (features aleatórias)
- Reduz variância significativamente
- Florestas mais robustas

**Benefícios:**
- Paralelizável
- Reduz overfitting
- Estável com novos dados
- Funciona com árvores de decisão
        """)
    
    with tab3:
        st.markdown("### 🔹 Boosting")
        st.text("""
**Conceito:**
Sequência de modelos, cada um corrigindo erros do anterior.

**Tipos principais:**

🔸 AdaBoost (Adaptativo):
- Pondera instâncias erradas
- Foca em exemplos difíceis
- Atualiza pesos sequencialmente

🔸 Gradient Boosting:
- Usa gradiente da loss function
- Cada modelo corrige resíduos
- Otimização funcional

🔸 Diferenças:
- AdaBoost: reweighting de samples
- Gradient Boosting: gradient descent funcional
- Ambos: sequenciais e sequenciais

**Vantagens:**
- Reduz viés e variância
- Performance state-of-the-art
- Versátil
        """)
    
    with tab4:
        st.markdown("### 🔹 Grid Search")
        st.text("""
**O que é?**
Busca exaustiva em grid de hiperparâmetros.

**Quando usar?**
- Espaço de hiperparâmetros pequeno
- Hiperparâmetros críticos
- Quando há tempo computacional

**Como funciona:**
1. Definir grid de parâmetros
2. Testar todas combinações
3. Avaliar com cross-validation
4. Selecionar melhor combinação

**Código exemplo (sklearn):**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.1]
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'
)
grid.fit(X, y)
```

**Cuidados:**
- Pode ser muito lento
- Overfitting ao conjunto de validação
- Considere RandomizedSearchCV
- Use early stopping
        """)

def render_comparison_table():
    st.markdown("## ⚖️ Tabela Comparativa de Algoritmos")
    
    comparison_data = {
        'Algoritmo': ['Regressão Linear', 'Regressão Logística', 'Árvore de Decisão', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost', 'KNN', 'SVM', 'Naive Bayes', 'Redes Neurais', 'K-Means', 'DBSCAN', 'PCA'],
        'Tipo': ['Sup.', 'Sup.', 'Sup.', 'Sup.', 'Sup.', 'Sup.', 'Sup.', 'Sup.', 'Sup.', 'Sup.', 'Sup.', 'Sup.', 'Não Sup.', 'Não Sup.', 'Não Sup.'],
        'Tarefa': ['Reg.', 'Class.', 'Class/Reg', 'Class/Reg', 'Class/Reg', 'Class/Reg', 'Class/Reg', 'Class/Reg', 'Class/Reg', 'Class/Reg', 'Class.', 'Class/Reg', 'Cluster', 'Cluster', 'Red.Dim.'],
        'Escala': ['Sim', 'Sim', 'Não', 'Não', 'Não', 'Não', 'Não', 'Não', 'Sim', 'Sim', 'Não', 'Sim', 'Sim', 'Sim', 'Sim'],
        'Outliers': ['Alta', 'Média', 'Baixa', 'Baixa', 'Média', 'Média', 'Média', 'Baixa', 'Alta', 'Alta', 'Baixa', 'Média', 'Alta', 'Baixa', 'Alta'],
        'Interpretabilidade': ['Alta', 'Alta', 'Alta', 'Média', 'Baixa', 'Baixa', 'Baixa', 'Média', 'Média', 'Baixa', 'Alta', 'Muito Baixa', 'Média', 'Média', 'Média'],
        'Performance': ['Média', 'Média', 'Média', 'Alta', 'Muito Alta', 'Muito Alta', 'Muito Alta', 'Muito Alta', 'Baixa', 'Alta', 'Média', 'Muito Alta', 'Média', 'Média', 'Alta'],
        'Velocidade': ['🚀 Rápido', '🚀 Rápido', '🚀 Rápido', '⚡ Média', '🐢 Lento', '🐢 Lento', '⚡ Rápida', '⚡ Rápida', '🐢 Lento', '🐢 Lento', '🚀 Rápido', '🐢 Lento', '⚡ Média', '🐢 Lento', '⚡ Rápida']
    }
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("""
**Legenda:**
- **Sup.**: Supervisionado | **Não Sup.**: Não Supervisionado
- **Reg.**: Regressão | **Class.**: Classificação | **Cluster**: Clustering | **Red.Dim.**: Redução de Dimensionalidade
- **Escala**: Se escalonamento é necessário
- **Outliers**: Sensibilidade a outliers
    """)

def main():
    st.title("🧠 ML Master - Estudo Completo de Algoritmos")
    st.markdown("### Selecione um algoritmo para estudar em profundidade")
    
    with st.sidebar:
        st.header("🎛️ Configurações")
        algorithm = st.selectbox(
            "**Selecione o Algoritmo:**",
            ALGORITHMS,
            index=0
        )
        
        st.markdown("---")
        st.markdown("### 📚 Navegação Rápida")
        section = st.radio(
            "Seção:",
            ["🏠 Visão Geral", "📊 Estudo Completo", "🚀 Avançado", "⚖️ Comparação"]
        )
    
    if section == "🏠 Visão Geral":
        st.markdown("## 🏠 Visão Geral")
        st.markdown(f"""
        ### Algoritmo Selecionado: **{algorithm}**
        
        Bem-vindo ao ML Master! Este aplicativo permite estudar algoritmos de Machine Learning em profundidade.
        
        **Funcionalidades:**
        - 📊 **Estudo Completo**: Análise detalhada de cada algoritmo
        - 🚀 **Técnicas Avançadas**: Ensemble Learning, Grid Search, etc.
        - ⚖️ **Comparação**: Tabela comparativa entre algoritmos
        
        **Use o menu lateral para navegar entre as seções.**
        """)
        
        data = ALGORITHM_DATA[algorithm]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Tipo", data['tipo'])
        with col2:
            st.metric("Tarefa", data['tarefa'])
        with col3:
            st.metric("Paramétrico", "Sim" if data['parametrico'] else "Não")
        with col4:
            st.metric("Escalas", "Sim" if data['escalas'] else "Não")
        
        render_pros_cons(data)
    
    elif section == "📊 Estudo Completo":
        st.markdown(f"## 📊 Estudo Completo: {algorithm}")
        
        data = ALGORITHM_DATA[algorithm]
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📌 Fundamentos", "🔄 Funcionamento", "📈 Detalhes", "💡 Extra"
        ])
        
        with tab1:
            render_fundamentals(data)
        
        with tab2:
            render_functioning(data)
            render_generalization(data)
        
        with tab3:
            render_technical(data)
            render_preprocessing(data)
            render_metrics(data)
            render_performance(data)
        
        with tab4:
            render_pros_cons(data)
            render_applications(data)
            render_bias_variance()
    
    elif section == "🚀 Avançado":
        render_ensemble_learning()
    
    elif section == "⚖️ Comparação":
        render_comparison_table()

if __name__ == "__main__":
    main()

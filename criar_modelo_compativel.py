# criar_modelo_compativel.py
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

print("⚙️ Criando modelo compatível com o novo Front-end...")

# 1. Simular dados de treinamento com as colunas NOVAS do seu HTML
# Isso garante que o modelo saiba ler exatamente o que o site envia.
data = {
    'Age': np.random.randint(20, 80, 100),
    'Gender': np.random.choice(['M', 'F'], 100),
    'Cholesterol': np.random.randint(150, 500, 100),
    'HeartRate': np.random.randint(60, 200, 100),
    'Diabetes': np.random.choice([0, 1], 100),
    'FamilyHistory': np.random.choice([0, 1], 100),
    'Smoking': np.random.choice([0, 1], 100),
    'Obesity': np.random.choice([0, 1], 100),
    'AlcoholConsumption': np.random.choice([0, 1], 100),
    'PreviousHeartProblems': np.random.choice([0, 1], 100),
    'Triglycerides': np.random.randint(50, 400, 100),
    'StressLevel': np.random.randint(1, 10, 100),
    'Target': np.random.choice([0, 1], 100) # 0 = Saudável, 1 = Risco
}
df = pd.DataFrame(data)

# 2. Separar Features (X) e Alvo (y)
X = df.drop('Target', axis=1)
y = df['Target']

# 3. Definir o Pipeline de Pré-processamento
# Numéricos
features_num = ['Age', 'Cholesterol', 'HeartRate', 'Diabetes', 'FamilyHistory', 
                'Smoking', 'Obesity', 'AlcoholConsumption', 'PreviousHeartProblems', 
                'Triglycerides', 'StressLevel']
# Categóricos (Texto)
features_cat = ['Gender']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), features_num),
    ('cat', OneHotEncoder(handle_unknown='ignore'), features_cat)
])

# 4. Criar e Treinar o Pipeline Completo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X, y)

# 5. Salvar o arquivo .pkl
joblib.dump(pipeline, 'modelo_pipeline_final.pkl')
print("✅ Novo 'modelo_pipeline_final.pkl' criado com sucesso!")
print("Agora o seu back-end vai conseguir ler os dados do front-end.")
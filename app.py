import pandas as pd
import joblib
import os
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # Permite a conexão do HTML com o Python

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_FILE = 'modelo_pipeline_final.pkl'

# Carrega o modelo
model_pipeline = None
try:
    path = os.path.join(BASE_DIR, MODEL_FILE)
    if os.path.exists(path):
        model_pipeline = joblib.load(path)
        print(f"✅ Modelo carregado: {MODEL_FILE}")
    else:
        print(f"❌ ERRO: {MODEL_FILE} não encontrado. Rode o script de criação primeiro!")
except Exception as e:
    print(f"❌ Erro ao abrir modelo: {e}")

# --- A LISTA MÁGICA DE CONEXÃO ---
FEATURES_ORDER = [
    'Age', 
    'Gender', 
    'Cholesterol', 
    'HeartRate', 
    'Diabetes', 
    'FamilyHistory', 
    'Smoking', 
    'Obesity', 
    'AlcoholConsumption', 
    'PreviousHeartProblems', 
    'Triglycerides', 
    'StressLevel'
]

# Servir as páginas do Front-end
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/resultado")
def resultado():
    return send_from_directory(BASE_DIR, "resultado.html")

@app.route("/style.css")
def style():
    return send_from_directory(BASE_DIR, "style.css")

# Endpoint da Predição
@app.route('/predict', methods=['POST'])
def predict():
    if not model_pipeline:
        return jsonify({'error': 'Modelo não carregado no servidor.'}), 500

    try:
        # 1. Recebe os dados do HTML
        data = request.get_json(force=True)
        print("📥 Dados recebidos do Front:", data)

        # 2. Organiza os dados na ordem exata que o modelo exige
        input_data = {key: [data.get(key)] for key in FEATURES_ORDER}
        input_df = pd.DataFrame(input_data)

        # 3. Faz a previsão
        prediction_proba = model_pipeline.predict_proba(input_df)[0]
        prediction_class = model_pipeline.predict(input_df)[0]

        # Lógica de resposta (0 = Saudável, 1 = Risco)
        prob_risco = round(prediction_proba[1] * 100, 2)
        prob_saude = round(prediction_proba[0] * 100, 2)
        
        msg_final = "Alerta: Alto Risco Cardíaco Detectado" if prediction_class == 1 else "Resultado: Baixo Risco Cardíaco"

        print(f"📤 Resposta enviada: {msg_final} ({prob_risco}%)")

        # Aqui enviamos as chaves: 'probabilidade_risco' e 'probabilidade_nao_risco'
        return jsonify({
            'resultado': msg_final,
            'predicted_class': int(prediction_class),
            'probabilidade_risco': prob_risco,
            'probabilidade_nao_risco': prob_saude
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f"Erro no processamento: {str(e)}"}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
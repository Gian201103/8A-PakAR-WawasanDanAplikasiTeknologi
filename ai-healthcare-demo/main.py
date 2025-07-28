from modules.radiology_ai import AIRadiologyDiagnosis
from modules.chatbot_ai import MedicalChatbot
from modules.epidemic_ai import EpidemicPrediction

print("="*60)
print("SIMULASI PENERAPAN AI DALAM SISTEM KESEHATAN")
print("="*60)

# Demo Radiology AI
print("\n[1] DEMO DIAGNOSIS RADIOLOGI")
ai_radiology = AIRadiologyDiagnosis()
data = ai_radiology.generate_sample_data()
results = ai_radiology.train_model(data)
print(f"Akurasi Model: {results['accuracy']:.2f}")
diagnosis = ai_radiology.diagnose([0.8, 0.9, 0.7, 0.5, 0.3, 0.4])
print("Hasil Diagnosis:", diagnosis)

# Demo Chatbot
print("\n[2] DEMO CHATBOT MEDIS")
chatbot = MedicalChatbot()
response = chatbot.generate_response("Saya mengalami demam dan menggigil")
print("Chatbot:", response)

# Demo Prediksi Wabah
print("\n[3] DEMO PREDIKSI WABAH")
epidemic = EpidemicPrediction()
data = epidemic.generate_epidemic_data()
print(data.head())
risk = epidemic.predict_outbreak_risk(30, 85, 100, 50)
print("Prediksi Risiko Wabah:", risk)

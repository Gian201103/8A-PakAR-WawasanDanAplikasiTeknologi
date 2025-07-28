import datetime
import pandas as pd

class MedicalChatbot:
    def __init__(self):
        self.medical_knowledge = {
            'demam': {'symptoms': ['panas', 'menggigil'], 'advice': 'Minum air putih, istirahat cukup', 'urgency': 'sedang'},
            'batuk': {'symptoms': ['batuk', 'tenggorokan gatal'], 'advice': 'Minum air hangat', 'urgency': 'rendah'},
            'sesak napas': {'symptoms': ['napas pendek'], 'advice': 'Segera ke IGD!', 'urgency': 'tinggi'}
        }
        self.consultation_log = []

    def preprocess_input(self, user_input):
        user_input = user_input.lower()
        keywords = [cond for cond, data in self.medical_knowledge.items() if cond in user_input]
        return keywords

    def generate_response(self, user_input, patient_name="Pasien"):
        keywords = self.preprocess_input(user_input)
        if not keywords:
            return {'message': 'Gejala tidak dikenali', 'urgency': 'rendah'}
        primary = keywords[0]
        data = self.medical_knowledge[primary]
        self.consultation_log.append({'timestamp': datetime.datetime.now(), 'patient': patient_name, 'input': user_input})
        return {'message': f'Kemungkinan {primary}.', 'advice': data['advice'], 'urgency': data['urgency']}

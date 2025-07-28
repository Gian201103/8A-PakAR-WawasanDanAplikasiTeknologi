import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

class AIRadiologyDiagnosis:
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def generate_sample_data(self, n_samples=1000):
        np.random.seed(42)
        data = []
        labels = []
        for i in range(n_samples):
            condition = np.random.choice([0, 1, 2, 3], p=[0.4, 0.25, 0.2, 0.15])
            if condition == 0:
                features = [np.random.normal(0.3, 0.1), np.random.normal(0.5, 0.1),
                            np.random.normal(0.4, 0.1), np.random.normal(0.2, 0.05),
                            np.random.uniform(0, 1), np.random.uniform(0, 1)]
            elif condition == 1:
                features = [np.random.normal(0.7, 0.1), np.random.normal(0.8, 0.1),
                            np.random.normal(0.6, 0.1), np.random.normal(0.4, 0.1),
                            np.random.uniform(0, 1), np.random.uniform(0, 1)]
            elif condition == 2:
                features = [np.random.normal(0.9, 0.1), np.random.normal(0.9, 0.1),
                            np.random.normal(0.8, 0.1), np.random.normal(0.6, 0.15),
                            np.random.uniform(0, 1), np.random.uniform(0, 1)]
            else:
                features = [np.random.normal(0.1, 0.05), np.random.normal(0.9, 0.1),
                            np.random.normal(0.2, 0.05), np.random.normal(0.8, 0.1),
                            np.random.uniform(0, 1), np.random.uniform(0, 1)]
            features = [max(0, min(1, f)) for f in features]
            data.append(features)
            labels.append(condition)
        df = pd.DataFrame(data, columns=['density', 'contrast', 'texture', 'size', 'location_x', 'location_y'])
        df['condition'] = labels
        df['condition_name'] = df['condition'].map({0: 'Normal', 1: 'Pneumonia', 2: 'Tumor', 3: 'Fracture'})
        return df

    def train_model(self, data):
        X = data[['density', 'contrast', 'texture', 'size', 'location_x', 'location_y']]
        y = data['condition']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        y_pred = self.model.predict(X_test_scaled)
        return {
            'accuracy': self.model.score(X_test_scaled, y_test),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

    def diagnose(self, patient_data):
        if not self.is_trained:
            return "Model belum dilatih!"
        patient_scaled = self.scaler.transform([patient_data])
        prediction = self.model.predict(patient_scaled)[0]
        probability = self.model.predict_proba(patient_scaled)[0]
        conditions = ['Normal', 'Pneumonia', 'Tumor', 'Fracture']
        return {
            'diagnosis': conditions[prediction],
            'confidence': max(probability) * 100
        }

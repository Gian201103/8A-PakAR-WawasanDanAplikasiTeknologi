import numpy as np
import pandas as pd

class EpidemicPrediction:
    def generate_epidemic_data(self, days=30):
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=days)
        cases = np.random.randint(10, 100, size=days)
        return pd.DataFrame({'date': dates, 'cases': cases})

    def predict_outbreak_risk(self, temperature, humidity, rainfall, recent_cases):
        score = (temperature/40 + humidity/100 + rainfall/200 + recent_cases/100) / 4
        if score > 0.6: return {'risk': 'Tinggi', 'score': score}
        elif score > 0.3: return {'risk': 'Sedang', 'score': score}
        else: return {'risk': 'Rendah', 'score': score}

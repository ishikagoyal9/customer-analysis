from flask import Flask, request, jsonify, render_template
import pickle, numpy as np

app = Flask(__name__)

try:
    with open('churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Model and Scaler loaded!")
except Exception as e:
    print(f"Warning: {e}")
    model = None
    scaler = None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        d = request.get_json()
        internet = d['InternetService']
        contract = d['Contract']
        payment  = d['PaymentMethod']
        total    = float(d['TotalCharges'])
        
        features = [
            int(d['gender']),           
            int(d['SeniorCitizen']),    
            int(d['Partner']),           
            int(d['Dependents']),
            float(d['tenure']),
            int(d['PhoneService']),
            int(d['MultipleLines']),
            int(d['OnlineSecurity']),
            int(d['OnlineBackup']),
            int(d['DeviceProtection']),
            int(d['TechSupport']),
            int(d['StreamingTV']),
            int(d['StreamingMovies']),
            int(d['PaperlessBilling']),
            float(d['MonthlyCharges']),
            total,
            np.log1p(total),            
            1 if internet == 'Fiber optic' else 0,   
            1 if internet == 'No' else 0,           
            1 if contract == 'One year' else 0,       
            1 if contract == 'Two year' else 0,       
            1 if payment == 'Credit card (automatic)' else 0,
            1 if payment == 'Electronic check' else 0,
            1 if payment == 'Mailed check' else 0,
        ]

        arr = np.array([features])
        if scaler:
            arr = scaler.transform(arr)

        if model:
            pred = int(model.predict(arr)[0])
            prob = model.predict_proba(arr)[0].tolist() if hasattr(model, 'predict_proba') else [0.5, 0.5]
        else:
            pred, prob = 0, [0.72, 0.28]

        return jsonify({'success': True, 'churn': pred,
                        'prob_stay': round(prob[0]*100, 1),
                        'prob_churn': round(prob[1]*100, 1)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)

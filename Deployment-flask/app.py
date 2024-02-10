import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('classifier_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # Récupérer les valeurs du formulaire
    credit_history = request.form['Credit_History']
    married = request.form['Married']
    coapplicant_income = float(request.form['CoapplicantIncome'])

    # Convertir les valeurs de chaînes de caractères en valeurs numériques
    credit_history = 1 if credit_history == '1' else 0
    married = 1 if married == '1' else 0

    # Préparer les caractéristiques finales pour la prédiction
    final_features = [[credit_history, married, coapplicant_income]]

    # Prédire avec le modèle chargé
    prediction = model.predict(final_features)

    # Mapper les valeurs prédites à des libellés
    prediction_text = 'Approuvé' if prediction[0] == 1 else 'Non approuvé'

    return render_template('index.html', prediction_text='La réponse à votre requête est : {}'.format(prediction_text))



# @app.route('/predict', methods=['POST'])
# def predict():
#     '''
#     For rendering results on HTML GUI
#     '''
#     # Mapping des valeurs textuelles aux valeurs numériques
#     credit_history_map = {'No Credit History': 0, 'Has Credit History': 1}
#     married_map = {'Unmarried': 0, 'Married': 1}

#     # Récupération des valeurs sélectionnées dans le formulaire
#     credit_history = request.form['Credit_History']
#     married = request.form['Married']
#     coapplicant_income = int(request.form['CoapplicantIncome'])

#     # Conversion des valeurs textuelles en valeurs numériques
#     credit_history = credit_history_map.get(credit_history)
#     married = married_map.get(married)

#     # Préparation des caractéristiques finales pour la prédiction
#     final_features = [[credit_history, married, coapplicant_income]]

#     # Prédiction avec le modèle chargé
#     prediction = model.predict(final_features)

#     # Mapping des prédictions numériques à des labels textuels
#     prediction_text = 'Approved' if prediction[0] == 1 else 'Not Approved'

#     return render_template('index.html', prediction_text='La réponse à votre requête est : {}'.format(prediction_text))

if __name__ == "__main__":
    app.run(debug=True)
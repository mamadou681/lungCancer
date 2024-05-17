from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Загрузка модели
normalizer = joblib.load("normalizer.pkl")
model = joblib.load("model.pkl")

@app.route('/')
def my_form():
    return render_template('home.html', data="")

@app.route('/predict', methods=['POST'])
def predict():
    # Get the values of the comboboxes from the form data
    anxiety = request.form.get('anxiety')
    alcohol = request.form.get('alcohol')
    yellow_fingers = request.form.get('yellow_fingers')
    gender = request.form.get('gender')
    swallowing = request.form.get('swallowing')
    peer_pressure = request.form.get('peer_pressure')
    shortness_of_breath = request.form.get('shortness_of_breath')
    fatigue = request.form.get('fatigue')

    if(gender == "masculine"):
        gender = 1
    else:
        gender = 0


    # Создание DataFrame с указанными столбцами
    new_df = pd.DataFrame(columns=['ANXIETY', 'ALCOHOL CONSUMING', 'YELLOW_FINGERS', 'GENDER', 'SWALLOWING DIFFICULTY', 'PEER_PRESSURE',
                                'SHORTNESS OF BREATH', 'FATIGUE '],
                       data=[[transform_feature(anxiety),	transform_feature(alcohol),	 transform_feature(yellow_fingers),	gender,	transform_feature(swallowing),	transform_feature(peer_pressure),	transform_feature(shortness_of_breath),	transform_feature(fatigue)]])
    
    # Обрабатываем данные пред передачей в модель
    data_normalize = normalizer.transform(new_df) 
    # предсказание
    res = model.predict(data_normalize)
    # Ответ
    resMessage = res[0]

    if resMessage == 1:
        resMessage = "With the given data, the patient is diagnosed with lung cancer."
    else:
        resMessage = "With the given data, the patient is not diagnosed with lung cancer."

    dataToSend = [resMessage]

    return render_template('home.html', data=dataToSend)


def transform_feature(feature):
    if(str(feature) == "yes"):
        feature = 2
    else:
        feature = 1
    return feature

if __name__ == '__main__':
    app.run(debug=True)

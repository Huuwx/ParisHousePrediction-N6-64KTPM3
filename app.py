from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)

# Tải các mô hình
models = {}
model_names = ['Linear', 'Lasso', 'NeuralNetwork', 'Stacking']
for model_name in model_names:
    with open(f'Model/{model_name}Model', 'rb') as file:
        models[model_name] = pickle.load(file)

# Khởi tạo scaler và lấy giá trị mean và std từ mô hình đã huấn luyện
with open('scaler', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        squareMeters = float(request.form['squareMeters'])
        numbeOfRoom = float(request.form['numbeOfRoom'])
        hasYard = float(request.form['hasYard'])
        hasPool = float(request.form['hasPool'])
        floors = float(request.form['floors'])
        cityCode = float(request.form['cityCode'])
        cityPartRange = float(request.form['cityPartRange'])
        numPrevOwners = float(request.form['numPrevOwners'])
        made = float(request.form['made'])
        isNewBuilt = float(request.form['isNewBuilt'])
        hasStormProtector = float(request.form['hasStormProtector'])
        basement = float(request.form['basement'])
        attic = float(request.form['attic'])
        garage = float(request.form['garage'])
        hasStorageRoom = float(request.form['hasStorageRoom'])
        hasGuestRoom = float(request.form['hasGuestRoom'])
        category = float(request.form['category'])
        selected_model = request.form['model']

        input_data = np.array([[squareMeters, numbeOfRoom, hasYard, hasPool, floors, cityCode, cityPartRange, numPrevOwners, made, isNewBuilt, hasStormProtector, basement, attic, garage, hasStorageRoom, hasGuestRoom, category]])
        input_data_scaled = scaler.transform(input_data)

        # Dự đoán với mô hình đã chọn
        prediction = models[selected_model].predict(input_data_scaled)[0]


    return render_template('index.html', prediction=prediction, model_names=model_names)

if __name__ == '__main__':
    app.run(debug=True)

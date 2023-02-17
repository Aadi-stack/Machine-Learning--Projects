from flask import Flask, jsonify, render_template, request
import joblib
import os
import numpy as np
#import sklearn
import sklearn.preprocessing

app = Flask(__name__,template_folder='tempates')


@app.route("/")
def index():
    return render_template("build.html")




@app.route('/predict', methods=['POST', 'GET'])
def result():

    item_weight = float(request.form['item_weight'])
    item_fat_content = float(request.form['item_fat_content'])
    item_visibility = float(request.form['item_visibility'])
    item_type = float(request.form['item_type'])
    item_mrp = float(request.form['item_mrp'])
    outlet_establishment_year = float(request.form['outlet_establishment_year'])
    outlet_size = float(request.form['outlet_size'])
    outlet_location_type = float(request.form['outlet_location_type'])
    outlet_type = float(request.form['outlet_type'])

    X = np.array([[item_weight, item_fat_content, item_visibility, item_type, item_mrp,
                   outlet_establishment_year, outlet_size, outlet_location_type, outlet_type]])

    
    loaded_model = joblib.load(open(r'C:\Users\DELLS\OneDrive\Documents\Downloads\sc.sav','rb'))

  

    X_std = loaded_model.transform(X)


    model = joblib.load(open(r'C:\Users\DELLS\OneDrive\Documents\Downloads\rf.sav', 'rb'))


    Y_pred = model.predict(X_std)

    my_pred= Y_pred
    
    return render_template("build.html",my_marks=my_pred)


    #return jsonify({'Prediction': float(my_pred)})


if __name__ == "__main__":
    app.run(debug=True, port=9457)
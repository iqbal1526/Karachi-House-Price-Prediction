#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, render_template, jsonify
import pickle
import json
from flask_cors import cross_origin
import pandas as pd
# In[2]:


app = Flask(__name__)

# Load the trained model from the pickle file
with open('housingProject.pickle', 'rb') as f:
    model = pickle.load(f)

# Load the column names used during training
with open('columns.json', 'r') as f:
    columns = json.load(f)['data_columns']



feature_mapping = {
    'bahria town': 'Bahria Town',
    'clifton': 'Clifton',
    'dha': 'DHA',
    'flat': 'Flat',
    'house': 'House'
}




@app.route("/")
@cross_origin()
def home():
    return render_template("index.html")




@app.route('/predict', methods=["GET",'POST'])
#@cross_origin()

def predict():
    # Get the input data from the request
    if request.method == "POST":

        # Preprocess the input data if necessary
        # ... add preprocessing code here ...
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        area = int(request.form['area'])
        
        location = request.form['location']
        bahria_town = 1 if location == 'bahria town' else 0
        dha = 1 if location == 'dha' else 0
        clifton = 1 if location == 'clifton' else 0
        
        property_type = request.form['houseFlat']
        house = 1 if property_type == 'house' else 0
        flat = 1 if property_type == 'flat' else 0

        
        
        # Create a dataframe from the input data using the column names
        input_data = pd.DataFrame([[bedrooms, bathrooms, area, flat, house, 
                                bahria_town, dha, clifton]],
                              columns=columns[1:])

        input_data.rename(columns=feature_mapping, inplace=True)

        # Generate the predicted price using the loaded model
        predictedPrice = round(model.predict(input_data)[0],2)

        # Return the predicted price as a JSON response
        return jsonify({'prediction_text': "The house price is Rs. {} crore".format(predictedPrice)})
    return render_template("index.html")

if __name__ == '__main__':
    app.run()


# In[ ]:





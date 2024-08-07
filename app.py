import os
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
from datetime import datetime
from jinja2 import Environment
import pickle

#crop_model
crop_model = pickle.load(open('random.pkl','rb'))

# ChatBot
# -----------------------------------------------------------------------------------------------------------------------------
# Load environment variables

load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')

# Configure Google Generative AI
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])
# ----------------------------------------------------------------------------------------------------------------------------------

# Creating Flask app and linking it to Mongo_DB and creating login credentials
app = Flask(__name__)
app.secret_key = "your_secret_key"

app.config["MONGO_URI"] = "mongodb://localhost:27017/CFRS_db"   #//configuring mangodb
mongo = PyMongo(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, username, password, id=None):
        self.username = username
        self.password = password
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    user_data = mongo.db.users.find_one({"_id": ObjectId(user_id)})
    if user_data:
        return User(username=user_data['username'], password=user_data['password'], id=str(user_data['_id']))
    return None

# Jinja environment with the `str` function
env = Environment()
env.filters['str'] = str

# Define the zip_lists function
def zip_lists(list1, list2):
    return zip(list1, list2)

# Register the zip_lists function as a custom filter named 'zip'
app.jinja_env.filters['zip'] = zip_lists

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        users = mongo.db.users
        login_user_data = users.find_one({'username': request.form['username']})

        if login_user_data and check_password_hash(login_user_data['password'], request.form['password']):
            user = User(username=login_user_data['username'], password=login_user_data['password'], id=str(login_user_data['_id']))
            login_user(user)
            return redirect(url_for('index'))

        return 'Invalid username/password combination'

    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    if request.method == 'POST':
        users = mongo.db.users
        existing_user = users.find_one({'username': request.form['username']})

        if existing_user is None:
            hashpass = generate_password_hash(request.form['password'])
            user_id = users.insert_one({'username': request.form['username'], 'password': hashpass}).inserted_id
            user = User(username=request.form['username'], password=hashpass, id=str(user_id))
            login_user(user)
            return redirect(url_for('index'))

        return 'That username already exists!'

    return redirect(url_for('index'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# Route for rendering the chat page
@app.route('/chat')
def chat_page():
    return render_template('chat.html')

# Route for handling chat messages
@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    question = data['message']
    response = chat.send_message(question)
    return jsonify(reply=response.text)

# Crop recommendation model
# ==================================================================================================================
@app.route('/crop')
def crop_recommendation():
    return render_template("crop.html")

@app.route("/predict", methods=['POST'])
def predict():
    N = request.form.get('Nitrogen')
    P = request.form.get('Phosporus')
    K = request.form.get('Potassium')
    temp = request.form.get('Temperature')
    humidity = request.form.get('Humidity')
    ph = request.form.get('Ph')
    rainfall = request.form.get('Rainfall')
    data = np.array([[N, P,K,temp, humidity, ph, rainfall]])
# Making predictions
    prediction = crop_model.predict(data)
    ans=prediction[0]
    store_prediction(current_user.id, ans, temp, humidity, ph, rainfall,N,P,K)
    return render_template('crop.html', result=ans)

    
    

# ============================================================================================================

# history

@app.route('/history')
@login_required
def history():
    user_id = ObjectId(current_user.id)
    crop_predictions = fetch_prediction_history(user_id)
    fertilizer_predictions = fetch_fertilizer_prediction_history(user_id)
    return render_template('history.html', crop_predictions=crop_predictions, fertilizer_predictions=fertilizer_predictions)


@app.route('/delete_prediction/<prediction_id>/<prediction_type>', methods=['GET'])
@login_required
def delete_prediction(prediction_id, prediction_type):
    try:
        if prediction_type == 'crop':
            del_history(prediction_id)
        elif prediction_type == 'fertilizer':
            del_fertilizer_history(prediction_id)
        return redirect(url_for('history'))
    except Exception as e:
        return str(e)


def del_history(prediction_id):
    user_id = ObjectId(current_user.id)
    mongo.db.crop_prediction_history.delete_one({'_id': ObjectId(prediction_id), 'user_id': user_id})
def del_fertilizer_history(prediction_id):
    user_id = ObjectId(current_user.id)
    mongo.db.fertilizer_prediction_history.delete_one({'_id': ObjectId(prediction_id), 'user_id': user_id})



# Fertilizer recommendation model
# ============================================================================================================
@app.route('/fertilizer')
def fertilizer_recommendation():
    return render_template("fertilizer.html")

@app.route("/fertilizer_predict", methods=['POST'])
def fertilizer_predict():
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    moisture = float(request.form['Moisture'])
    soil_type = request.form['Soil_Type']
    crop_type = request.form['Crop_Type']
    nitro = float(request.form['Nitrogen'])
    pot = float(request.form['Potassium'])
    phosp = float(request.form['Phosphorous'])

    # Encode categorical variables
    soil_type_encoded = encode_soil.transform([soil_type])[0]
    crop_type_encoded = encode_crop.transform([crop_type])[0]

    # Prepare input data
    input_data = pd.DataFrame([[temp, humidity, moisture, soil_type_encoded, crop_type_encoded, nitro, pot, phosp]], 
                              columns=['Temperature', 'Humidity', 'Moisture', 'Soil_Type', 'Crop_Type', 'Nitrogen', 'Potassium', 'Phosphorous'])

    # Predict fertilizer
    fertilizer_prediction = fertilizer_model.predict(input_data)
    predicted_fertilizer = encode_ferti.inverse_transform(fertilizer_prediction)

     # Store the prediction data
    store_fertilizer_prediction(current_user.id, predicted_fertilizer[0], temp, humidity, moisture, soil_type, crop_type, nitro, pot, phosp)
    return render_template('fertilizer.html', result=predicted_fertilizer[0])

# Fertilizer recommendation data preparation
fertilizer_data = pd.read_csv("Fertilizer Prediction.csv")
fertilizer_data.rename(columns={'Humidity ': 'Humidity', 'Soil Type': 'Soil_Type', 'Crop Type': 'Crop_Type', 'Fertilizer Name': 'Fertilizer'}, inplace=True)

# Ensure that the 'Temperature' column is correctly named
fertilizer_data.rename(columns={'Temparature': 'Temperature'}, inplace=True)

encode_soil = LabelEncoder()
fertilizer_data.Soil_Type = encode_soil.fit_transform(fertilizer_data.Soil_Type)

encode_crop = LabelEncoder()
fertilizer_data.Crop_Type = encode_crop.fit_transform(fertilizer_data.Crop_Type)

encode_ferti = LabelEncoder()
fertilizer_data.Fertilizer = encode_ferti.fit_transform(fertilizer_data.Fertilizer)

x_fertilizer = fertilizer_data.drop('Fertilizer', axis=1)
y_fertilizer = fertilizer_data.Fertilizer

x_train_fertilizer, x_test_fertilizer, y_train_fertilizer, y_test_fertilizer = train_test_split(x_fertilizer, y_fertilizer, test_size=0.2, random_state=1)

fertilizer_model = RandomForestClassifier()
fertilizer_model.fit(x_train_fertilizer, y_train_fertilizer)

# =======================================================================================================================

# Utility functions
def store_prediction(user_id, crop_prediction, temperature, humidity, ph, rainfall,N,P,K):
    current_time = datetime.now()
    mongo.db.crop_prediction_history.insert_one({
        'user_id': ObjectId(user_id),
        'crop_prediction': crop_prediction,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall,
        'timestamp': current_time,
        'N':N,
        'P':P,
        'K':K
    })

def fetch_prediction_history(user_id):
    crop_predictions = mongo.db.crop_prediction_history.find({"user_id": ObjectId(user_id)})
    history = []
    for prediction in crop_predictions:
        crop_data = {
            'crop_predictions': prediction.get('crop_prediction','N/A'),
            'temperature': prediction.get('temperature', 'N/A'),
            'humidity': prediction.get('humidity', 'N/A'),
            'ph': prediction.get('ph', 'N/A'),
            'rainfall': prediction.get('rainfall', 'N/A'),
            'timestamp': prediction['timestamp'],
            'N':prediction.get('N'),
            'P':prediction.get('P'),
            'K':prediction.get('K'),
            '_id': str(prediction['_id'])  # Ensure the _id is included and converted to string
        }
        history.append(crop_data)
    return history

def store_fertilizer_prediction(user_id, fertilizer_prediction, temperature, humidity, moisture, soil_type, crop_type, N, P, K):
    current_time = datetime.now()
    mongo.db.fertilizer_prediction_history.insert_one({
        'user_id': ObjectId(user_id),
        'fertilizer_prediction': fertilizer_prediction,
        'temperature': temperature,
        'humidity': humidity,
        'moisture': moisture,
        'soil_type': soil_type,
        'crop_type': crop_type,
        'timestamp': current_time,
        'N': N,
        'P': P,
        'K': K
    })
def fetch_fertilizer_prediction_history(user_id):
    fertilizer_predictions = mongo.db.fertilizer_prediction_history.find({"user_id": ObjectId(user_id)})
    history = []
    for prediction in fertilizer_predictions:
        fertilizer_data = {
            'fertilizer_predictions': prediction.get('fertilizer_prediction', 'N/A'),
            'temperature': prediction.get('temperature', 'N/A'),
            'humidity': prediction.get('humidity', 'N/A'),
            'moisture': prediction.get('moisture', 'N/A'),
            'soil_type': prediction.get('soil_type', 'N/A'),
            'crop_type': prediction.get('crop_type', 'N/A'),
            'timestamp': prediction['timestamp'],
            'N': prediction.get('N'),
            'P': prediction.get('P'),
            'K': prediction.get('K'),
            '_id': str(prediction['_id'])  # Ensure the _id is included and converted to string
        }
        history.append(fertilizer_data)
    return history



# Main function to run the Flask app
if __name__ == "__main__":
    app.run(debug=True)

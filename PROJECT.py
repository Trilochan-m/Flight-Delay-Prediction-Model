from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier

app = Flask(__name__)

# ---------------------------
# LOAD DATA
# ---------------------------

df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\project\flight_schedule.csv")
dd = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\project\weather_2026.csv")

# Encode Status

status_encoder = LabelEncoder()
df['Status'] = status_encoder.fit_transform(df['Status'])

# Weather mapping

weather_map = {
    'Sunny': 0,
    'Clear': 1,
    'Rainy': 2,
    'Cloudy': 3,
    'Windy': 4,
    'Foggy': 5,
    'Snowy': 6,
    'Stormy': 7
}

df['Weather'] = df['Weather'].map(weather_map)
dd['Weather_Condition'] = dd['Weather_Condition'].map(weather_map)

# ---------------------------
# TRAIN MODEL
# ---------------------------

X = df[['Flight_ID', 'Weather']]
y = df['Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# ---------------------------
# ROUTES
# ---------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        fid = int(request.form['flight_id'])
        date = request.form['date']

        # Get weather for selected date
        weather_row = dd[dd['Date'] == date]

        if weather_row.empty:
            return render_template("index.html", prediction_text="Date not found!")

        condition = weather_row.iloc[0]['Weather_Condition']

        new_data = np.array([[fid, condition]])
        new_data_scaled = scaler.transform(new_data)

        prediction = model.predict(new_data_scaled)

        if prediction[0] == 1:
            result = "ON TIME"
        else:
            result = "DELAYED"

        return render_template("index.html", prediction_text=result)
    
    
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
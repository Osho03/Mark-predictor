from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# 🔁 Load the trained model
model = pickle.load(open("model.pkl", "rb"))
print("✅ Model loaded successfully.")

# 🏠 Home route
@app.route("/")
def home():
    return render_template("index.html")

# 🎯 Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    hours = float(data["hours"])
    sleep = float(data["sleep"])
    focus = float(data["focus"])
    distraction = float(data["distraction"])

    # 📈 Make prediction
    prediction = model.predict([[hours, sleep, focus, distraction]])

    return jsonify({"predicted_marks": prediction[0]})

if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    app.run(debug=True)
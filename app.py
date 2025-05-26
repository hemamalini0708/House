from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("linear_regression_model.pkl")


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data as JSON
        data = request.get_json()

        # Convert JSON into DataFrame
        df = pd.DataFrame([data])

        # Ensure all required features are present
        required_features = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view",
                             "condition", "grade", "sqft_above", "sqft_basement", "yr_built", "yr_renovated", "zipcode",
                             "lat", "long", "sqft_living15", "sqft_lot15"]
        for feature in required_features:
            if feature not in df.columns:
                return jsonify({"error": f"Missing feature: {feature}"})

        # Predict using the loaded model
        prediction = model.predict(df)

        # Return the prediction as JSON
        return jsonify({"predicted_price": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
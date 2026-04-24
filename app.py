
# ------------------ IMPORTS ------------------
import os
import warnings
import pickle
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, request, render_template, jsonify, redirect, url_for
from flask_login import LoginManager, login_user, login_required, logout_user

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models import users, User
from utils import generate_inventory_report, get_low_stock_products, get_near_expiry_products

# ------------------ CONFIG ------------------
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = "inventory_secret_key_123"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'data_set')
app.config['MODEL_PATH'] = os.path.join(BASE_DIR, 'trained_model.pkl')
app.config['DATA_PATH'] = os.path.join(app.config['UPLOAD_FOLDER'], 'data.csv')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# ------------------ LOGIN SETUP ------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    for user in users.values():
        if str(user.id) == user_id:
            return user
    return None

# ------------------ MODEL LOAD (SAFE) ------------------
def load_trained_model():
    try:
        with open(app.config['MODEL_PATH'], 'rb') as f:
            model_data = pickle.load(f)

        if hasattr(model_data, 'predict'):
            print("✅ Model loaded")
            return model_data
        elif isinstance(model_data, dict) and 'model' in model_data:
            print("✅ Model loaded from dict")
            return model_data['model']
        else:
            print("⚠️ Invalid model format")
            return None

    except FileNotFoundError:
        print("⚠️ Model file not found")
        return None
    except Exception as e:
        print("❌ Model load error:", e)
        return None

model = load_trained_model()

# ------------------ SIMPLE PREDICTION ------------------
def simple_prediction(q1, q2, q3):
    return (0.2*q1 + 0.3*q2 + 0.5*q3)

# ------------------ ROUTES ------------------

@app.route("/")
@login_required
def home():
    return render_template("index.html")

# ------------------ LOGIN ------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        user = users.get(username)

        if user and user.password == password:
            login_user(user)
            return redirect(url_for("home"))

        return "Invalid username or password"

    return render_template("login.html")

# ------------------ LOGOUT ------------------
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# ------------------ FILE UPLOAD ------------------
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file part"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"success": False, "error": "No selected file"}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({"success": False, "error": "Upload CSV only"}), 400

        file_path = app.config['DATA_PATH']
        file.save(file_path)

        return jsonify({"success": True, "message": "Uploaded successfully"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ------------------ INVENTORY ------------------
@app.route('/inventory')
def inventory():
    try:
        if not os.path.exists(app.config['DATA_PATH']):
            return render_template('error.html', error="Upload CSV first")

        df = pd.read_csv(app.config['DATA_PATH'])

        low_stock = get_low_stock_products(df)
        near_expiry = get_near_expiry_products(df)

        from utils import calculate_inventory_metrics
        metrics = calculate_inventory_metrics(df)

        return render_template('inventory.html',
                               restock_recommendations=low_stock,
                               near_expiry_recommendations=near_expiry,
                               metrics=metrics)

    except Exception as e:
        return render_template('error.html', error=str(e))

# ------------------ PREDICTION ------------------
@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            data = request.get_json()

            q1 = float(data.get('quantity1', 0))
            q2 = float(data.get('quantity2', 0))
            q3 = float(data.get('quantity3', 0))

            if model:
                try:
                    pred = model.predict(np.array([[q1, q2, q3]]))[0][0]
                except:
                    pred = simple_prediction(q1, q2, q3)
            else:
                pred = simple_prediction(q1, q2, q3)

            return jsonify({"success": True, "prediction": float(pred)})

        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

    return render_template("prediction.html")

# ------------------ ANALYTICS ------------------
@app.route('/analytics')
def analytics():
    try:
        if not os.path.exists(app.config['DATA_PATH']):
            return render_template('error.html', error="Upload CSV first")

        df = pd.read_csv(app.config['DATA_PATH'])

        total_sales = float(df["total_revenue"].sum())
        avg_order = float(df["total_revenue"].mean())

        top = df.nlargest(5, "quantity_stock")
        bottom = df.nsmallest(5, "quantity_stock")

        fig, ax = plt.subplots()
        ax.plot(df['total_revenue'])
        fig.savefig("static/sales.png")
        plt.close()

        return render_template('analytics.html',
                               total_sales=total_sales,
                               average_order_value=avg_order,
                               top_selling_products=top.to_dict('records'),
                               bottom_selling_products=bottom.to_dict('records'))

    except Exception as e:
        return render_template('error.html', error=str(e))

# ------------------ INVENTORY API ------------------
@app.route('/api/inventory-summary')
def inventory_summary():
    try:
        if not os.path.exists(app.config['DATA_PATH']):
            return jsonify({"metrics": {}})

        report = generate_inventory_report(app.config['DATA_PATH'])
        return jsonify(report)

    except Exception as e:
        return jsonify({"error": str(e)})

# ------------------ MAIN ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
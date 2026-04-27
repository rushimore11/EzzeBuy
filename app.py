# ------------------ IMPORTS ------------------
import os
import warnings
import pickle
import numpy as np
import pandas as pd

from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash

from models import users, User, save_users
from utils import (
    generate_inventory_report,
    get_low_stock_products,
    get_near_expiry_products,
    calculate_inventory_metrics
)

warnings.filterwarnings('ignore')

# ------------------ APP CONFIG ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = "inventory_secret_key_123"

app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'data_set')
app.config['MODEL_PATH'] = os.path.join(BASE_DIR, 'trained_model.pkl')
app.config['DATA_PATH'] = os.path.join(app.config['UPLOAD_FOLDER'], 'data.csv')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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

# ------------------ MODEL LOAD ------------------
def load_model():
    try:
        with open(app.config['MODEL_PATH'], 'rb') as f:
            model = pickle.load(f)
        print("✅ Model Loaded")
        return model
    except:
        print("⚠️ Model not found, using fallback")
        return None

model = load_model()

def simple_prediction(q1, q2, q3):
    return (0.2*q1 + 0.3*q2 + 0.5*q3)

# ------------------ ROUTES ------------------

@app.route("/")
@login_required
def home():
    return render_template("index.html")

# ------------------ AUTH ------------------

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if len(username) < 3:
            flash("Username too short", "error")
            return render_template("signup.html")

        if len(password) < 6:
            flash("Password too short", "error")
            return render_template("signup.html")

        if username in users:
            flash("User already exists", "error")
            return render_template("signup.html")

        uid = max((u.id for u in users.values()), default=0) + 1
        users[username] = User(uid, username, generate_password_hash(password))
        save_users(users)

        flash("Account created. Login now.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = users.get(username)

        if user and user.verify_password(password):
            login_user(user)
            return redirect(url_for("home"))

        flash("Invalid username or password", "error")

    return render_template("login.html")


@app.route("/logout", methods=["GET", "POST"])
@login_required
def logout():
    if request.method == "POST":
        logout_user()
        flash("Logged out successfully", "success")
        return redirect(url_for("login"))

    return render_template("logout.html")

# ------------------ FILE UPLOAD ------------------

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    try:
        file = request.files.get('file')

        if not file:
            return jsonify({"success": False, "error": "No file uploaded"})

        if not file.filename.endswith('.csv'):
            return jsonify({"success": False, "error": "Upload CSV only"})

        file.save(app.config['DATA_PATH'])
        return jsonify({"success": True, "message": "File uploaded successfully"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ------------------ INVENTORY PAGE ------------------

@app.route('/inventory')
@login_required
def inventory():
    try:
        if not os.path.exists(app.config['DATA_PATH']):
            return render_template('error.html', error="Upload CSV first")

        df = pd.read_csv(app.config['DATA_PATH'])

        return render_template(
            'inventory.html',
            restock_recommendations=get_low_stock_products(df),
            near_expiry_recommendations=get_near_expiry_products(df),
            metrics=calculate_inventory_metrics(df)
        )

    except Exception as e:
        return render_template('error.html', error=str(e))

# ------------------ ANALYTICS ------------------

@app.route('/analytics')
@login_required
def analytics():
    try:
        if not os.path.exists(app.config['DATA_PATH']):
            return render_template('error.html', error="Upload CSV first")

        df = pd.read_csv(app.config['DATA_PATH'])

        total_sales = float(df["total_revenue"].sum())
        avg_order = float(df["total_revenue"].mean())

        return render_template(
            'analytics.html',
            total_sales=total_sales,
            average_order_value=avg_order
        )

    except Exception as e:
        return render_template('error.html', error=str(e))

# ------------------ PREDICTION ------------------

@app.route('/predict', methods=["GET", "POST"])
@login_required
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

# ------------------ API ------------------

@app.route('/api/inventory-summary')
@login_required
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

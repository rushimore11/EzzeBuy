# ------------------ IMPORTS ------------------
import os
import sys
import warnings
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, request, render_template, jsonify, redirect, url_for, flash, send_from_directory
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.exceptions import BadRequest

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add models and utils imports - ensure these files exist
try:
    from models import users, User, save_users
    from utils import generate_inventory_report, get_low_stock_products, get_near_expiry_products, calculate_inventory_metrics
except ImportError as e:
    print(f"❌ Missing required files: {e}")
    sys.exit(1)

# ------------------ PRODUCTION CONFIG ------------------
warnings.filterwarnings('ignore')

# Flask app setup
app = Flask(__name__)

# Production security settings
app.secret_key = os.environ.get('SECRET_KEY', 'inventory_secret_key_123_PROD')
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour

# File paths - Render compatible
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Create directories
for directory in [DATA_DIR, STATIC_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)

app.config.update({
    'UPLOAD_FOLDER': DATA_DIR,
    'MODEL_PATH': os.path.join(MODEL_DIR, 'trained_model.pkl'),
    'DATA_PATH': os.path.join(DATA_DIR, 'data.csv'),
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB max file size
})

# ------------------ PRODUCTION LOGGING ------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(DATA_DIR, 'app.log'), mode='a')
    ]
)
logger = logging.getLogger(__name__)

# ------------------ LOGIN MANAGER ------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access this page."
login_manager.login_message_category = "warning"

@login_manager.user_loader
def load_user(user_id):
    try:
        for user in users.values():
            if str(user.id) == user_id:
                return user
        return None
    except Exception as e:
        logger.error(f"Error loading user {user_id}: {e}")
        return None

# ------------------ MODEL LOADING (PRODUCTION SAFE) ------------------
def load_trained_model():
    """Safely load trained model with fallback"""
    model_path = app.config['MODEL_PATH']
    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            if hasattr(model_data, 'predict'):
                logger.info("✅ Production model loaded successfully")
                return model_data
            elif isinstance(model_data, dict) and 'model' in model_data:
                logger.info("✅ Production model loaded from dictionary")
                return model_data['model']
        
        logger.warning("⚠️ No valid model found, using fallback")
        return None
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return None

# Global model instance
model = load_trained_model()

def simple_prediction(q1, q2, q3):
    """Fallback prediction function"""
    return float(0.2*q1 + 0.3*q2 + 0.5*q3)

# ------------------ HELPER FUNCTIONS ------------------
def validate_csv_file(file):
    """Validate uploaded CSV file"""
    if not file.filename.endswith('.csv'):
        return False, "Only CSV files are allowed"
    
    if file.filename == '':
        return False, "No file selected"
    
    try:
        # Quick validation - read first few rows
        df = pd.read_csv(file)
        if df.empty:
            return False, "CSV file is empty"
        return True, "Valid file"
    except:
        return False, "Invalid CSV format"

# ------------------ ROUTES ------------------

@app.route("/")
@login_required
def home():
    """Main dashboard"""
    return render_template("index.html")

# ------------------ AUTHENTICATION ROUTES ------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        try:
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            confirm_password = request.form.get("confirm_password", "")

            # Validation
            if len(username) < 3:
                flash("Username must be at least 3 characters long.", "error")
                return render_template("signup.html")

            if len(password) < 6:
                flash("Password must be at least 6 characters long.", "error")
                return render_template("signup.html")

            if password != confirm_password:
                flash("Passwords do not match.", "error")
                return render_template("signup.html")

            if username in users:
                flash("Username already exists.", "error")
                return render_template("signup.html")

            # Create user
            next_user_id = max((u.id for u in users.values()), default=0) + 1
            new_user = User(
                id=next_user_id, 
                username=username, 
                password_hash=generate_password_hash(password)
            )
            users[username] = new_user
            save_users(users)

            logger.info(f"New user registered: {username}")
            flash("Account created successfully. Please sign in.", "success")
            return redirect(url_for("login"))

        except Exception as e:
            logger.error(f"Signup error: {e}")
            flash("Registration failed. Please try again.", "error")

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    if request.method == "POST":
        try:
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")

            user = users.get(username)
            if user and user.verify_password(password):
                login_user(user)
                logger.info(f"User logged in: {username}")
                return redirect(url_for("home"))
            
            flash("Invalid username or password.", "error")
            logger.warning(f"Failed login attempt for: {username}")

        except Exception as e:
            logger.error(f"Login error: {e}")
            flash("Login failed. Please try again.", "error")

    return render_template("login.html")

@app.route("/logout", methods=["GET", "POST"])
@login_required
def logout():
    if request.method == "POST":
        username = current_user.username
        logout_user()
        logger.info(f"User logged out: {username}")
        flash("You have been logged out successfully.", "success")
        return redirect(url_for("login"))
    return render_template("logout.html")

# ------------------ FILE UPLOAD (PRODUCTION SECURE) ------------------
@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file part"}), 400

        file = request.files['file']
        is_valid, message = validate_csv_file(file)
        
        if not is_valid:
            return jsonify({"success": False, "error": message}), 400

        # Secure save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data_{timestamp}.csv"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file.save(filepath)
        
        # Update data path to latest file
        app.config['DATA_PATH'] = filepath
        
        logger.info(f"File uploaded by {current_user.username}: {filename}")
        return jsonify({
            "success": True, 
            "message": "File uploaded successfully",
            "filename": filename
        })

    except BadRequest:
        return jsonify({"success": False, "error": "File too large"}), 413
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"success": False, "error": "Upload failed"}), 500

# Serve static files
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)

# ------------------ INVENTORY ------------------
@app.route('/inventory')
@login_required
def inventory():
    try:
        data_path = app.config['DATA_PATH']
        if not os.path.exists(data_path):
            return render_template('error.html', error="Please upload a CSV file first")

        df = pd.read_csv(data_path)
        
        if df.empty:
            return render_template('error.html', error="Uploaded CSV is empty")

        low_stock = get_low_stock_products(df)
        near_expiry = get_near_expiry_products(df)
        metrics = calculate_inventory_metrics(df)

        return render_template('inventory.html',
                             restock_recommendations=low_stock,
                             near_expiry_recommendations=near_expiry,
                             metrics=metrics)

    except Exception as e:
        logger.error(f"Inventory error: {e}")
        return render_template('error.html', error=f"Error loading inventory: {str(e)}")

# ------------------ PREDICTION ------------------
@app.route('/predict', methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        try:
            data = request.get_json()
            if not data:
                return jsonify({"success": False, "error": "No data provided"}), 400

            q1 = float(data.get('quantity1', 0))
            q2 = float(data.get('quantity2', 0))
            q3 = float(data.get('quantity3', 0))

            if model:
                try:
                    pred = model.predict(np.array([[q1, q2, q3]]))[0][0]
                except Exception:
                    pred = simple_prediction(q1, q2, q3)
            else:
                pred = simple_prediction(q1, q2, q3)

            logger.info(f"Prediction made by {current_user.username}: {pred}")
            return jsonify({"success": True, "prediction": float(pred)})

        except (ValueError, KeyError) as e:
            return jsonify({"success": False, "error": "Invalid input data"}), 400
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return jsonify({"success": False, "error": "Prediction failed"}), 500

    return render_template("prediction.html")

# ------------------ ANALYTICS ------------------
@app.route('/analytics')
@login_required
def analytics():
    try:
        data_path = app.config['DATA_PATH']
        if not os.path.exists(data_path):
            return render_template('error.html', error="Please upload a CSV file first")

        df = pd.read_csv(data_path)
        if df.empty:
            return render_template('error.html', error="No data available")

        # Calculate metrics
        total_sales = float(df["total_revenue"].sum() if "total_revenue" in df.columns else 0)
        avg_order = float(df["total_revenue"].mean() if "total_revenue" in df.columns else 0)

        # Top/Bottom products
        top = df.nlargest(5, "quantity_stock").to_dict('records') if "quantity_stock" in df.columns else []
        bottom = df.nsmallest(5, "quantity_stock").to_dict('records') if "quantity_stock" in df.columns else []

        # Generate chart
        try:
            plt.figure(figsize=(10, 6))
            if "total_revenue" in df.columns:
                plt.plot(df['total_revenue'].values)
                plt.title('Sales Trend')
                plt.xlabel('Orders')
                plt.ylabel('Revenue')
                plt.tight_layout()
                chart_path = os.path.join(STATIC_DIR, 'sales.png')
                plt.savefig(chart_path, dpi=100, bbox_inches='tight')
                plt.close()
            else:
                chart_path = None
        except Exception as e:
            logger.warning(f"Chart generation failed: {e}")
            chart_path = None

        return render_template('analytics.html',
                             total_sales=total_sales,
                             average_order_value=avg_order,
                             top_selling_products=top,
                             bottom_selling_products=bottom,
                             chart_exists=chart_path is not None)

    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return render_template('error.html', error=f"Analytics error: {str(e)}")

# ------------------ API ENDPOINTS ------------------
@app.route('/api/inventory-summary')
@login_required
def inventory_summary():
    try:
        data_path = app.config['DATA_PATH']
        if not os.path.exists(data_path):
            return jsonify({"metrics": {}, "message": "No data uploaded"})

        report = generate_inventory_report(data_path)
        return jsonify(report)

    except Exception as e:
        logger.error(f"API inventory summary error: {e}")
        return jsonify({"error": "Failed to generate report"}), 500

# ------------------ HEALTH CHECK ------------------
@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/ready')
def ready_check():
    return jsonify({"status": "ready"})

# ------------------ ERROR HANDLERS ------------------
@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 error: {error}")
    return render_template('error.html', error="Internal server error"), 500

# ------------------ MAIN ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    logger.info(f"🚀 Starting Inventory Management System on {host}:{port}")
    logger.info(f"📁 Data directory: {DATA_DIR}")
    logger.info(f"📊 Model loaded: {'Yes' if model else 'No (using fallback)'}")
    
    app.run(host=host, port=port, debug=False, threaded=True)

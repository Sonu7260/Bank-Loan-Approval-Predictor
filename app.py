import os
import sys
import pickle
import traceback
import numpy as np
from flask import Flask, request, render_template_string

# --- AUTOMATED VERSION COMPATIBILITY PATCH ---
# This prevents 'No module named _loss' errors on legacy server environments
try:
    import sklearn.ensemble._gb_losses
except ModuleNotFoundError:
    try:
        # Trick the pickle loader into routing newer loss queries to the correct fallback paths
        import sklearn._loss as dummy_loss
        sys.modules['sklearn.ensemble._loss'] = dummy_loss
    except ImportError:
        pass

app = Flask(__name__)

# File name lookup mapping
MODEL_FILENAME = 'gradient_pkl (1).pkl'
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)

model = None
load_error_message = None

# Safe loader wrapper
try:
    if not os.path.exists(MODEL_PATH):
        load_error_message = f"File missing. Model must be placed at: '{MODEL_PATH}'."
    else:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded into system memory seamlessly.")
except Exception as e:
    load_error_message = f"Library Configuration Error: {str(e)}.\n\nTo permanently resolve this, add 'scikit-learn>=1.4.0' to your requirements.txt file so the server updates its framework packages."

# UI Interface Code Block
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bank Credit Processing System</title>
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            --card-bg: #ffffff;
            --primary-color: #2563eb;
            --primary-hover: #1d4ed8;
            --text-dark: #1e293b;
            --border-color: #e2e8f0;
            --success: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', sans-serif; }
        body { background: var(--bg-gradient); min-height: 100vh; display: flex; justify-content: center; align-items: center; padding: 40px 20px; color: var(--text-dark); }
        .container { background: var(--card-bg); width: 100%; max-width: 850px; border-radius: 16px; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3); overflow: hidden; }
        .header { background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%); color: #ffffff; padding: 30px; text-align: center; }
        .header h1 { font-size: 24px; font-weight: 700; margin-bottom: 8px; }
        .header p { font-size: 14px; opacity: 0.9; }
        form { padding: 40px; }
        .form-section-title { font-size: 16px; font-weight: 600; color: var(--primary-color); margin-bottom: 16px; border-bottom: 2px solid var(--border-color); padding-bottom: 6px; grid-column: span 2; }
        .grid-container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }
        @media (max-width: 650px) { .grid-container { grid-template-columns: 1fr; } .form-section-title { grid-column: span 1; } }
        .form-group { display: flex; flex-direction: column; }
        .form-group label { font-size: 13px; font-weight: 600; margin-bottom: 6px; color: #475569; }
        .form-group input, .form-group select { padding: 10px 14px; font-size: 14px; border: 1px solid var(--border-color); border-radius: 8px; outline: none; background-color: #f8fafc; }
        .form-group input:focus, .form-group select:focus { border-color: var(--primary-color); background-color: #ffffff; box-shadow: 0 0 0 3px rgba(37, 99, 211, 0.15); }
        .submit-btn { background: var(--primary-color); color: white; border: none; padding: 14px; font-size: 16px; font-weight: 600; border-radius: 8px; cursor: pointer; width: 100%; box-shadow: 0 4px 6px -1px rgba(37, 99, 211, 0.2); }
        .submit-btn:hover { background: var(--primary-hover); }
        .result-container { margin-top: 24px; padding: 16px; border-radius: 8px; text-align: center; font-weight: 600; font-size: 14px; white-space: pre-wrap; word-break: break-word; }
        .result-container.success { background-color: #d1fae5; color: #065f46; border: 1px solid #a7f3d0; }
        .result-container.danger { background-color: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
        .result-container.warning { background-color: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>Credit Line Assessment & Risk Predictor</h1>
        <p>Supply profile parameters to automatically screen risk arrays via Gradient Boost evaluations.</p>
    </div>

    {% if init_error %}
    <div class="result-container warning" style="margin: 20px 40px 0 40px; text-align: left;">
        <strong>Server Configuration Warning:</strong><br>{{ init_error }}
    </div>
    {% endif %}

    <form action="/" method="POST">
        <div class="grid-container">
            <h3 class="form-section-title">Demographics Summary</h3>
            
            <div class="form-group">
                <label for="person_age">Age</label>
                <input type="number" id="person_age" name="person_age" min="18" max="100" placeholder="Ex: 31" required>
            </div>

            <div class="form-group">
                <label for="gender">Gender</label>
                <select id="gender" name="gender" required>
                    <option value="Female">Female</option>
                    <option value="Male">Male</option>
                </select>
            </div>

            <div class="form-group">
                <label for="education">Highest Education Tracked</label>
                <select id="education" name="education" required>
                    <option value="High School">High School</option>
                    <option value="Bachelor">Bachelor Degree</option>
                    <option value="Master">Master Degree</option>
                    <option value="Doctorate">Doctorate</option>
                </select>
            </div>

            <div class="form-group">
                <label for="home_ownership">Housing Tenure</label>
                <select id="home_ownership" name="home_ownership" required>
                    <option value="RENT">Renting</option>
                    <option value="OWN">Home Owner</option>
                    <option value="MORTGAGE">Mortgaged Properties</option>
                    <option value="OTHER">Other Status</option>
                </select>
            </div>

            <h3 class="form-section-title">Financial Profile</h3>

            <div class="form-group">
                <label for="person_income">Total Annual Income ($)</label>
                <input type="number" id="person_income" name="person_income" min="1" placeholder="Ex: 65000" required>
            </div>

            <div class="form-group">
                <label for="person_emp_exp">Employment Seniority (Years)</label>
                <input type="number" id="person_emp_exp" name="person_emp_exp" min="0" max="50" placeholder="Ex: 6" required>
            </div>

            <div class="form-group">
                <label for="credit_score">Risk Score Matrix (300-850)</label>
                <input type="number" id="credit_score" name="credit_score" min="300" max="850" placeholder="Ex: 720" required>
            </div>

            <div class="form-group">
                <label for="cb_person_cred_hist_length">Credit Account Seniority (Years)</label>
                <input type="number" id="cb_person_cred_hist_length" name="cb_person_cred_hist_length" min="0" max="50" placeholder="Ex: 8" required>
            </div>

            <div class="form-group">
                <label for="previous_default">Historic Defaults Flagged?</label>
                <select id="previous_default" name="previous_default" required>
                    <option value="No">No Records Found</option>
                    <option value="Yes">Yes, Historic Defaults Exist</option>
                </select>
            </div>

            <h3 class="form-section-title">Requested Loan Breakdown</h3>

            <div class="form-group">
                <label for="loan_amnt">Requested Principal Allocation ($)</label>
                <input type="number" id="loan_amnt" name="loan_amnt" min="0" placeholder="Ex: 20000" required>
            </div>

            <div class="form-group">
                <label for="loan_int_rate">Assessed Coupon / Interest Rate (%)</label>
                <input type="number" id="loan_int_rate" name="loan_int_rate" step="0.01" min="0" max="100" placeholder="Ex: 8.99" required>
            </div>

            <div class="form-group">
                <label for="loan_intent">Financing Target Domain</label>
                <select id="loan_intent" name="loan_intent" required>
                    <option value="PERSONAL">Personal Restructuring</option>
                    <option value="EDUCATION">Educational Tuition</option>
                    <option value="MEDICAL">Medical Accounts</option>
                    <option value="HOMEIMPROVEMENT">Residential Renovation</option>
                    <option value="VENTURE">Commercial Venture</option>
                </select>
            </div>
        </div>

        <button type="submit" class="submit-btn">Run Engine Check</button>

        {% if prediction_text %}
        <div class="result-container {{ prediction_class }}">
            {{ prediction_text }}
        </div>
        {% endif %}
    </form>
</div>

</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    prediction_class = None
    
    if request.method == 'POST':
        try:
            age = float(request.form.get('person_age', 0))
            income = float(request.form.get('person_income', 0))
            emp_exp = float(request.form.get('person_emp_exp', 0))
            loan_amnt = float(request.form.get('loan_amnt', 0))
            int_rate = float(request.form.get('loan_int_rate', 0))
            cred_hist = float(request.form.get('cb_person_cred_hist_length', 0))
            credit_score = float(request.form.get('credit_score', 0))
            
            pct_income = (loan_amnt / income) if income > 0 else 0.0

            gender = request.form.get('gender')
            education = request.form.get('education')
            home = request.form.get('home_ownership')
            intent = request.form.get('loan_intent')
            default = request.form.get('previous_default')

            gender_male = 1 if gender == 'Male' else 0
            edu_bachelor = 1 if education == 'Bachelor' else 0
            edu_doctorate = 1 if education == 'Doctorate' else 0
            edu_high_school = 1 if education == 'High School' else 0
            edu_master = 1 if education == 'Master' else 0
            
            home_other = 1 if home == 'OTHER' else 0
            home_own = 1 if home == 'OWN' else 0
            home_rent = 1 if home == 'RENT' else 0
            
            intent_edu = 1 if intent == 'EDUCATION' else 0
            intent_home = 1 if intent == 'HOMEIMPROVEMENT' else 0
            intent_med = 1 if intent == 'MEDICAL' else 0
            intent_person = 1 if intent == 'PERSONAL' else 0
            intent_venture = 1 if intent == 'VENTURE' else 0
            
            default_yes = 1 if default == 'Yes' else 0

            features = np.array([[
                age, income, emp_exp, loan_amnt, int_rate, pct_income, cred_hist, credit_score,
                gender_male, edu_bachelor, edu_doctorate, edu_high_school, edu_master,
                home_other, home_own, home_rent,
                intent_edu, intent_home, intent_med, intent_person, intent_venture,
                default_yes
            ]])

            if model is not None:
                prediction = model.predict(features)[0]
                if prediction == 1:
                    prediction_text = "Risk Analysis Notice: High Probability of Default Risk Flagged."
                    prediction_class = "danger"
                else:
                    prediction_text = "Approval Authorized! Applicant meets clean underwriting requirements."
                    prediction_class = "success"
            else:
                prediction_text = "Cannot evaluate entry form. The model engine failed to read correctly due to server dependency versions."
                prediction_class = "warning"

        except Exception as e:
            prediction_text = f"Input Error: Unable to process fields safely. Details: {str(e)}"
            prediction_class = "warning"

    return render_template_string(HTML_TEMPLATE, init_error=load_error_message, prediction_text=prediction_text, prediction_class=prediction_class)

if __name__ == '__main__':
    app.run(debug=True)

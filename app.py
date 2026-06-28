import os
import sys
import pickle
import numpy as np
from flask import Flask, request, render_template_string

# --- FIXED ADAPTIVE UNPICKLE PATCH ---
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ["sklearn.ensemble._gb_losses", "sklearn.utils._loss", "_loss", "sklearn.ensemble._loss"]:
            module = "sklearn._loss"
            
        if name == "CyHalfBinomialLoss":
            try:
                import sklearn._loss._loss
                return getattr(sklearn._loss._loss, name)
            except (ImportError, AttributeError):
                pass

        if module == "sklearn.ensemble._gb":
            module = "sklearn.ensemble"
            
        if module in ["numpy._core.multiarray", "numpy.core.multiarray"]:
            try:
                import numpy._core.multiarray as ma
                return getattr(ma, name)
            except ImportError:
                import numpy.core.multiarray as ma
                return getattr(ma, name)
                
        return super().find_class(module, name)

app = Flask(__name__)

MODEL_FILENAME = 'gradient_pkl (1).pkl'
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)

model = None
load_error_message = None

try:
    if not os.path.exists(MODEL_PATH):
        load_error_message = f"Model missing. Please drop the file into: '{MODEL_PATH}'."
    else:
        with open(MODEL_PATH, 'rb') as f:
            model = SafeUnpickler(f).load()
        print("Model integrated cleanly into virtual system environment memory.")
except Exception as e:
    import traceback
    load_error_message = f"System Initialization Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bank Credit Underwriting System</title>
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            --card-bg: #ffffff;
            --primary-color: #1e40af;
            --primary-hover: #1d4ed8;
            --text-dark: #1e293b;
            --border-color: #e2e8f0;
            --success-bg: #d1fae5;
            --success-text: #065f46;
            --success-border: #a7f3d0;
            --danger-bg: #fee2e2;
            --danger-text: #991b1b;
            --danger-border: #fecaca;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, sans-serif; }
        body { background: var(--bg-gradient); min-height: 100vh; display: flex; justify-content: center; align-items: center; padding: 40px 20px; color: var(--text-dark); }
        .container { background: var(--card-bg); width: 100%; max-width: 850px; border-radius: 16px; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3); overflow: hidden; }
        .header { background: linear-gradient(135deg, #111827 0%, #1e40af 100%); color: #ffffff; padding: 35px; text-align: center; border-bottom: 4px solid #3b82f6; }
        .header h1 { font-size: 26px; font-weight: 700; margin-bottom: 8px; letter-spacing: 0.5px; }
        .header p { font-size: 14px; opacity: 0.85; }
        
        /* New Professional Result UI */
        .status-box { margin: 30px 40px 10px 40px; padding: 25px; border-radius: 12px; border: 1px solid; text-align: center; }
        .status-box.APPROVED { background-color: var(--success-bg); color: var(--success-text); border-color: var(--success-border); }
        .status-box.NOT_APPROVED { background-color: var(--danger-bg); color: var(--danger-text); border-color: var(--danger-border); }
        .status-title { font-size: 28px; font-weight: 800; letter-spacing: 2px; margin-bottom: 5px; text-transform: uppercase; }
        .status-desc { font-size: 15px; font-weight: 500; opacity: 0.9; }
        
        form { padding: 40px; }
        .form-section-title { font-size: 16px; font-weight: 600; color: #2563eb; margin-bottom: 16px; border-bottom: 2px solid var(--border-color); padding-bottom: 6px; grid-column: span 2; }
        .grid-container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }
        @media (max-width: 650px) { .grid-container { grid-template-columns: 1fr; } .form-section-title { grid-column: span 1; } }
        .form-group { display: flex; flex-direction: column; }
        .form-group label { font-size: 13px; font-weight: 600; margin-bottom: 6px; color: #475569; }
        .form-group input, .form-group select { padding: 12px 14px; font-size: 14px; border: 1px solid var(--border-color); border-radius: 8px; outline: none; background-color: #f8fafc; transition: all 0.2s; }
        .form-group input:focus, .form-group select:focus { border-color: #2563eb; background-color: #ffffff; box-shadow: 0 0 0 3px rgba(37, 99, 211, 0.15); }
        .submit-btn { background: #2563eb; color: white; border: none; padding: 16px; font-size: 16px; font-weight: 700; border-radius: 8px; cursor: pointer; width: 100%; box-shadow: 0 4px 6px -1px rgba(37, 99, 211, 0.2); transition: background 0.2s; text-transform: uppercase; letter-spacing: 0.5px; }
        .submit-btn:hover { background: var(--primary-hover); }
        
        .error-banner { background-color: #fff9db; color: #92400e; border: 1px solid #ffe3e3; padding: 15px; margin: 20px 40px 0 40px; border-radius: 8px; font-size: 14px; }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>Retail Credit Decisioning Portal</h1>
        <p>Automated Automated Scoring Engine & Loan Risk Assessment Matrix</p>
    </div>

    {% if init_error %}
    <div class="error-banner">
        <strong>System Initialization Warning:</strong> {{ init_error }}
    </div>
    {% endif %}

    {% if decision_status %}
        <div class="status-box {{ decision_status }}">
            <div class="status-title">
                {% if decision_status == 'APPROVED' %}
                    ✓ LOAN APPROVED
                {% else %}
                    ✕ LOAN NOT APPROVED
                {% endif %}
            </div>
            <div class="status-desc">{{ decision_message }}</div>
        </div>
    {% endif %}

    <form action="/" method="POST">
        <div class="grid-container">
            <h3 class="form-section-title">Demographics Summary</h3>
            <div class="form-group"><label for="person_age">Age</label><input type="number" id="person_age" name="person_age" min="18" max="100" placeholder="Ex: 31" required></div>
            <div class="form-group"><label for="gender">Gender</label><select id="gender" name="gender" required><option value="Female">Female</option><option value="Male">Male</option></select></div>
            <div class="form-group"><label for="education">Highest Education Tracked</label><select id="education" name="education" required><option value="High School">High School</option><option value="Bachelor">Bachelor Degree</option><option value="Master">Master Degree</option><option value="Doctorate">Doctorate</option></select></div>
            <div class="form-group"><label for="home_ownership">Housing Tenure</label><select id="home_ownership" name="home_ownership" required><option value="RENT">Renting</option><option value="OWN">Home Owner</option><option value="MORTGAGE">Mortgaged Properties</option><option value="OTHER">Other Status</option></select></div>
            
            <h3 class="form-section-title">Financial Profile</h3>
            <div class="form-group"><label for="person_income">Total Annual Income ($)</label><input type="number" id="person_income" name="person_income" min="1" placeholder="Ex: 65000" required></div>
            <div class="form-group"><label for="person_emp_exp">Employment Seniority (Years)</label><input type="number" id="person_emp_exp" name="person_emp_exp" min="0" max="50" placeholder="Ex: 6" required></div>
            <div class="form-group"><label for="credit_score">Risk Score Matrix (300-850)</label><input type="number" id="credit_score" name="credit_score" min="300" max="850" placeholder="Ex: 720" required></div>
            <div class="form-group"><label for="cb_person_cred_hist_length">Credit Account Seniority (Years)</label><input type="number" id="cb_person_cred_hist_length" name="cb_person_cred_hist_length" min="0" max="50" placeholder="Ex: 8" required></div>
            <div class="form-group"><label for="previous_default">Historic Defaults Flagged?</label><select id="previous_default" name="previous_default" required><option value="No">No Records Found</option><option value="Yes">Yes, Historic Defaults Exist</option></select></div>
            
            <h3 class="form-section-title">Requested Loan Breakdown</h3>
            <div class="form-group"><label for="loan_amnt">Requested Principal Allocation ($)</label><input type="number" id="loan_amnt" name="loan_amnt" min="0" placeholder="Ex: 20000" required></div>
            <div class="form-group"><label for="loan_int_rate">Assessed Coupon / Interest Rate (%)</label><input type="number" id="loan_int_rate" name="loan_int_rate" step="0.01" min="0" max="100" placeholder="Ex: 8.99" required></div>
            <div class="form-group"><label for="loan_intent">Financing Target Domain</label><select id="loan_intent" name="loan_intent" required><option value="PERSONAL">Personal Restructuring</option><option value="EDUCATION">Educational Tuition</option><option value="MEDICAL">Medical Accounts</option><option value="HOMEIMPROVEMENT">Residential Renovation</option><option value="VENTURE">Commercial Venture</option></select></div>
        </div>

        <button type="submit" class="submit-btn">Evaluate Application</button>
    </form>
</div>

</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    decision_status = None
    decision_message = None
    
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
                # prediction == 1 means Default Risk (Reject)
                # prediction == 0 means Safe (Approve)
                if prediction == 1:
                    decision_status = "NOT_APPROVED"
                    decision_message = "Applicant fails to meet standard credit risk criteria. Risk profile indicates a high probability of default."
                else:
                    decision_status = "APPROVED"
                    decision_message = "Applicant meets underwriting requirements. Credit risk profile cleared for deployment."
            else:
                decision_status = "NOT_APPROVED"
                decision_message = "System error: Backend AI scoring core is offline."
        except Exception as e:
            decision_status = "NOT_APPROVED"
            decision_message = f"Processing Error: Could not evaluate application attributes. ({str(e)})"

    return render_template_string(HTML_TEMPLATE, init_error=load_error_message, decision_status=decision_status, decision_message=decision_message)

if __name__ == '__main__':
    app.run(debug=True)

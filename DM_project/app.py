from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the models
dtree = pickle.load(open('decision_tree_model.pkl', 'rb'))
rfc = pickle.load(open('random_forest_model.pkl', 'rb'))

# List of features used during training (Must match model input order)
FEATURES = [
    'BirthAsphyxia', 'HypDistrib', 'HypoxiaInO2', 'CO2', 'ChestXray', 
    'Grunting', 'LVHreport', 'LowerBodyO2', 'RUQO2', 'CO2Report', 
    'XrayReport', 'Disease', 'GruntingReport', 'Age', 
    'LVH', 'DuctFlow', 'CardiacMixing', 'LungParench', 'LungFlow'
]

FEATURE_META = {
    "BirthAsphyxia": {
        "label": "Birth asphyxia",
        "note": "History of perinatal asphyxia.",
        "type": "select",
        "options": {0: "No", 1: "Yes"},
    },
    "HypDistrib": {
        "label": "Hypoxemia distribution",
        "note": "Whether hypoxemia appears evenly distributed.",
        "type": "select",
        "options": {0: "Equal", 1: "Unequal"},
    },
    "HypoxiaInO2": {
        "label": "Hypoxemia severity on oxygen",
        "note": "Severity category while receiving supplemental O₂.",
        "type": "select",
        "options": {0: "Mild", 1: "Moderate", 2: "Severe"},
    },
    "CO2": {
        "label": "CO₂ level category",
        "note": "Categorized CO₂ measurement (e.g., PaCO₂).",
        "type": "select",
        "options": {0: "Low", 1: "Normal", 2: "High"},
    },
    "ChestXray": {
        "label": "Chest X‑ray pattern (coded)",
        "note": "Radiographic appearance category.",
        "type": "select",
        "options": {
            0: "Asymmetric/patchy opacities",
            1: "Ground-glass appearance",
            2: "Normal",
            3: "Oligaemic (reduced pulmonary vascularity)",
            4: "Plethoric (increased pulmonary vascularity)",
        },
    },
    "Grunting": {
        "label": "Expiratory grunting",
        "note": "Clinical sign of respiratory distress.",
        "type": "select",
        "options": {0: "No", 1: "Yes"},
    },
    "LVHreport": {
        "label": "LVH reported",
        "note": "Left ventricular hypertrophy noted in report.",
        "type": "select",
        "options": {0: "No", 1: "Yes"},
    },
    "LowerBodyO2": {
        "label": "Pre-/post-ductal O₂ difference (lower body)",
        "note": "Encoded difference category (as used during model training).",
        "type": "select",
        "options": {0: "≥ 12", 1: "5–12", 2: "< 5"},
    },
    "RUQO2": {
        "label": "Pre-/post-ductal O₂ difference (right upper quadrant)",
        "note": "Encoded difference category (as used during model training).",
        "type": "select",
        "options": {0: "≥ 12", 1: "5–12", 2: "< 5"},
    },
    "CO2Report": {
        "label": "CO₂ report threshold",
        "note": "Thresholded CO₂ report category.",
        "type": "select",
        "options": {0: "< 7.5", 1: "≥ 7.5"},
    },
    "XrayReport": {
        "label": "X‑ray report category (coded)",
        "note": "Coded category from radiology report.",
        "type": "select",
        "options": {
            0: "Asymmetric/patchy opacities",
            1: "Ground-glass appearance",
            2: "Normal",
            3: "Oligaemic (reduced pulmonary vascularity)",
            4: "Plethoric (increased pulmonary vascularity)",
        },
    },
    "Disease": {
        "label": "Primary diagnosis (coded)",
        "note": "Select the category matching the clinical diagnosis.",
        "type": "select",
        "options": {
            0: "Tetralogy of Fallot",
            1: "Primary lung disease",
            2: "Pulmonary atresia with intact ventricular septum (PAIVS)",
            3: "Persistent fetal circulation / PPHN (PFC)",
            4: "Total anomalous pulmonary venous drainage (TAPVD)",
            5: "Transposition of the great arteries (TGA)",
        },
    },
    "GruntingReport": {
        "label": "Grunting reported",
        "note": "Grunting noted in clinical documentation.",
        "type": "select",
        "options": {0: "No", 1: "Yes"},
    },
    "Age": {
        "label": "Postnatal age (coded)",
        "note": "Age band as used during model training.",
        "type": "select",
        "options": {0: "0–3 days", 2: "4–10 days", 1: "11–30 days"},
    },
    "LVH": {
        "label": "Left ventricular hypertrophy (LVH)",
        "note": "Indicator for LVH.",
        "type": "select",
        "options": {0: "No", 1: "Yes"},
    },
    "DuctFlow": {
        "label": "Ductal flow direction (PDA)",
        "note": "Directionality category of ductal flow.",
        "type": "select",
        "options": {0: "Left-to-right", 1: "None", 2: "Right-to-left"},
    },
    "CardiacMixing": {
        "label": "Cardiac mixing (coded)",
        "note": "Degree of intracardiac mixing (coded).",
        "type": "select",
        "options": {0: "Complete", 1: "Mild", 2: "None", 3: "Transposition (Transp.)"},
    },
    "LungParench": {
        "label": "Lung parenchyma status (coded)",
        "note": "Coded parenchymal status.",
        "type": "select",
        "options": {0: "Abnormal", 1: "Congested", 2: "Normal"},
    },
    "LungFlow": {
        "label": "Pulmonary blood flow (coded)",
        "note": "Coded pulmonary flow assessment.",
        "type": "select",
        "options": {0: "Low", 1: "Normal", 2: "High"},
    },
}


def _validate_and_parse_inputs(form_data: dict) -> tuple[list[float], dict]:
    errors: dict[str, str] = {}
    values: list[float] = []

    for feature in FEATURES:
        raw_value = (form_data.get(feature) or "").strip()
        if raw_value == "":
            errors[feature] = "This field is required."
            continue

        meta = FEATURE_META.get(feature, {})
        try:
            numeric_value = float(raw_value)
        except ValueError:
            errors[feature] = "Enter a valid number."
            continue

        if meta.get("type") == "select":
            try:
                numeric_value_int = int(numeric_value)
            except (TypeError, ValueError):
                errors[feature] = "Select a valid option."
                continue

            options = meta.get("options") or {}
            if options and numeric_value_int not in options:
                errors[feature] = "Select a valid option."
                continue

            numeric_value = float(numeric_value_int)

        values.append(numeric_value)

    return values, errors

@app.route('/')
def home():
    return render_template(
        'index.html',
        features=FEATURES,
        feature_meta=FEATURE_META,
        form_values={},
        errors={},
    )

@app.route('/predict', methods=['POST'])
def predict():
    form_values = request.form.to_dict(flat=True)
    input_values, errors = _validate_and_parse_inputs(form_values)
    if errors:
        return render_template(
            'index.html',
            features=FEATURES,
            feature_meta=FEATURE_META,
            form_values=form_values,
            errors=errors,
        )
    
    # Convert to DataFrame (models expect 2D array/DataFrame with feature names)
    final_features = pd.DataFrame([input_values], columns=FEATURES)

    # Predictions
    dt_pred = dtree.predict(final_features)[0]
    rf_pred = rfc.predict(final_features)[0]

    # Probabilities
    dt_prob = np.max(dtree.predict_proba(final_features)) * 100
    rf_prob = np.max(rfc.predict_proba(final_features)) * 100

    results = {
        'dt_is_high_risk': bool(dt_pred == 1),
        'dt_label': "Higher-risk pattern" if dt_pred == 1 else "Lower-risk pattern",
        'dt_conf': round(dt_prob, 2),
        'rf_is_high_risk': bool(rf_pred == 1),
        'rf_label': "Higher-risk pattern" if rf_pred == 1 else "Lower-risk pattern",
        'rf_conf': round(rf_prob, 2),
    }

    return render_template(
        'index.html',
        features=FEATURES,
        feature_meta=FEATURE_META,
        form_values=form_values,
        results=results,
        errors={},
    )

if __name__ == "__main__":
    app.run(debug=True)

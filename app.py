"""Flask backend for the COPD Risk Predictor.

Loads the saved scikit-learn pipeline, the ordered feature list, the tuned
probability cutoff, and a metadata JSON with dropdown options/defaults at
import time, then serves a single page from templates/index.html.

The prediction code path is one line: model.predict_proba(...) compared to the
saved threshold. Same logic the inference cell in the notebook uses.

    Menna Allah AlSayed (211001864)
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


# ---------------------------------------------------------------------------
# Paths and artefacts
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent

ARTIFACT_PATHS = {
    "model":     HERE / "copd_risk_model.pkl",
    "features":  HERE / "copd_risk_features.pkl",
    "threshold": HERE / "copd_risk_threshold.pkl",
    "metadata":  HERE / "copd_risk_metadata.json",
}

MODEL     = joblib.load(ARTIFACT_PATHS["model"])
FEATURES  = joblib.load(ARTIFACT_PATHS["features"])
THRESHOLD = float(joblib.load(ARTIFACT_PATHS["threshold"]))
META      = json.loads(ARTIFACT_PATHS["metadata"].read_text(encoding="utf-8"))

OPTIONS  = META["options"]
DEFAULTS = META["defaults"]

# Sensible "if the user didn't pick anything" categorical defaults. These keep
# hidden form fields realistic instead of falling back to "Unknown" everywhere.
SENSIBLE_CATEGORICAL = {
    "State": "California",
    "Sex": "Female",
    "AgeGroup": "45-49",
    "GeneralHealth": "Good",
    "HasPersonalDoctor": "One",
    "CouldNotAffordDoctor": "No",
    "RecentCheckup": "Within past year",
    "ExercisePastMonth": "Yes",
    "HeartAttack": "No",
    "CoronaryHeartDisease": "No",
    "Stroke": "No",
    "AsthmaEver": "No",
    "AsthmaCurrent": "No",
    "Depression": "No",
    "KidneyDisease": "No",
    "Arthritis": "No",
    "Diabetes": "No",
    "MaritalStatus": "Married",
    "Education": "High school graduate",
    "Employment": "Employed for wages",
    "Income": "$50,000-$75,000",
    "Deaf": "No",
    "Blind": "No",
    "DifficultyDeciding": "No",
    "DifficultyWalking": "No",
    "DifficultyDressing": "No",
    "DifficultyAlone": "No",
    "Smoked100": "No",
    "SmokingFrequency": "Not at all",
    "SmokelessTobacco": "Not at all",
    "AlcoholPast30Days": "No",
    "BMICategory": "Overweight",
    "CurrentSmokerFlag": "No",
    "CurrentECigFlag": "No",
    "NoPhysicalActivityFlag": "No",
    "FairPoorHealthFlag": "No",
    "PhysicalDistressFlag": "No",
    "MentalDistressFlag": "No",
    "HasInsuranceFlag": "Yes",
    "Age18To64InsuranceFlag": "Yes",
    "HeartDiseaseFlag": "No",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_number(form, name):
    """Pull a numeric field out of the form, falling back to the saved default."""
    fallback = DEFAULTS.get(name, 0)
    raw = form.get(name)
    if raw is None or raw == "":
        return float(fallback)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(fallback)


def _pick_category(form, name):
    """Pull a categorical field, ensuring the chosen value is in the allowed set."""
    allowed = OPTIONS[name]
    submitted = form.get(name)
    if submitted in allowed:
        return submitted
    preferred = SENSIBLE_CATEGORICAL.get(name)
    if preferred in allowed:
        return preferred
    return allowed[0]


def assemble_row(form):
    """Build a one-row dict in the column order the model expects."""
    row = {col: _parse_number(form, col) for col in META["numeric_features"]}
    row.update({col: _pick_category(form, col) for col in META["categorical_features"]})
    return row


def render_page(**extra):
    """Render index.html with the standard set of context variables plus extras."""
    context = {
        "options": OPTIONS,
        "defaults": DEFAULTS,
        "metrics": META["metrics"],
        "rows": META["rows"],
        "threshold": round(THRESHOLD * 100, 1),
        "categorical_defaults": SENSIBLE_CATEGORICAL,
    }
    context.update(extra)
    return render_template("index.html", **context)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/")
def home():
    return render_page()


@app.route("/predict", methods=["POST"])
def predict():
    try:
        row = assemble_row(request.form)
        frame = pd.DataFrame([row])[FEATURES]
        probability = float(MODEL.predict_proba(frame)[0, 1])
        is_high_risk = probability >= THRESHOLD

        return render_page(
            prediction="Higher COPD Risk" if is_high_risk else "Lower COPD Risk",
            probability=round(probability * 100, 1),
            risk="high" if is_high_risk else "low",
            form_data=request.form,
        )
    except Exception as exc:
        return render_page(error=str(exc), form_data=request.form)


if __name__ == "__main__":
    app.run(debug=True, port=5000)

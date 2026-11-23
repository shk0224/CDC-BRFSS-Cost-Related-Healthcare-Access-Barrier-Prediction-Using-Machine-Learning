"""
Cost-Barrier Screener - web app (FastAPI)

Loads the 12-question screener saved by notebook 2 (models/brfss2024_short_screener.joblib)
and shows a form. The person answers 12 questions; the app returns a calibrated risk %
and a Yes/No flag using the threshold saved with the model.

Run from the project root:
    python app.py              (then open http://127.0.0.1:8000)
or  uvicorn app:app --reload
"""
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "brfss2024_short_screener.joblib"
APP_HOST, APP_PORT = "127.0.0.1", 8000

# ---------------------------------------------------------------------------
# The questions. "value" must match the labels used in training exactly
# (BRFSS2024_access.csv). "Unknown" = the person prefers not to answer.
# ---------------------------------------------------------------------------
UNKNOWN = ("Unknown", "Prefer not to say / don't know")
Option = Tuple[str, str]  # (model value, text shown to the user)

QUESTIONS: List[Dict] = [
    {"key": "income", "label": "What is your yearly household income?",
     "options": [("<$15k", "Less than $15,000"), ("$15k-<25k", "$15,000 to $24,999"),
                 ("$25k-<35k", "$25,000 to $34,999"), ("$35k-<50k", "$35,000 to $49,999"),
                 ("$50k-<100k", "$50,000 to $99,999"), ("$100k-<200k", "$100,000 to $199,999"),
                 ("$200k+", "$200,000 or more"), UNKNOWN]},
    {"key": "insurance_type", "label": "What is your main source of health coverage?",
     "options": [("Employer/union", "Employer or union plan"), ("Self-purchased private", "Plan I buy myself (e.g. Marketplace)"),
                 ("Medicare", "Medicare"), ("Medicaid/CHIP/State", "Medicaid, CHIP or a state plan"),
                 ("Military/VA/IHS/Other gov", "Military, VA, Indian Health Service or other government"),
                 ("No coverage", "No health coverage"), UNKNOWN]},
    {"key": "age_group", "label": "What is your age?",
     "options": [(a, a.replace("-", " to ").replace("80+", "80 or older"))
                 for a in ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
                           "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]] + [UNKNOWN]},
    {"key": "employment", "label": "What is your current work status?",
     "options": [("Employed for wages", "Employed for wages"), ("Self-employed", "Self-employed"),
                 ("Out of work <1 yr", "Out of work for less than 1 year"),
                 ("Out of work 1+ yr", "Out of work for 1 year or more"), ("Homemaker", "Homemaker"),
                 ("Student", "Student"), ("Retired", "Retired"), ("Unable to work", "Unable to work"), UNKNOWN]},
    {"key": "mental_unhealthy_days",
     "label": "In the past 30 days, on how many days was your mental health (stress, depression, emotions) not good?",
     "options": [("0 days", "None"), ("1-13 days", "1 to 13 days"), ("14+ days", "14 days or more"), UNKNOWN]},
    {"key": "general_health", "label": "In general, how would you rate your health?",
     "options": [("Excellent", "Excellent"), ("Very good", "Very good"), ("Good", "Good"),
                 ("Fair", "Fair"), ("Poor", "Poor"), UNKNOWN]},
    {"key": "home_ownership", "label": "Do you own or rent your home?",
     "options": [("Own", "Own"), ("Rent", "Rent"), ("Other arrangement", "Other arrangement"), UNKNOWN]},
    {"key": "physical_unhealthy_days",
     "label": "In the past 30 days, on how many days was your physical health (illness, injury) not good?",
     "options": [("0 days", "None"), ("1-13 days", "1 to 13 days"), ("14+ days", "14 days or more"), UNKNOWN]},
    # smoking_status is built from two questions, exactly as CDC does (_SMOKER3)
    {"key": "smoked_100", "label": "Have you smoked at least 100 cigarettes in your entire life?",
     "options": [("Yes", "Yes"), ("No", "No"), UNKNOWN]},
    {"key": "smoke_now", "label": "Do you now smoke cigarettes every day, some days, or not at all?",
     "hint": "Answer only if you said Yes above.",
     "options": [("Every day", "Every day"), ("Some days", "Some days"), ("Not at all", "Not at all"), UNKNOWN],
     "optional": True},
    {"key": "cognitive_difficulty",
     "label": "Because of a physical, mental or emotional condition, do you have serious difficulty concentrating, remembering or making decisions?",
     "options": [("Yes", "Yes"), ("No", "No"), UNKNOWN]},
    {"key": "marital_status", "label": "What is your marital status?",
     "options": [("Married", "Married"), ("Unmarried couple", "Member of an unmarried couple"),
                 ("Divorced", "Divorced"), ("Widowed", "Widowed"), ("Separated", "Separated"),
                 ("Never married", "Never married"), UNKNOWN]},
]


def smoking_status(smoked_100: str, smoke_now: str) -> str:
    """Combine the two smoking questions into the model's smoking_status, like CDC's _SMOKER3."""
    if smoked_100 == "No":
        return "Never"
    if smoked_100 == "Yes":
        return {"Every day": "Current daily", "Some days": "Current some days",
                "Not at all": "Former"}.get(smoke_now, "Unknown")
    return "Unknown"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
screener = joblib.load(MODEL_PATH)
MODEL, THRESHOLD, FEATURES = screener["model"], screener["threshold"], screener["features"]


def to_model_row(answers: Dict[str, str]) -> pd.DataFrame:
    row = {k: answers.get(k, "Unknown") for k in FEATURES if k != "smoking_status"}
    row["smoking_status"] = smoking_status(answers.get("smoked_100", "Unknown"), answers.get("smoke_now", "Unknown"))
    return pd.DataFrame([row])[FEATURES]


def predict(answers: Dict[str, str]) -> Dict:
    risk = float(MODEL.predict_proba(to_model_row(answers))[0, 1])
    flagged = risk >= THRESHOLD
    return {"risk_pct": round(100 * risk, 1), "flagged": flagged, "threshold_pct": round(100 * THRESHOLD)}


# ---------------------------------------------------------------------------
# Web app
# ---------------------------------------------------------------------------
app = FastAPI(title="Cost-Barrier Screener")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

def render(request: Request, context: Dict):
    """Works with both new and older FastAPI/Starlette versions."""
    try:
        return templates.TemplateResponse(request=request, name="index.html", context=context)
    except TypeError:  # older Starlette: TemplateResponse(name, {"request": request, ...})
        return templates.TemplateResponse("index.html", {"request": request, **context})


VALID = {q["key"]: {v for v, _ in q["options"]} for q in QUESTIONS}


@app.get("/")
async def index(request: Request):
    return render(request, {"questions": QUESTIONS, "answers": {}, "result": None})


@app.post("/")
async def screen(request: Request):
    form = await request.form()
    answers, errors = {}, []
    for q in QUESTIONS:
        value = form.get(q["key"]) or ("Unknown" if q.get("optional") else None)
        if value not in VALID[q["key"]]:
            errors.append(q["label"])
            continue
        answers[q["key"]] = value
    result = None if errors else predict(answers)
    return render(request, {"questions": QUESTIONS, "answers": answers, "result": result, "errors": errors})


@app.post("/api/predict")
async def api_predict(answers: Dict[str, str]):
    """JSON version, e.g. {"income": "<$15k", "insurance_type": "No coverage", ...}. Missing keys count as Unknown."""
    bad = [k for k, v in answers.items() if k in VALID and v not in VALID[k]]
    if bad:
        return {"error": f"invalid values for: {bad}"}
    return predict(answers)


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
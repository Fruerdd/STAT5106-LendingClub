"""Training-month weights and application-time features for the default-F1 comparison."""
import numpy as np
import pandas as pd

from src.joint_accuracy_f1 import operating_curve

REFERENCE = "ordinal_subgrade_credit_ratios"
BLEND = "reference_plus_combined"
# (Use additional application features, give recent training loans more weight.)
SETTINGS = {REFERENCE: (False, False), "recent": (False, True),
            "loan_terms": (True, False), "combined": (True, True)}
CANDIDATES = ("recent", "loan_terms", "combined", BLEND)


def recency_weights(issue_dates):
    """One-year half-life in calendar months. Preserve the total training weight."""
    dates = pd.DatetimeIndex(pd.to_datetime(issue_dates))
    if dates.empty or dates.hasnans:
        raise ValueError("Training dates must be present")
    month = np.asarray(dates.year * 12 + dates.month)
    weight = np.exp2(-(month.max() - month) / 12.)
    return weight / weight.mean()


def loan_terms(frame):
    income = frame.annual_inc.where(frame.annual_inc.gt(0))
    monthly_debt = income * frame.dti.where(frame.dti.ge(0)) / 1200.
    amount = frame.loan_amnt.where(frame.loan_amnt.gt(0))
    balance = frame.revol_bal.where(frame.revol_bal.ge(0))
    return pd.DataFrame({
        "implied_monthly_debt": monthly_debt,
        "installment_to_existing_debt": frame.installment / monthly_debt.where(monthly_debt.gt(0)),
        "revolving_balance_per_open_account": balance / frame.open_acc.where(frame.open_acc.gt(0)),
        "loan_amount_multiple_of_1000": amount.mod(1000).eq(0).astype(float).where(amount.notna())
    }, index=frame.index)


def matrices(encoder, frame):
    base = encoder.transform(frame)
    extra = loan_terms(frame).to_numpy(dtype="float32")
    return base, np.column_stack((base, extra))


def f1_rule(y, p, accuracy_floor=0.):
    curve = operating_curve(y, p)
    eligible = curve.loc[curve.accuracy >= accuracy_floor]
    if eligible.empty:
        raise ValueError("No threshold satisfies the accuracy floor")
    return eligible.sort_values(["f1", "accuracy", "threshold"], ascending=False).iloc[0]


def member_names(choice):
    return (REFERENCE, "combined") if choice == BLEND else (choice,)


def predict(bundle, frame):
    inputs = matrices(bundle["encoder"], frame)
    scores = []
    for name in member_names(bundle["candidate"]):
        extra, _ = SETTINGS[name]
        scores.append(bundle["models"][name].predict_proba(inputs[int(extra)])[:, 1].astype(float))
    return np.mean(scores, axis=0)

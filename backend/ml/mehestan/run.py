import pandas as pd
from ml.inputs import MlInput
from .individual import compute_individual_score


def get_individual_scores(ml_input: MlInput):
    comparisons_df = ml_input.get_comparisons(criteria="largely_recommended")
    individual_scores = []
    for (user_id, user_comparisons) in comparisons_df.groupby("user_id"):
        scores = compute_individual_score(user_comparisons)
        scores["user_id"] = user_id
        individual_scores.append(scores.reset_index())
    result = pd.concat(individual_scores, ignore_index=True, copy=False)
    return result[["user_id", "entity_id", "score", "uncertainty"]]


def run_mehestan(ml_input: MlInput):
    pass
    

    




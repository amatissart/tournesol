import pandas as pd

from ml.inputs import MlInput
from tournesol.models import Poll
from .individual import compute_individual_score
from .global_scores import get_global_scores


def get_individual_scores(ml_input: MlInput, criteria: str) -> pd.DataFrame:
    comparisons_df = ml_input.get_comparisons(criteria=criteria)
    individual_scores = []
    for (user_id, user_comparisons) in comparisons_df.groupby("user_id"):
        scores = compute_individual_score(user_comparisons)
        if scores is None:
            continue
        scores["user_id"] = user_id
        individual_scores.append(scores.reset_index())
    result = pd.concat(individual_scores, ignore_index=True, copy=False)
    return result[["user_id", "entity_id", "score", "uncertainty"]]


def run_mehestan(ml_input: MlInput, poll: Poll):
    for criteria in poll.criterias_list:
        indiv_scores = get_individual_scores(ml_input, criteria=criteria)
        global_scores = get_global_scores(ml_input, individual_scores=indiv_scores)

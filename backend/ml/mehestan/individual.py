from django.db import transaction
from django.db.models import F
import numpy as np
import pandas as pd


from tournesol.models import (
    ComparisonCriteriaScore,
    ContributorRatingCriteriaScore,
    ContributorRating,
)

R_MAX = 10
ALPHA = 0.01


def get_comparisons_scores(username, criteria):
    public_dataset = pd.read_csv(
        "~/workspace/data/tournesol_public_export_2022-02-18.csv"
    )
    df = public_dataset[
        (public_dataset.public_username == username)
        & (public_dataset.criteria == criteria)
    ]
    df = df[["video_a", "video_b", "score"]]
    df.columns = ["entity_a", "entity_b", "score"]
    return df


def compute_individual_score(scores):
    scores = scores[["entity_a", "entity_b", "score"]]

    scores_sym = pd.concat(
        [
            scores,
            pd.DataFrame(
                {
                    "entity_a": scores.entity_b,
                    "entity_b": scores.entity_a,
                    "score": -1 * scores.score,
                }
            ),
        ]
    )

    r = scores_sym.pivot(index="entity_a", columns="entity_b", values="score")
    assert r.index.equals(r.columns)

    r_tilde = r / (1.0 + R_MAX)
    r_tilde2 = r_tilde ** 2

    l = r_tilde / np.sqrt(1.0 - r_tilde2)
    k = (1.0 - r_tilde2) ** 3

    L = k.mul(l).sum(axis=1)
    K_diag = pd.DataFrame(
        data=np.diag(k.sum(axis=1) + ALPHA),
        index=k.index,
        columns=k.index,
    )
    K = K_diag.sub(k, fill_value=0)

    # theta_star = K^-1 * L
    theta_star = pd.Series(np.linalg.solve(K, L), index=L.index)

    # Compute uncertainties
    if len(scores) <= 1:
        # FIXME: default value for uncertainty?
        delta_star = 0.0
    else:
        theta_star_numpy = theta_star.to_numpy()
        theta_star_ab = pd.DataFrame(
            np.subtract.outer(theta_star_numpy, theta_star_numpy),
            index=theta_star.index,
            columns=theta_star.index,
        )
        sigma2 = np.nansum(k * (l - theta_star_ab) ** 2) / 2 / (len(scores) - 1)
        delta_star = pd.Series(np.sqrt(sigma2) / np.sqrt(np.diag(K)), index=K.index)

    # r.loc[a:b] is negative when a is prefered to b.
    # The sign of the result is inverted.
    result = pd.DataFrame(
        {
            "score": -1 * theta_star,
            "uncertainty": delta_star,
        }
    )
    result.index.name = "entity_id"
    return result


# def save_individual_scores(user_id, scores):
#     rating_ids = {
#         entity_id: rating_id
#         for rating_id, entity_id in ContributorRating.objects.filter(
#             poll__name=POLL_NAME,
#             user_id=user_id
#         ).values_list("id", "entity_id")
#     }

#     with transaction.atomic():
#         ContributorRatingCriteriaScore.objects.filter(
#             contributor_rating__user_id=user_id,
#             contributor_rating__poll__name=POLL_NAME,
#             criteria=criteria
#         )
#         ratings = [
#             ContributorRatingCriteriaScore(pk=entity_id,  )
#             for entity_id, columns in scores.iterrows()
#         ]
#     # TODO

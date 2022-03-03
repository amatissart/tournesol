import pandas as pd
import numpy as np
from django.db.models import F, Case, When

from tournesol.models import ContributorRatingCriteriaScore
from core.models import User

from .primitives import BrMean, QrMed, QrUnc


W = 5.0

SCALING_WEIGHT_SUPERTRUSTED = W
SCALING_WEIGHT_TRUSTED = 1.0
SCALING_WEIGHT_NONTRUSTED = 0.0

VOTE_WEIGHT_TRUSTED_PUBLIC = 1.0
VOTE_WEIGHT_TRUSTED_PRIVATE = 0.5

TOTAL_VOTE_WEIGHT_NONTRUSTED_DEFAULT = 2.0  # w_⨯,default
TOTAL_VOTE_WEIGHT_NONTRUSTED_FRACTION = 0.1  # f_⨯


def get_trusted_user_ids():
    return set(User.trusted_users().values_list("pk", flat=True))


def get_supertrusted_user_ids():
    return set(User.supertrusted_users().values_list("pk", flat=True))


def get_user_scaling_weights():
    values = (
        User.objects.all()
        .annotate(
            scaling_weight=Case(
                When(
                    pk__in=User.supertrusted_users(), then=SCALING_WEIGHT_SUPERTRUSTED
                ),
                When(pk__in=User.trusted_users(), then=SCALING_WEIGHT_TRUSTED),
                default=SCALING_WEIGHT_NONTRUSTED,
            )
        )
        .values("scaling_weight", user_id=F("pk"))
    )
    return {u["user_id"]: u["scaling_weight"] for u in values}


def get_contributor_criteria_score(users, poll_name, criteria):
    values = ContributorRatingCriteriaScore.objects.filter(
        contributor_rating__poll__name=poll_name,
        contributor_rating__user__in=users,
        criteria=criteria,
    ).values(
        "score",
        "uncertainty",
        user_id=F("contributor_rating__user__pk"),
        uid=F("contributor_rating__entity__uid"),
    )
    return pd.DataFrame(values)


def compute_scaling(
    df: pd.DataFrame,
    users_to_compute=None,
    reference_users=None,
    compute_uncertainties=False,
):
    scaling_weights = get_user_scaling_weights()

    def get_significantly_different_pairs(scores: pd.DataFrame):
        # To optimize: this cross product may be expensive in memory
        return (
            scores.merge(scores, how="cross", suffixes=("_a", "_b"))
            .query("uid_a < uid_b")
            .query("abs(score_a - score_b) >= 2*(uncertainty_a + uncertainty_b)")
            .set_index(["uid_a", "uid_b"])
        )

    if users_to_compute is None:
        users_to_compute = df.user_id.unique()

    if reference_users is None:
        reference_users = df.user_id.unique()

    s_dict = {}
    for user_n in users_to_compute:
        user_scores = df[df.user_id == user_n].drop("user_id", axis=1)
        s_nqm = []
        delta_s_nqm = []
        s_weights = []
        for user_m in (u for u in reference_users if u != user_n):
            m_scores = df[df.user_id == user_m].drop("user_id", axis=1)
            common_uids = set(user_scores.uid).intersection(m_scores.uid)

            m_scores = m_scores[m_scores.uid.isin(common_uids)]
            n_scores = user_scores[user_scores.uid.isin(common_uids)]

            ABn = get_significantly_different_pairs(n_scores)
            ABm = get_significantly_different_pairs(m_scores)
            ABnm = ABn.join(ABm, how="inner", lsuffix="_n", rsuffix="_m")
            if len(ABnm) == 0:
                continue
            s_nqmab = np.abs(ABnm.score_a_m - ABnm.score_b_m) / np.abs(
                ABnm.score_a_n - ABnm.score_b_n
            )

            # To check: is it correct to subtract s_nqmab?
            delta_s_nqmab = (
                (
                    np.abs(ABnm.score_a_m - ABnm.score_b_m)
                    + ABnm.uncertainty_a_m
                    + ABnm.uncertainty_b_m
                )
                / (
                    np.abs(ABnm.score_a_n - ABnm.score_b_n)
                    - ABnm.uncertainty_a_n
                    - ABnm.uncertainty_b_n
                )
            ) - s_nqmab

            s = QrMed(1, 1, s_nqmab, delta_s_nqmab)
            s_nqm.append(s)
            delta_s_nqm.append(QrUnc(1, 1, 1, s_nqmab, delta_s_nqmab, qr_med=s))
            s_weights.append(scaling_weights[user_m])

        theta_inf = np.max(user_scores.score)
        s_nqm = np.array(s_nqm)
        delta_s_nqm = np.array(delta_s_nqm)
        s_dict[user_n] = 1 + BrMean(
            8 * W * theta_inf, np.array(s_weights), s_nqm - 1, delta_s_nqm
        )

    s_nq = pd.Series(s_dict)

    tau_dict = {}
    for user_n in users_to_compute:
        user_scores = df[df.user_id == user_n].drop("user_id", axis=1)
        tau_nqm = []
        delta_tau_nqm = []
        s_weights = []
        for user_m in (u for u in reference_users if u != user_n):
            m_scores = df[df.user_id == user_m].drop("user_id", axis=1)
            common_uids = set(user_scores.uid).intersection(m_scores.uid)

            if len(common_uids) == 0:
                continue

            m_scores = m_scores[m_scores.uid.isin(common_uids)]
            n_scores = user_scores[user_scores.uid.isin(common_uids)]

            tau_nqmab = s_nq[user_m] * m_scores.score - s_nq[user_n] * n_scores.score
            delta_tau_nqmab = (
                s_nq[user_n] * n_scores.uncertainty
                + s_nq[user_m] * m_scores.uncertainty
            )
            tau = QrMed(1, 1, tau_nqmab, delta_tau_nqmab)
            tau_nqm.append(tau)
            delta_tau_nqm.append(QrUnc(1, 1, 1, tau_nqmab, delta_tau_nqmab, qr_med=tau))
            s_weights.append(scaling_weights[user_m])

        tau_nqm = np.array(tau_nqm)
        delta_tau_nqm = np.array(delta_tau_nqm)
        tau_dict[user_n] = BrMean(8 * W, np.array(s_weights), tau_nqm, delta_tau_nqm)

    tau_nq = pd.Series(tau_dict)
    return s_nq, tau_nq


def compute_scaling_for_supertrusted():
    df = get_contributor_criteria_score(User.supertrusted_users(), poll_name="videos", criteria="largely_recommended")
    return compute_scaling(df)

def compute_scaling_for_all_users():
    raise NotImplementedError

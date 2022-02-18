import numpy as np
import pandas as pd

R_MAX = 10
ALPHA = 0.01

public_dataset = pd.read_csv('~/workspace/data/tournesol_public_export_2022-02-18.csv')

def compute_individual_score(username, criteria):

    df = public_dataset[
        (public_dataset.public_username==username)
        & (public_dataset.criteria==criteria)
    ]
    scores = df[["video_a","video_b","score"]]

    scores_sym = pd.concat([
        scores,
        pd.DataFrame({
            "video_a": scores.video_b,
            "video_b": scores.video_a,
            "score": -1 * scores.score,
        })
    ])

    r = scores_sym.pivot(index="video_a", columns="video_b", values="score")
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


    ## Compute uncertainties
    #
    # if len(scores) < 2:
    #     delta_star = None
    # else:
    #     theta_star_numpy = theta_star.to_numpy()
    #     theta_star_ab = pd.DataFrame(
    #         np.subtract.outer(theta_star_numpy, theta_star_numpy),
    #         index=theta_star.index,
    #         columns=theta_star.index
    #     )
    #     sigma2 = np.nansum(k * (l - theta_star_ab)**2) / 2 / (len(scores) - 1)
    #
    #     # FIXME: K.sum() always equals to ALPHA by definition.
    #     delta_star = np.sqrt(sigma2) / np.sqrt(K.sum(axis=1))


    # r.loc[a:b] is negative when a is prefered to b.
    # The sign of the result is inverted.
    return -1 * theta_star

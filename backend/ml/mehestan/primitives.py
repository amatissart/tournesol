import pandas as pd
import numpy as np
from scipy.optimize import bisect

EPSILON = 1e-6

def QrMed(W: float, w: pd.Series, x: pd.Series, delta: pd.Series):
    delta_2 = delta ** 2
    def L_prime(m: float):
        return W * m + np.sum(w * (x-m)/(np.sqrt(delta_2 + (x-m)**2)))
    
    m_low = -1.0
    while L_prime(m_low) > 0:
        m_low *= 2
    
    m_up = 1.0
    while L_prime(m_up) < 0:
        m_up *= 2

    return bisect(L_prime, m_low, m_up, xtol=EPSILON)


def QrDev(W: float, w: pd.Series, x: pd.Series, delta: pd.Series, m: float, default_dev: float):
    return default_dev + QrMed(W, w, np.abs(x-m) - default_dev, delta)


def QrUnc(W: float, qr_dev: float, w: pd.Series, x: pd.Series, delta: pd.Series, m: float, ):
    delta_2 = delta ** 2
    h = W + np.sum(w * np.minimum(1, delta_2 * (delta_2 + (x-m)**2)**(-3/2)))

    if h <= W:
        return qr_dev
    
    k = (h-W) ** (-1/2)
    return (np.exp(-qr_dev)*qr_dev + np.exp(-k)*k) / (np.exp(-qr_dev) + np.exp(-k))


def Clip(x: pd.Series, center, radius):
    return x.clip(center-radius, center+radius)

def ClipMean(w, x, center, radius):
    return np.sum(w * Clip(x, center, radius)) / np.sum(w)

def BrMean(W, w, x, delta):
    return ClipMean(w, x, center=QrMed(4*W, w, x, delta), radius=np.sum(w) / (4*W))

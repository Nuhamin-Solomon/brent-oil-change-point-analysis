import pymc as pm
import numpy as np

def bayesian_change_point(price_series):
    n = len(price_series)
    with pm.Model() as model:
        tau = pm.DiscreteUniform("tau", lower=0, upper=n-1)
        mu1 = pm.Normal("mu1", mu=np.mean(price_series), sigma=np.std(price_series))
        mu2 = pm.Normal("mu2", mu=np.mean(price_series), sigma=np.std(price_series))
        sigma = pm.HalfNormal("sigma", sigma=np.std(price_series))
        mu = pm.math.switch(tau >= np.arange(n), mu1, mu2)
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=price_series)
        trace = pm.sample(2000, tune=1000, target_accept=0.95, cores=1, random_seed=42)
    return model, trace
import sys
import os

# Add project root to sys.path
project_root = os.path.abspath("..")
if project_root not in sys.path:
    sys.path.append(project_root)

from src.analysis.change_point_model import bayesian_change_point

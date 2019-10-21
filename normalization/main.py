"""Run simulations."""

import numpy as np

from nistats.design_matrix import make_first_level_design_matrix
from nistats.first_level_model import run_glm

from events import simulate_events, load_events
from fmri_signal import simulate_signal
from normalization import normalization_matrix

# Simulation parameters
MODE = 'Simulate'  # Load or Simulate events


# 'Simulate' mode
t_r = 2.72
n_voxels = 1492+757
n_conditions = 10
n_trials = 3
n_rests = 5
n_tr_trials = 3
n_tr_rests = 6
n_scans = 123
# ---------

# 'Load' mode
events_path = 'data/events'
# ---------

# Design matrix
drift_model = 'polynomial'
drift_order = 0
hrf_model = 'glover'

# Signal simulation
N_voxels = 200
G = 10*np.eye(n_conditions+n_rests+drift_order+1)
# s_noise = 2.
s_spatial = 0.9




if MODE == 'Simulate':
    frame_times, events = simulate_events(n_conditions=n_conditions,
                                          n_scans=n_scans,
                                          t_r=t_r,
                                          n_trials=n_trials,
                                          n_rests=n_rests,
                                          n_tr_trials=n_tr_trials,
                                          n_tr_rests=n_tr_rests)

elif MODE == 'Load':
    frame_times, events = load_events(events_path)

else:
    exit()


X = make_first_level_design_matrix(frame_times, events,
                                    drift_model=drift_model,
                                    drift_order=drift_order,
                                    hrf_model=hrf_model)


def pipeline(X, N_voxels, G, s_noise, s_spatial):
    T, T_noised, B = simulate_signal(N_voxels, X, G, s_noise, s_spatial)
    labels, results = run_glm(T_noised, X, noise_model='ols')
    R = results[0.].resid
    Theta = results[0.].theta
    S = normalization_matrix(R, verbose=True)
    Theta_normalized = np.dot(Theta, S)

    # print(f'R\n{R}\n')
    # print(f'S\n{S}\n')
    # print(f'Theta\n{Theta}\n')
    # print(f'Theta normalized\n{Theta_normalized}\n')
    return Theta, Theta_normalized, B


estimates = dict()
for s_noise in [0.01, 1, 2, 3, 4, 5, 6, 7]:
    Theta, Theta_normalized, Theta_true = pipeline(X, N_voxels, G, s_noise, s_spatial)
    estimates[s_noise] = (Theta, Theta_normalized)
    # print(f'Theta shape : {Theta.shape}')
    # print(f'Theta normalized shape : {Theta_normalized.shape}')
    d1 = np.linalg.norm(Theta-Theta_true)
    d2 = np.linalg.norm(Theta_normalized-Theta_true)
    print(f'{s_noise} {d1:.2f} {d2:.2f}')

# print(estimates)

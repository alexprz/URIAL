"""Simulate or load events and build corresponding design matrix."""

import numpy as np
import pandas as pd


def simulate_events(n_conditions, n_scans, t_r, n_trials=1, n_rests=0, n_tr_trials=1, n_tr_rests=1):
    """Simulate events given context."""
    len_trials = n_tr_trials*t_r
    len_rest = n_tr_rests*t_r
    frame_times = np.arange(n_scans) * t_r

    conditions = np.array(['c{}'.format(k) for k in range(n_conditions) for _ in range(n_trials)])
    durations = len_trials*np.ones(n_trials*n_conditions)

    rests_conditions = np.array(['r{}'.format(k) for k in range(n_rests)])
    rests_durations = len_rest*np.ones(n_rests)

    conditions = np.concatenate((conditions, rests_conditions))
    durations = np.concatenate((durations, rests_durations))

    # 1: Build random permutation
    sigma = dict(enumerate(np.random.permutation(len(conditions))))
    sigma_inv = {v: k for k, v in sigma.items()}

    # 2: Shuffle the durations
    shuffled_durations = [durations[sigma[k]] for k in range(len(conditions))]

    # 3: Cumsum the durations to have the onsets
    shuffled_onsets = np.cumsum([0.]+shuffled_durations[:-1])

    # 4: Unshuffle the onsets to match with the conditions order
    onsets = np.array([shuffled_onsets[sigma_inv[k]] for k in range(len(conditions))])

    events = pd.DataFrame({'trial_type': conditions, 'onset': onsets, 'duration': durations})

    return frame_times, events


def load_events(path):
    """Load events and frame_times from file."""
    frame_times, events = None, None
    return frame_times, events

import numpy as np
import pandas as pd
import nilearn
from nilearn import plotting, surface
from nistats.first_level_model import FirstLevelModel, run_glm
from nistats.reporting import plot_design_matrix
from nistats.design_matrix import make_first_level_design_matrix
from nistats.contrasts import compute_contrast
import scipy
import sklearn
from sklearn.covariance import LedoitWolf
from time import time
import nibabel as nib
import copy

from joblib import Memory
cache_dir = 'cache_joblib'
mem = Memory(cache_dir)

import matplotlib
matplotlib.use('MacOSX')
from matplotlib import pyplot as plt

raw = 'Raw/'
preproc = 'Preproc/'
data = preproc
subject = 'sub-02/'

@mem.cache
def normalization_matrix(R, verbose=False):
    '''
        Compute the matrix containing the normalization factors to apply to the patterns.

        Args:
            R (array of shape (n_time_points, n_voxels)): Array containing residuals of your pattern activation.

        Returns:
            S (array of shape (n_voxels, n_voxels)): Array containing the normalization factor to apply to the patterns (e.g : pattern_normalized = pattern*S)
    '''
    n_time_points, _ = R.shape

    if verbose: print('Building residuals covariance matrix...', end=' ', flush=True)
    S = np.dot(R.transpose(), R)/n_time_points

    if verbose: print('Done\nShrinking covariance matrix...', end=' ', flush=True)
    S = sklearn.covariance.LedoitWolf().fit(S).covariance_

    if verbose: print('Done\nInverting covariance matrix...', end=' ', flush=True)
    S = np.linalg.inv(S)

    if verbose: print('Done\nSquare root invert covariance matrix...', end=' ', flush=True)
    S = scipy.linalg.sqrtm(S)
    if verbose:print('Done')

    return np.real(S)

def simulate_events(n_conditions, n_scans, t_r, n_trials=1, n_rests=0, n_tr_trials=1, n_tr_rests=1):
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

def fmri_simulation(n_voxels, n_conditions, X, G, sigma_noise, s):
    '''
        Performs the simulation of a first level analysis.

        Args:
            n_voxels (int): number of voxels
            n_conditions (int): number of conditions
            X (array of shape (n_time_points, n_conditions)): design matrix used to create the true signal from the simulated Betas coefficients and gaussian noise
            G (array of shape (n_conditions, n_conditions): Covariance matrix. Conditions are supposed to have the similarity structure determined by G.
            sigma_noise (float): Standard deviation of the Gaussian noise added to the signal obtained by multiplying X by the Betas.

        Returns:
            S (array of shape (n_time_points, n_voxels)): Array storing the simulated noised signal
            B (array of shape (n_conditions, n_voxels)): Array storing the simulated true Betas
            R (array of shape (n_time_points, n_voxels)): Array storing the simulated true residuals (R = S-X*B)
    '''
    B = np.random.multivariate_normal(np.zeros(n_conditions), G, n_voxels)
    
    # R = 
    np.dot(X, B)


    return S, B, R


if __name__ == '__main__':
    # fmri_img = nilearn.image.load_img(data+subject+'sub-02_task-NC2U_run-01_tFilter_None.100.0_run-1_sFilter_LP_7.577999999999999mm.nii.gz')
    # events = pd.read_table(raw+subject+'func/'+'sub-02_task-NC2U_run-1_events.tsv')
    # t_r = 1.5
    # n_scans = fmri_img.shape[-1]
    # frame_times = t_r * np.arange(n_scans)

    # design_matrix = make_first_level_design_matrix(frame_times,
    #                                            events=events,
    #                                            hrf_model='spm + derivative',
    #                                            period_cut=128
    #                                            )
    
    t_r = 2.72

    n_voxels = 1492+757
    n_conditions = 5
    n_trials = 3
    n_rests = 5
    n_tr_trials = 3
    n_tr_rests = 6
    n_scans=123

    frame_times, events = simulate_events(n_conditions=n_conditions,
                                          n_scans=n_scans,
                                          t_r=t_r,
                                          n_trials=n_trials,
                                          n_rests=n_rests,
                                          n_tr_trials=n_tr_trials,
                                          n_tr_rests=n_tr_rests)

    X = make_first_level_design_matrix(frame_times, events,
                                       drift_model='polynomial',
                                       drift_order=0,
                                       hrf_model='glover')

    plot_design_matrix(X)
    plt.show()

    G = np.eye(n_conditions)

    # fmri_img = nilearn.image.load_img(data+subject+'sub-02_task-NC2U_run-01_tFilter_None.100.0_run-1_sFilter_LP_7.577999999999999mm.nii.gz')
    # print(fmri_img.shape)

    # events = pd.read_table(raw+subject+'func/'+'sub-02_task-NC2U_run-1_events.tsv')
    # # mask = nilearn.image.load_image()
    # mask = nilearn.image.resample_to_img('HO_HG_left.nii', 'Raw/sub-02/anat/sub-02_T1w.nii.gz')
    # mask = nib.Nifti1Image(np.round(mask.get_data()), mask.affine)
    # # mask.to_filename('mask.nii')
    # #nilearn.plotting.plot_roi('HO_HG_left.nii')
    # # nilearn.plotting.plot_roi(mask, bg_img='Raw/sub-02/anat/sub-02_T1w.nii.gz')

    # #nilearn.plotting.plot_stat_map(data+subject+'sub-02_task-NC2U_run-01_tFilter_None.100.0_run-1_mean.nii.gz', bg_img='Raw/sub-02/anat/sub-02_T1w.nii.gz')
    # # nilearn.plotting.plot_roi(mask, bg_img=data+subject+'sub-02_task-NC2U_run-01_tFilter_None.100.0_run-1_mean.nii.gz')
    # # plotting.show()
    # print(mask.get_data())
    # t_r = 1.5

    # # fmri_glm = FirstLevelModel(t_r=t_r,
    # #                        noise_model='ar1',
    # #                        standardize=False,
    # #                        hrf_model='spm',
    # #                        drift_model='cosine',
    #                        # period_cut=160)

    # fmri_glm = FirstLevelModel(mask=mask, minimize_memory=False)

    # n_scans = fmri_img.shape[-1]
    # frame_times = t_r * np.arange(n_scans)

    # design_matrix = make_first_level_design_matrix(frame_times,
    #                                            events=events,
    #                                            hrf_model='spm + derivative',
    #                                            period_cut=128
    #                                            )

    # fmri_glm = fmri_glm.fit(fmri_img, design_matrices=design_matrix)

    # print(fmri_glm)

    # print(fmri_glm.results_)

    # print(fmri_glm.labels_)

    # for key, reg_results in fmri_glm.results_[0].items():
    #     print(reg_results.resid.shape)

    # # exit()

    # labels = fmri_glm.labels_[0]
    # estimates = fmri_glm.results_[0]


    # # fsaverage = nilearn.datasets.fetch_surf_fsaverage5()

    # # texture = surface.vol_to_surf(fmri_img, fsaverage.pial_right)

    # # n_scans = texture.shape[1]
    # # frame_times = t_r * np.arange(n_scans)

    # # design_matrix = make_first_level_design_matrix(frame_times,
    # #                                            events=events,
    # #                                            hrf_model='glover + derivative'
    # #                                            )

    # # labels, estimates = run_glm(texture.T, design_matrix.values)

    # time0 = time()

    # # print('Previous length')
    # # print(len(labels))
    # # print(len(estimates))

    # # # Remove first key
    # # for label, reg_result in estimates.items():
    # #     del estimates[label]
    # #     removed_label = label
    # #     break

    # # labels = np.delete(labels, np.where(labels == removed_label), axis=0)
    # # print('New length')
    # # print(len(labels))
    # # print(len(estimates))

    # n_voxels = len(labels)
    # n_tasks = design_matrix.shape[1]
    # T = fmri_img.shape[-1]

    # R = np.zeros((T, n_voxels))
    # Thetas = np.zeros((n_tasks, n_voxels))

    # # Index between global voxel index and local voxel index (atlas' area voxel index)
    # L = [(label, local_voxel_id) for label, reg_result in estimates.items() for local_voxel_id in range(reg_result.theta.shape[1])]
    # decode_voxel = dict(enumerate(L))
    # encode_voxel = {v: k for k, v in decode_voxel.items()}

    # print('Building residuals matrix...', end=' ', flush=True)
    # for label, reg_result in estimates.items():
    #     residuals = reg_result.resid
    #     _, n_voxels_in_label = residuals.shape

    #     for local_voxel_id in range(n_voxels_in_label):
    #         global_voxel_id = encode_voxel[label, local_voxel_id]
    #         R[:, global_voxel_id] = residuals[:, local_voxel_id]
    #         Thetas[:, global_voxel_id] = reg_result.theta[:, local_voxel_id]
    # print('Done')


    # # print('Building residuals covariance matrix...', end=' ', flush=True)
    # # S = np.dot(R.transpose(), R)/T
    # # print('Done')

    # # print('Shrinking covariance matrix...', end=' ', flush=True)
    # # S_shrinked = sklearn.covariance.LedoitWolf().fit(S).covariance_
    # # print('Done')

    # # print('Inverting covariance matrix...', end=' ', flush=True)
    # # invert_S = np.linalg.inv(S_shrinked)
    # # print('Done')

    # # print('Square root invert covariance matrix...', end=' ', flush=True)
    # # S2 = scipy.linalg.sqrtm(invert_S)
    # # print('Done')

    # # S2_real = np.real(S2)

    # # print('Normalizing activation patterns...', end=' ', flush=True)
    # # Thetas_normalized = np.dot(Thetas, S2_real)
    # # print('Done')


    # # print('R :\n', R)
    # # print('S :\n', S)
    # # print('S_shrinked :\n', S_shrinked)
    # # print('invert_S :\n', invert_S)
    # # print('S2 :\n', S2)
    # # print('S2_real :\n', S2_real)
    # # print('Thetas :\n', Thetas)
    # # print('Thetas_normalized :\n', Thetas_normalized)

    # S = normalization_matrix(R, verbose=True)
    # Thetas_normalized = np.dot(Thetas, S)

    # print('Writing in new GLM results...', end=' ', flush=True)
    # fmri_glm_normalized = copy.deepcopy(fmri_glm)
    # for label, reg_result in fmri_glm_normalized.results_[0].items():
    #     _, n_voxels_in_label = reg_result.theta.shape

    #     reg_result.theta_normalized = np.ones((n_tasks, n_voxels_in_label))

    #     for local_voxel_id in range(n_voxels_in_label):
    #         global_voxel_id = encode_voxel[label, local_voxel_id]
    #         reg_result.theta[:, local_voxel_id] = Thetas_normalized[:, global_voxel_id]

    # print('Done')

    # # for label, reg_result in fmri_glm_normalized.results_[0].items():
    # #     _, n_voxels_in_label = reg_result.theta.shape

    # #     for local_voxel_id in range(n_voxels_in_label):
    # #         for k in range(n_tasks):
    # #             print('{} {}'.format(fmri_glm.results_[0][label].theta[k, local_voxel_id], reg_result.theta[k, local_voxel_id]))

    # print('Normalization done in {}s'.format(time()-time0))


    # # Contrast computation

    # contrast_matrix = np.eye(design_matrix.shape[1])
    # # basic_contrasts = dict([(column, contrast_matrix[i]) for i, column in enumerate(design_matrix.columns)])
    # basic_contrasts = [contrast_matrix[i] for i in range(design_matrix.shape[1])]

    # z_map = fmri_glm.compute_contrast(basic_contrasts, output_type='z_score')
    # z_map_normalized = fmri_glm_normalized.compute_contrast(basic_contrasts, output_type='z_score')

    # plotting.plot_stat_map(
    #     z_map, bg_img=data+subject+'sub-02_task-NC2U_run-01_tFilter_None.100.0_run-1_mean.nii.gz',
    #      threshold=0.2, display_mode='z',
    #     cut_coords=3, black_bg=True)

    # plotting.plot_stat_map(
    #     z_map_normalized, bg_img=data+subject+'sub-02_task-NC2U_run-01_tFilter_None.100.0_run-1_mean.nii.gz',
    #      threshold=3, display_mode='z',
    #     cut_coords=3, black_bg=True)

    # plotting.show()




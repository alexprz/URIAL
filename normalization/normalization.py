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

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

raw = 'Raw/'
preproc = 'Preproc/'
data = preproc
subject = 'sub-02/'

if __name__ == '__main__':
    fmri_img = nilearn.image.load_img(data+subject+'sub-02_task-NC2U_run-01_tFilter_None.100.0_run-1_sFilter_LP_7.577999999999999mm.nii.gz')
    print(fmri_img.shape)

    t_r = 1.5

    fsaverage = nilearn.datasets.fetch_surf_fsaverage5()

    texture = surface.vol_to_surf(fmri_img, fsaverage.pial_right)

    n_scans = texture.shape[1]
    frame_times = t_r * np.arange(n_scans)

    design_matrix = make_first_level_design_matrix(frame_times,
                                               events=events,
                                               hrf_model='glover + derivative'
                                               )

    labels, estimates = run_glm(texture.T, design_matrix.values)

    time0 = time()

    print('Previous length')
    print(len(labels))
    print(len(estimates))

    # Remove first key
    for label, reg_result in estimates.items():
        del estimates[label]
        removed_label = label
        break

    labels = np.delete(labels, np.where(labels == removed_label), axis=0)
    print('New length')
    print(len(labels))
    print(len(estimates))

    n_voxels = len(labels)
    n_tasks = design_matrix.shape[1]
    T = fmri_img.shape[-1]

    R = np.zeros((T, n_voxels))
    Thetas = np.zeros((n_tasks, n_voxels))

    # Index between global voxel index and local voxel index (atlas' area voxel index)
    L = [(label, local_voxel_id) for label, reg_result in estimates.items() for local_voxel_id in range(reg_result.theta.shape[1])]
    decode_voxel = dict(enumerate(L))
    encode_voxel = {v: k for k, v in decode_voxel.items()}

    print('Building residuals matrix...', end=' ', flush=True)
    for label, reg_result in estimates.items():
        residuals = reg_result.resid
        _, n_voxels_in_label = residuals.shape

        for local_voxel_id in range(n_voxels_in_label):
            global_voxel_id = encode_voxel[label, local_voxel_id]
            R[:, global_voxel_id] = residuals[:, local_voxel_id]
            Thetas[:, global_voxel_id] = reg_result.theta[:, local_voxel_id]
    print('Done')


    print('Building residuals covariance matrix...', end=' ', flush=True)
    S = np.dot(R.transpose(), R)/T
    print('Done')

    print('Shrinking covariance matrix...', end=' ', flush=True)
    S_shrinked = sklearn.covariance.LedoitWolf().fit(S).covariance_
    print('Done')

    print('Inverting covariance matrix...', end=' ', flush=True)
    invert_S = np.linalg.inv(S_shrinked)
    print('Done')

    print('Square root invert covariance matrix...', end=' ', flush=True)
    S2 = scipy.linalg.sqrtm(invert_S)
    print('Done')

    S2_real = np.real(S2)

    print('Normalizing activation patterns...', end=' ', flush=True)
    Thetas_normalized = np.dot(Thetas, S2_real)
    print('Done')


    print('R :\n', R)
    print('S :\n', S)
    print('S_shrinked :\n', S_shrinked)
    print('invert_S :\n', invert_S)
    print('S2 :\n', S2)
    print('S2_real :\n', S2_real)
    print('Thetas :\n', Thetas)
    print('Thetas_normalized :\n', Thetas_normalized)

    print('Writing in GLM results...', end=' ', flush=True)
    for label, reg_result in estimates.items():
        _, n_voxels_in_label = reg_result.theta.shape

        for local_voxel_id in range(n_voxels_in_label):
            global_voxel_id = encode_voxel[label, local_voxel_id]

            reg_result.theta_normalized = np.zeros((n_tasks, n_voxels_in_label)) 
            reg_result.theta_normalized[:, local_voxel_id] = Thetas_normalized[:, global_voxel_id]
    print('Done')

    for label, reg_result in estimates.items():
        _, n_voxels_in_label = reg_result.theta.shape

        for local_voxel_id in range(n_voxels_in_label):
            for k in range(n_tasks):
                print('{} {}'.format(reg_result.theta[k, local_voxel_id], reg_result.theta_normalized[k, local_voxel_id]))

    print('Normalization done in {}s'.format(time()-time0))


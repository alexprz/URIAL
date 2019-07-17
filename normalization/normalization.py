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

    first_img = nilearn.image.index_img(fmri_img, 0)
    print(first_img.shape)

    # plotting.plot_stat_map(first_img)
    # plotting.show()

    fmri_glm = FirstLevelModel(t_r=7,
                           noise_model='ar1',
                           standardize=False,
                           hrf_model='spm',
                           drift_model='cosine',
                           period_cut=160)

    events = pd.read_table(raw+subject+'func/'+'sub-02_task-NC2U_run-1_events.tsv')

    print(len(events))
    # fmri_glm.fit(img, events)
    fmri_glm = fmri_glm.fit(fmri_img, events)

    print('')
    print(fmri_glm)

    # design_matrix = fmri_glm.design_matrices_[0]

    # dmn_contrast = np.array([1] + [0]*(design_matrix.shape[1]-1))

    # z_map = fmri_glm.compute_contrast(dmn_contrast, output_type='z_score')

    # plotting.plot_stat_map(z_map, threshold=1.0, title='Seed based GLM')
    # plotting.show()

    # plot_design_matrix(design_matrix)
    # plt.show()

    # print(run_glm(np.transpose(img), design_matrix))


    # print(fmri_glm.labels)

    # print(fmri_glm.resid())

    # from nistats.datasets import fetch_localizer_first_level
    # data = fetch_localizer_first_level()
    # fmri_img = data.epi_img

    # print(fmri_img)

    t_r = 1.5

    fsaverage = nilearn.datasets.fetch_surf_fsaverage5()

    texture = surface.vol_to_surf(fmri_img, fsaverage.pial_right)

    n_scans = texture.shape[1]
    frame_times = t_r * np.arange(n_scans)

    design_matrix = make_first_level_design_matrix(frame_times,
                                               events=events,
                                               hrf_model='glover + derivative'
                                               )
    print(design_matrix.shape)
    # plot_design_matrix(design_matrix)
    # plt.show()

    labels, estimates = run_glm(texture.T, design_matrix.values)

    time0 = time()


    print('Previous length')
    print(len(labels))
    print(len(estimates))

    # Remove first key
    for voxel_label, reg_result in estimates.items():
        del estimates[voxel_label]
        removed_label = voxel_label
        break

    labels = np.delete(labels, np.where(labels == removed_label), axis=0)
    print('New length')
    print(len(labels))
    print(len(estimates))

    # print(labels)
    # print(estimates)

    n_voxels = len(labels)
    n_tasks = design_matrix.shape[1]
    T = fmri_img.shape[-1]

    R = np.zeros((T, n_voxels))
    Thetas = np.zeros((n_tasks, n_voxels))

    # decode_voxel = dict(enumerate(labels))
    # encode_voxel = {v: k for k, v in decode_voxel.items()}

    # print(n_voxels)
    L = [(label, voxel_id_in_label) for label, reg_result in estimates.items() for voxel_id_in_label in range(reg_result.theta.shape[1])]
    # L = [(label, voxel_id_in_label) for label in labels for voxel_id_in_label in range(estimates[label].theta.shape[1])]
    # L = [label for label in labels]
    # print(L)
    # print(len(L))

    decode_voxel = dict(enumerate(L))
    encode_voxel = {v: k for k, v in decode_voxel.items()}

    # print(encode_voxel)

    print('Building residuals matrix...', end=' ', flush=True)
    for voxel_label, reg_result in estimates.items():
        residuals = reg_result.resid
        _, n_voxels_in_label = residuals.shape

        for voxel_id_in_label in range(n_voxels_in_label):
            global_voxel_id = encode_voxel[voxel_label, voxel_id_in_label]
            R[:, global_voxel_id] = residuals[:, voxel_id_in_label]
            Thetas[:, global_voxel_id] = reg_result.theta[:, voxel_id_in_label]
    print('Done')

    # print(R)

    label, id_voxel = decode_voxel[0]
    # print(estimates[label].resid)
    # print(estimates[label].resid.shape)
    # print(label)
    # print(labels[0])


    print('Building residuals covariance matrix...', end=' ', flush=True)
    S = np.dot(R.transpose(), R)/T
    print('Done')

    # print(S)
    # print(S.shape)

    # print(Thetas)
    # print(Thetas.shape)
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
    for voxel_label, reg_result in estimates.items():
        _, n_voxels_in_label = reg_result.theta.shape

        for voxel_id_in_label in range(n_voxels_in_label):
            global_voxel_id = encode_voxel[voxel_label, voxel_id_in_label]

            reg_result.theta_normalized = np.zeros((n_tasks, n_voxels_in_label)) 
            reg_result.theta_normalized[:, voxel_id_in_label] = Thetas_normalized[:, global_voxel_id]
    print('Done')

    # for voxel_label, reg_result in estimates.items():
    #     _, n_voxels_in_label = reg_result.theta.shape

    #     for voxel_id_in_label in range(n_voxels_in_label):
    #         for k in range(n_tasks):
    #             print('{} {}'.format(reg_result.theta[k, voxel_id_in_label], reg_result.theta_normalized[k, voxel_id_in_label]))

    print('Normalization done in {}s'.format(time()-time0))
    # print(encode_voxel)
    # print(decode_voxel)

    # print(labels) 
    # print(len(labels)) 
    # print(len(estimates))
    # print(estimates[0].resid)

    # k = 0
    # for voxel_label, reg_result in estimates.items():
    #     # print(key)
    #     # print(value)
    #     # print(value.resid)
    #     voxel_resid_over_time = reg_result.norm_resid
    #     # voxel_resid_over_time = np.mean(reg_result.resid, axis=1)
    #     # print(reg_result.resid)
    #     # print(voxel_resid_over_time)
    #     # print(voxel_label)
    #     # print(voxel_resid_over_time.shape)
    #     # break
    #     # R[:, encode_voxel[voxel_label]] = voxel_resid_over_time
    #     # print(voxel_resid_over_time.shape)
    #     # print(reg_result.predicted.shape)
    #     k += voxel_resid_over_time.shape[1]
    #     # print(reg_result.theta.shape)
    #     print(reg_result.cov.shape)

    # print(k)

    # I DON'T UNDERSTAND THE CHANGING SHAPE OF RESIDUALS !

    # print(R)

    # S = np.dot(R.transpose(), R)/T

    # print(S)


    # contrast_matrix = np.eye(design_matrix.shape[1])
    # basic_contrasts = dict([(column, contrast_matrix[i]) for i, column in enumerate(design_matrix.columns)])

    # for index, (contrast_id, contrast_val) in enumerate(basic_contrasts.items()):
    #     print('  Contrast % i out of %i: %s, right hemisphere' %
    #           (index + 1, len(basic_contrasts), contrast_id))
    #     contrast = compute_contrast(labels, estimates, contrast_val,
    #                                 contrast_type='t')
    #     z_score = contrast.z_score()
    #     # Plot the result
    #     plotting.plot_surf_stat_map(
    #         fsaverage, z_score)

    # plotting.show()

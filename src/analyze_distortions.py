import os
import os.path as op
import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np

from tools.image import vec_to_image, rescale, unprocess, clamp
from tools.utils import safe_mkdir
from matplotlib import cm
from tqdm import tqdm
from itertools import combinations
from scipy.linalg import subspace_angles
import pyrtools as pt
import seaborn as sns

""" Create distorted images with which to test in psychophysics experiment.
"""

DIR_DATA = op.join('/Users/lyndonduong/GoogleDriveNYU/Classes/2020 Classes/Bayesian Machine Learning/bml2020_project/data')
DIR_DATA = op.join('../data')

def display_first_and_last(image, synthesized_signal, alpha=5., beta=10.):
    r""" Displays the first and last synthesized eigendistortions alone, and added to the image.

    If image or eigendistortions have 3 channels, then it is assumed to be a color image and it is converted to
    grayscale. This is merely for display convenience and may change in the future.

    Parameters
    ----------
    alpha: float, optional
        Amount by which to scale eigendistortion for image + (alpha * eigendistortion) for display.
    beta: float, optional
        Amount by which to scale eigendistortion to be displayed alone.
    kwargs:
        Additional arguments for :meth:`pt.imshow()`.
    """

    assert len(synthesized_signal) > 1, "Assumes at least two eigendistortions were synthesized."

    image = image

    max_dist = synthesized_signal[0]
    min_dist = synthesized_signal[-1]

    fig_max = pt.imshow([unprocess(image),
                         unprocess(image + alpha * max_dist),
                         rescale(max_dist.squeeze().permute((1, 2, 0)))],
                        title=['original', f'original + {alpha:.0f} * maxdist', f'{alpha:.0f} * maxdist'],
                        vrange='indep1');

    fig_min = pt.imshow([unprocess(image),
                         unprocess(image + beta * min_dist),
                         rescale(min_dist.squeeze().permute((1, 2, 0)))],
                        title=['original', f'original + {beta:.0f} * mindist', f'{beta:.0f} * mindist'],
                        vrange='indep1');

    return fig_max, fig_min


def load_eigendistortions(img_name):
    """Load analyzed Jacobian SVD matrices from data folder"""
    assert img_name in ['church', 'dog', 'fish', 'horn', 'truck']
    img_name = img_name + '.pkl'
    all_models = sorted([f for f in os.listdir(op.join(DIR_DATA, 'eigendistortions')) if f.startswith('resnet')])
    jacobians = torch.zeros((len(all_models), 10, 3*256*256))
    for i, mdl_name in enumerate(all_models):
        with open(op.join(DIR_DATA, 'eigendistortions', mdl_name, img_name), 'rb') as f:
            d = pickle.load(f)

            U, S, V = d['U'].clone(), d['S'].clone(), d['V'].clone()

            jacobians[i] = U @ S @ V.T

    return jacobians


def plot_singular_spectra():
    """Plots mean and stdev singular spectra for different bootstrapped ensemble sizes"""

    jacobians = load_eigendistortions('dog')

    colors = cm.viridis(np.linspace(.1, 1, 20))
    n_boot = 20
    fig, ax = plt.subplots(1, 1)

    for j in tqdm(range(2, len(jacobians))):
        s_boot = torch.zeros(n_boot, 10)
        selector = torch.randint(0, len(jacobians), [n_boot, j])

        for i, boot in enumerate(tqdm(selector)):
            tmp = jacobians[boot].mean(0)
            _, S, _ = torch.svd(tmp)
            s_boot[i] = S

        ax.errorbar(range(10), s_boot.mean(0), s_boot.std(0), linewidth=3, color=colors[j])
    ax.set(xlabel='index', ylabel='singular value')


def compute_subspace_angles():
    """Computes average subspace angle between each pair of eigendistortions"""
    img_name = 'dog'
    jacobians = load_eigendistortions(img_name)
    combos = [comb for comb in combinations(range(20), 2)]
    combos = combos + [(i, i) for i in range(20)]

    all_angles = {}
    n_boot = 20
    save_dir = op.join(DIR_DATA, 'subspace_angles')
    filename = 'subspace_angles_' + img_name
    safe_mkdir(save_dir)
    if op.exists(op.join(save_dir, filename)):
        return

    for comb in tqdm(combos):
        tmp = []
        for _ in tqdm(range(n_boot)):
            ind1, ind2 = comb
            j1 = jacobians[torch.randint(0, 20, (ind1+1, ))]
            j2 = jacobians[torch.randint(0, 20, (ind2+1, ))]

            _, _, v1 = torch.svd(j1.mean(0))
            _, _, v2 = torch.svd(j2.mean(0))

            tmp.append(subspace_angles(v1, v2))

        all_angles.update({str(comb): tmp.copy()})

    with open(op.join(save_dir, filename + '.pkl'), 'wb') as f:
        pickle.dump(all_angles, f, pickle.HIGHEST_PROTOCOL)

    return all_angles


def plot_subspace_angles():
    data_dir = op.join(DIR_DATA, 'subspace_angles')
    filename = 'subspace_angles_dog'

    with open(op.join(data_dir, filename + '.pkl'), 'rb') as f:
        d = pickle.load(f)

    combos = [comb for comb in combinations(range(20), 2)]
    combos = combos + [(i, i) for i in range(20)]

    corr = np.zeros((20, 20))
    for combo in combos:
        tmp = np.stack(d[str(combo)])[:,0]  # nth eigenvector; delete sq brackets if want all eigenvecs

        corr[combo] = np.cos(tmp.mean())
    mask = np.zeros((20, 20))
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        ax = sns.heatmap(corr.T, mask=mask, vmin=0, vmax=1, square=True, cmap="YlGnBu_r",
                         cbar_kws={'label': 'avg cosine distance'})
        plt.xticks(range(1, 20))
        plt.yticks(range(1, 20))
        ax.set(xlabel='ensemble size', ylabel='ensemble size', xticklabels= [1] + [None]*17 +[19],
               title='eigenbasis similarity across different ensembles')
        plt.show()


def plot_eigendistortions():
    img_name = 'horn'
    mdl_name = 'resnet_num_1'

    with open(op.join(DIR_DATA, 'eigendistortions', mdl_name, img_name + '.pkl'), 'rb') as f:
        d = pickle.load(f)

    base_img = d['base_signal']
    jacobians = load_eigendistortions(img_name)
    _, _, V1 = torch.svd(jacobians[0])
    _, _, V2 = torch.svd(jacobians.mean(0))
    alpha = 60
    display_eigendistortion(base_img, V1[:,0], alpha=alpha)
    display_eigendistortion(base_img, V2[:,0], alpha=alpha)


def display_eigendistortion(image, dist_vec, alpha):
    """
    image:processed image,
    dist_vec: distortion
    alpha: int
    """
    dist = vec_to_image(dist_vec)

    distorted = image + alpha*dist
    distorted_np = unprocess(distorted)

    print(distorted[distorted<0].min(), distorted[distorted>1].max())
    dist2 = ((dist-dist.min())/(dist.max() - dist.min())).squeeze().permute((1,2,0))  # normalized distortion

    fig, ax = plt.subplots(1, 2, sharex='all', sharey='all')
    ax[0].imshow(clamp(distorted_np, 0, 1), vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    ax[1].imshow(clamp(dist2, 0, 1), vmin=0, vmax=1)


if __name__ == '__main__':
    # plot_singular_spectra()

    # compute_subspace_angles()
    # plot_subspace_angles()

    plot_eigendistortions()

import os
import os.path as op
import numpy as np
import torch
from torch import Tensor
from image import unprocess, clamp, preprocess_image, vec_to_image
import utils
import pickle
from typing import Dict, List

DIR_DATA = op.join('../../data')
PATH_IMAGE = op.join('../../data/to_synthesize')


def create_distortions() -> (Dict, Dict):
    """Create synthesized unit-vector eigendistortions for all images and models.
    Loads all jacobians made from synthesize_distortions, takes their weighted avg (by eigenvalue) to create single
    distortion.
    Saves in ../data/experiment_distortions.
    Loads from there if that directory exists.
    """

    save_dir = op.join(DIR_DATA, 'experiment_distortions')

    if op.exists(save_dir):
        print('Loading from pre-computed distortions')
        with open(op.join(save_dir, 'distortions_individual.pkl'), 'rb') as f:
            distortions_individual = pickle.load(f).copy()
        with open(op.join(save_dir, 'distortions_ensemble.pkl'), 'rb') as f:
            distortions_ensemble = pickle.load(f).copy()

    else:
        assert op.exists(op.join(DIR_DATA, 'eigendistortions')), "Must have all model/image Jacobians on disk."
        utils.safe_mkdir(op.join(DIR_DATA, 'experiment_distortions'))
        images = ['church', 'dog', 'fish', 'horn', 'truck']
        all_models = sorted([f for f in os.listdir(op.join(DIR_DATA, 'eigendistortions')) if f.startswith('resnet')])

        distortions_individual = dict()
        distortions_ensemble = dict()

        for img in images:  # iter thru all images
            img_name = img + '.pkl'
            jacobians_all = torch.zeros((10, 3 * 256 * 256))

            for mdl_name in all_models:  # get jac of each trained model
                with open(op.join(DIR_DATA, 'eigendistortions', mdl_name, img_name), 'rb') as f:
                    d = pickle.load(f)

                    u, s, v = d['U'].clone(), d['S'].clone(), d['V'].clone()

                    weighted_distortion_ind = torch.einsum('nk, k -> n', v, s.diag()**2)  # weight be eigenvalues
                    weighted_distortion_ind /= weighted_distortion_ind.norm()
                    weighted_distortion_ind = vec_to_image(weighted_distortion_ind.clone())
                    jacobians_all += u @ s @ v.T

            distortions_individual.update({img: weighted_distortion_ind})  # just grab the last one

            jacobians_all /= len(all_models)  # take the avg
            _, s_all, v_all = torch.svd(jacobians_all)  # compute avg eigendistortions

            weighted_distortion_all = torch.einsum('nk, k -> n', v_all, s.diag()**2)  # weighted by eigenvalues
            weighted_distortion_all /= weighted_distortion_all.norm()
            weighted_distortion_all = vec_to_image(weighted_distortion_all)
            distortions_ensemble.update({img: weighted_distortion_all.clone()})

        print('Saving distortions to', save_dir)
        with open(op.join(save_dir, 'distortions_individual.pkl'), 'wb') as f:
            pickle.dump(distortions_individual, f, pickle.HIGHEST_PROTOCOL)

        with open(op.join(save_dir, 'distortions_ensemble.pkl'), 'wb') as f:
            pickle.dump(distortions_ensemble, f, pickle.HIGHEST_PROTOCOL)

    return distortions_individual, distortions_ensemble


def load_images() -> dict:
    """Load all preprocessed images into memory for experiment.
    These are normalized using statistics of the training set, turned into torch Tensors, and permuted to (b,c,h,w).
    """
    images = ['church', 'dog', 'fish', 'horn', 'truck']

    preprocessed_images = {i: preprocess_image(op.join(PATH_IMAGE, i + '.jpeg')) for i in images}

    return preprocessed_images


def distort_image(base_image: Tensor, distortion: Tensor, alpha: float) -> np.ndarray:
    """Adds torch tensor distortion to tensor image, then transforms back to original image space, and clamps
    between (0, 1)"""
    return clamp(unprocess(base_image + alpha*distortion), 0, 1)


def random_choice(a: List):
    """Uses numpy to get random element of list or iterable"""
    i = np.random.randint(0, len(a))
    return a.copy()[i]


def random_trial(preprocessed_images: dict,
                 distortions_individual: dict,
                 distortions_ensemble: dict,
                 alpha: float,
                 img: str = None,
                 ensemble: bool = None):
    """
    Parameters
    ----------
    preprocessed_images: Dict
        Dict from load_images() containing all possible preprocessed base images
    distortions_individual
        Dict from create_distortions() containing all weighted individual distortions for each image
    distortions_ensemble
        Same as distortions_individual, but weighted ensemble distortions for each image.
    alpha: float
        Magnitude of distortion to add to base_image.
    img: str, optional

    ensemble, optional

    Returns
    -------
    base_image: np.ndarray
        Undistorted image with size (h, w, c)
    distorted_image: np.ndarray
        base_image + alpha*distortion, size (h, w, c)
    """

    if img is None:
        images = ['church', 'dog', 'fish', 'horn', 'truck']
        img = random_choice(images)
    if ensemble is None:
        ensemble = bool(np.random.randint(0, 2))

    base_image_torch = preprocessed_images[img]
    base_image = unprocess(base_image_torch)
    distortion = distortions_ensemble[img] if ensemble else distortions_individual[img]
    distorted_image = distort_image(base_image_torch, distortion, alpha).copy()
    return base_image, distorted_image


if __name__ == '__main__':
    # dist_ind, dist_ensemble = create_distortions()
    # preprocessed_images = load_images()
    #
    # np.random.seed(0)
    # alpha = 50.
    # base_image, distorted_image = random_trial(preprocessed_images, dist_ind, dist_ensemble, alpha=alpha)
    #
    # fig, ax = plt.subplots(1,2, sharex='all', sharey='all')
    # ax[0].imshow(base_image)
    # ax[1].imshow(distorted_image)
    # ax[0].set(title='image', xticks=[], yticks=[])
    # ax[1].set(title=f'image + {alpha} * distortion')
    # fig.show()

    print('test')
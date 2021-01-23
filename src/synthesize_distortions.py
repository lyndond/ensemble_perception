import os
import os.path as op
import torch
from torch import Tensor
from torch import nn
from torchvision.models import resnet18
import pickle
from tools.image import preprocess_image
from tools.utils import safe_mkdir
from src.eigendistortion import EigendistortionMulti
from tqdm import tqdm

""" Synthesize Eigendistortions of trained models.
"""

DIR_DATA = op.join('/data/')
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SEED = 0


class NthLayer(torch.nn.Module):
    """Wrap model to get the response of an intermediate layer
    Works for Resnet18
    """

    def __init__(self, model_dir, layer=None):
        """
        Parameters
        ----------
        model_dir: str
            Path to ResNet18 weights.
        layer: int
            Which model response layer to output.
        """
        super().__init__()
        model = resnet18(pretrained=False)
        model.fc = torch.nn.Linear(512, 10)
        state_dict = torch.load(model_dir, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()

        features = ([model.conv1, model.bn1, model.relu, model.maxpool] + [l for l in model.layer1] +
                    [l for l in model.layer2] + [l for l in model.layer3] + [l for l in model.layer4] +
                    [model.avgpool, model.fc])
        self.features = nn.ModuleList(features).eval()

        if layer is None:
            layer = len(self.features)
        self.layer = layer

    def forward(self, x: Tensor) -> Tensor:
        for ii, mdl in enumerate(self.features):
            x = mdl(x)
            if ii == self.layer:
                return x


def make_eigendistortions() -> None:
    all_models = sorted([f for f in os.listdir(op.join(DIR_DATA, 'models')) if f.startswith('resnet')])
    images = ['church', 'dog', 'fish', 'horn', 'truck']

    for im_name in tqdm(images):

        for which_model in tqdm(all_models):

            image_dir = op.join(DIR_DATA, 'to_synthesize', im_name + '.jpeg')
            model_dir = op.join(DIR_DATA, 'models', which_model, 'model.pt')
            save_dir = op.join(DIR_DATA, 'eigendistortions', which_model)
            file_name = f"{im_name}"
            if os.path.exists(op.join(save_dir, f"{file_name}.pkl")):  # skip if already exists
                continue
            safe_mkdir(save_dir)  # create folder to store synthesized images

            img = preprocess_image(image_dir, seed=SEED).to(DEVICE)

            # instantiate model and load trained weights
            model = resnet18(pretrained=False)
            model.fc = torch.nn.Linear(512, 10)
            state_dict = torch.load(model_dir, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.eval()

            # instantiate eigendistortion object and synthesize
            ed = EigendistortionMulti(model=model, base_signal=img)
            ed.synthesize(method='svd', k=10)

            U, S, V = ed.get_jacobian_svd()
            to_save = dict(U=U, S=S, V=V, base_signal=img, im_name=im_name, file_name=file_name)

            with open(op.join(save_dir, f"{file_name}.pkl"), 'wb') as f:
                pickle.dump(to_save, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    make_eigendistortions()

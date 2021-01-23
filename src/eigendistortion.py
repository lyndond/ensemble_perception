"""
Methods for synthesizing (multiple) eigendistortions
"""

from plenoptic.synthesize.autodiff import vector_jacobian_product, jacobian_vector_product
from plenoptic.synthesize.eigendistortion import Eigendistortion
from plenoptic.synthesize.eigendistortion import fisher_info_matrix_vector_product
import torch
from tqdm import tqdm
from torch import nn
from torch import Tensor


def fisher_info_matrix_eigenvalue(y, x, v):
    r"""Implicitly compute the eigenvalue of the Fisher Information Matrix corresponding to eigenvector v
    :math:`\lambda= v^T F v`
    """
    Fv = fisher_info_matrix_vector_product(y, x, v)
    lmbda = Fv.T @ v
    if v.shape[1] > 1:
        lmbda = torch.diag(lmbda)

    return lmbda


class _Temp(nn.Module):
    """Dummy linear model with which to test the Eigendistortion class below"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(30, 10, bias=False)

    def forward(self, x):
        return self.linear(x)


class EigendistortionMulti(Eigendistortion):
    def __init__(self, base_signal, model, device=torch.device('cpu')):
        super().__init__(base_signal, model)
        self._synthesis_complete = False
        self.U = None
        self.device = device

    def synthesize(self, k,method='power', n_steps=1000, tol=1e-10, store_progress=False):
        self.store_progress = store_progress

        if method == 'power':
            lmbda_new, v_new = self._synthesize_power(k, n_steps=n_steps, tol=tol)
        elif method == 'svd':  # random svd
            lmbda_new, v_new = self._synthesize_randomized_svd(k, 2, 2)

        eig_vecs = self._vector_to_image(v_new.detach())
        eig_vals = lmbda_new.squeeze()
        eig_vecs_ind = [0, len(self._input_flat) - 1]

        self.synthesized_signal = torch.stack(eig_vecs, 0) if len(eig_vecs) != 0 else []

        self.synthesized_eigenvalues = eig_vals.detach()
        self.synthesized_eigenindex = eig_vecs_ind

        self._synthesis_complete = True
        return self.synthesized_signal, self.synthesized_eigenvalues, self.synthesized_eigenindex

    def _synthesize_power(self, k, tol=1e-10, n_steps=1000):
        r""" Use (block) power method to obtain largest (smallest) eigenvalue/vector pair.
        Apply the power method algorithm to approximate the extremal eigenvalue and eigenvector of the Fisher
        Information Matrix, without explicitly representing that matrix.

        Parameters
        ----------
        k: int
            Number of eigenvectors to compute.
        tol: float, optional
            Tolerance value
        n_steps: int, optional
            Maximum number of steps

        Returns
        -------
        lmbda: float
            Eigenvalue corresponding to final vector of power iteration.
        v: torch.Tensor
            Final eigenvector (i.e. eigendistortion) of power iteration procedure.
        """

        idx = len(self._all_losses)
        if self.store_progress:
            self._all_losses.append([])

        x, y = self._input_flat, self._representation_flat

        v = torch.randn(len(x), k).to(self.device)
        v = v / v.norm(dim=0)

        Fv = fisher_info_matrix_vector_product(y, x, v)
        v = Fv / torch.norm(Fv)
        lmbda = fisher_info_matrix_eigenvalue(y, x, v)

        d_lambda = torch.tensor(1)
        lmbda_new, v_new = None, None
        pbar = tqdm(range(n_steps))
        postfix_dict = {'step': None, 'delta_eigenval': None}
        for i in pbar:
            postfix_dict.update(dict(step=f"{i + 1:d}/{n_steps:d}", delta_eigenval=f"{d_lambda.item():04.4f}"))
            pbar.set_postfix(**postfix_dict)

            if d_lambda <= tol:
                print(f"Tolerance {tol:.2E} reached. Stopping early.")
                break

            Fv = fisher_info_matrix_vector_product(y, x, v)
            v_new, _ = torch.qr(Fv)  # orthogonalize

            lmbda_new = fisher_info_matrix_eigenvalue(y, x, v_new)

            d_lambda = (lmbda - lmbda_new).norm()
            v = v_new
            lmbda = lmbda_new

            self._clamp_and_store(idx, v.clone(), d_lambda.clone())

        pbar.close()

        return lmbda_new, v_new

    def _synthesize_randomized_svd(self,
                                   r: int,
                                   p: int = 0,
                                   q: int = 0,
                                   device=torch.device('cpu')) -> (Tensor, Tensor):
        x, y = self._input_flat, self._representation_flat
        n = len(x)

        P = torch.randn(n, r+p).to(self.device)
        Z = fisher_info_matrix_vector_product(y, x, P)

        for _ in range(q):  # optional power iteration to squeeze the spectrum for more accurate estimate
            XZ = fisher_info_matrix_vector_product(y, x, Z)
            Z = fisher_info_matrix_vector_product(y, x, XZ)

        Q, _ = torch.qr(Z)
        Y = fisher_info_matrix_vector_product(y, x, Q).T
        _, S, V = torch.svd(Y, some=True)
        # U = Q @ Uy

        return S[:r], V[:, :r]

    def _clamp_and_store(self, idx, v, d_lambda):
        """Overwrite base class _clamp_and_store. We don't actually need to clamp the signal."""
        if self.store_progress:
            self._all_losses[idx].append(d_lambda.item())

    def get_jacobian_svd(self):
        """ Computes the (possibly reduced) SVD of the Jacobian. Assumes we already have the eigenvecs and vals of the
        Fisher matrix. We just need to push the eigenvecs in V scaled by the inverse singular values in order to get
        the left singular vecs.
        A = USV.T
        u = A @ v @ s^-1
        where S is sqrt(Lambda)
        """

        assert self._synthesis_complete, "Synthesize eigendistortions first"
        x, y = self._input_flat, self._representation_flat
        eig_vecs, Lambda = self.synthesized_signal, self.synthesized_eigenvalues

        U, V = [], []
        S = Lambda.sqrt().diag()

        for (v, s) in zip(eig_vecs, Lambda):
            v = v.flatten().unsqueeze(-1)
            s = s.sqrt()

            U.append(jacobian_vector_product(y, x, v/s).squeeze())
            V.append(v.squeeze())

        U = torch.stack(U, 1)
        V = torch.stack(V, 1)

        return U, S, V


def synthesize_randomized_svd(x, y,
                              r: int,
                              p: int = 0,
                              q: int = 0,
                              device=torch.device('cpu')) -> (Tensor, Tensor):
    """Prototyping """
    n = len(x)
    P = torch.randn(n, r + p).to(device)
    Z = fisher_info_matrix_vector_product(y, x, P)
    for _ in range(q):  # optional power iteration
        XZ = fisher_info_matrix_vector_product(y, x, Z)
        Z = fisher_info_matrix_vector_product(y, x, XZ)
    Q, _ = torch.qr(Z)
    Y = fisher_info_matrix_vector_product(y, x, Q).T
    _, S, V = torch.svd(Y, some=True)

    return S[:r], V[:, :r]


if __name__ == '__main__':
    torch.manual_seed(0)
    U, _ = torch.qr(torch.randn((10, 10)))
    V, _ = torch.qr(torch.randn((30, 30)))
    V = V[:, :10]
    s = torch.arange(1, 11).float()
    s = torch.randn(10).sort()[0]
    S = torch.sort(s)[0].diag()
    M = U @ S @ V.T
    tmp = _Temp()
    tmp.linear.weight.data = M

    k = 10
    x = torch.ones(30, 1).requires_grad_(True)
    y = tmp(x.T).T
    Z = torch.ones((30, k))

    for _ in range(0):
        Z = fisher_info_matrix_vector_product(y, x, Z)
        Z, _ = torch.qr(Z)

    lamb = fisher_info_matrix_eigenvalue(y, x, Z)

    ed = EigendistortionMulti(x.clone().view((1, 1, 1, 30)), tmp)
    ed.synthesize(k, method='svd')
    # plt.imshow(V.T @ ed.synthesized_signal.squeeze().T)
    # plt.show()

    S2, V2 = synthesize_randomized_svd(x, y, 10, 2, 2)

    U3, S3, V3, = ed.get_jacobian_svd()

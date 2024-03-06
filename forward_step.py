# import clip
# from PIL import Image
# import torch
# import torch.nn.functional as F
# from torch.autograd import Variable
#
# import numpy as np
# from torch.nn import Module
# from math import pi, sqrt
# import numpy as np
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
#
# image = preprocess(Image.open("frame_0001.jpg")).unsqueeze(0).to(device)
# text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
#
#
# def is_positive_definite(matrix):
#     try:
#         torch.linalg.cholesky(matrix)
#         return True
#     except RuntimeError:
#         return False
#
#
# with torch.no_grad():
#     image_features = model.encode_image(image)
#
#
# class ComputeLoss:
#     def __init__(self, model, lambda_energy, lambda_cov, device, n_gmm):
#         self.model = model
#         self.lambda_energy = lambda_energy
#         self.lambda_cov = lambda_cov
#         self.device = device
#         self.n_gmm = n_gmm
#         self.mmd = FastMMD(500, 1.0)
#
#     def forward(self, z1_front, z1_back, gamma1, z2_front, z2_back, gamma2):
#         """Computing the loss function for DAGMM."""
#         # z1_front = torch.squeeze(z1_front, dim=1)
#         # z2_front = torch.squeeze(z2_front, dim=1)
#         # print("_____________________")
#         # print("z1_front", z1_front.size())
#         # print("gamma111", gamma1.size())
#
#         sample_energy1, cov_diag1 = self.compute_energy(z1_front, gamma1)
#         sample_energy2, cov_diag2 = self.compute_energy(z2_front, gamma2)
#         print("________________________________")
#         print(sample_energy1)
#         print(self.lambda_energy * sample_energy1)
#         print(self.lambda_cov * cov_diag1)
#         print(self.lambda_energy * sample_energy2)
#         print(self.lambda_cov * cov_diag2)
#         print(self.mmd(z1_back, z2_back))
#         loss = (
#                 self.lambda_energy * sample_energy1 + self.lambda_cov * cov_diag1 + self.lambda_energy * sample_energy2 +
#                 self.lambda_cov * cov_diag2 + self.mmd(z1_back, z2_back))
#         return Variable(loss, requires_grad=True)
#
#     def compute_energy(self, z, gamma, phi=None, mu=None, cov=None, sample_mean=True):
#         """Computing the sample energy function"""
#         if (phi is None) or (mu is None) or (cov is None):
#             phi, mu, cov = self.compute_params(z, gamma)
#         z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))
#
#         eps = 1e-12
#         cov_inverse = []
#         det_cov = []
#         cov_diag = 0
#         for k in range(self.n_gmm):
#             print("cov[k]",cov[k].size())
#             cov_k = cov[k] + (torch.eye(cov[k].size(-1)) * eps).to(self.device)
#             cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
#             print("cov_k.size()", cov_k.size())
#             print("Cholesky.apply(cov_k.cpu() * (2 * np.pi))",Cholesky.apply(cov_k.cpu() * (2 * np.pi)))
#             det_cov.append((Cholesky.apply(cov_k.cpu() * (2 * np.pi)).diag().prod()).unsqueeze(0))
#             cov_diag += torch.sum(1 / cov_k.diag())
#
#         cov_inverse = torch.cat(cov_inverse, dim=0)
#         det_cov = torch.cat(det_cov).to(self.device)
#
#         E_z = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
#         E_z = torch.exp(E_z)
#         # print("phi",phi)
#         # print("det_cov",det_cov)
#         # print("E_z1",E_z)
#         E_z = -torch.log(torch.sum(phi.unsqueeze(0) * E_z / (torch.sqrt(det_cov)).unsqueeze(0), dim=1) + eps)
#         # print("E_z",E_z)
#
#         if sample_mean == True:
#             E_z = torch.mean(E_z)
#         return E_z, cov_diag
#
#     def compute_params(self, z, gamma):
#         """Computing the parameters phi, mu and gamma for sample energy function """
#         # K: number of Gaussian mixture components
#         # N: Number of samples
#         # D: Latent dimension
#         #  z = NxD
#         # gamma = NxK
#         z = z.squeeze(0).permute(1, 0)
#         print("z",z.size())
#         phi = torch.sum(gamma, dim=0) / gamma.size(0)
#         # mu = KxD
#         mu = torch.sum(z.unsqueeze(1) * gamma.unsqueeze(-1), dim=0)
#         mu /= torch.sum(gamma, dim=0).unsqueeze(-1)
#         print("_--------------------------------")
#         print("mu", mu.size())
#         print("gamma", gamma.size())
#         z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))
#         print("z_mu", z_mu.size())
#
#         z_mu_z_mu_t = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
#         print("z_mu_z_mu_t", z_mu_z_mu_t.size())
#
#         # cov = K x D x D
#         cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_z_mu_t, dim=0)
#         cov /= torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)
#         print("z_mu_z_mu_t", phi.size(), mu.size(), cov.size())
#         return phi, mu, cov
#
#
# class Cholesky(torch.autograd.Function):
#     def forward(ctx, a):
#         l = torch.linalg.cholesky(a)
#         ctx.save_for_backward(l)
#         return l.squeeze()
#
#     def backward(ctx, grad_output):
#         l, = ctx.saved_variables
#         linv = l.inverse()
#         inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
#             1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
#         s = torch.mm(linv.t(), torch.mm(inner, linv))
#         return s
#
#
# class FastMMD(Module):
#     """ Fast Maximum Mean Discrepancy approximated using the random kitchen sinks method.
#     """
#
#     def __init__(self, out_features, gamma):
#         super().__init__()
#         self.gamma = gamma
#         self.out_features = out_features
#
#     def forward(self, a, b):
#         in_features = a.shape[-1]
#
#         # W sampled from normal
#         w_rand = torch.randn((in_features, self.out_features), device=a.device)
#         # b sampled from uniform
#         b_rand = torch.zeros((self.out_features,), device=a.device).uniform_(0, 2 * pi)
#
#         phi_a = self._phi(a, w_rand, b_rand).mean(dim=0)
#         phi_b = self._phi(b, w_rand, b_rand).mean(dim=0)
#         mmd = torch.norm(phi_a - phi_b, 2)
#
#         return mmd
#
#     def _phi(self, x, w, b):
#         scale_a = sqrt(2 / self.out_features)
#         scale_b = sqrt(2 / self.gamma)
#         out = scale_a * (scale_b * (x @ w + b)).cos()
#         return out
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


def is_positive_definite(matrix):
    try:
        torch.linalg.cholesky(matrix)
        return True
    except RuntimeError:
        return False

class ComputeLoss:
    def __init__(self, model, lambda_energy, lambda_cov, device, n_gmm):
        self.model = model
        self.lambda_energy = lambda_energy
        self.lambda_cov = lambda_cov
        self.device = device
        self.n_gmm = n_gmm

    def forward(self, x, x_hat, z, gamma):
        """Computing the loss function for DAGMM."""
        reconst_loss = torch.mean((x - x_hat).pow(2))

        sample_energy, cov_diag = self.compute_energy(z, gamma)

        loss = reconst_loss + self.lambda_energy * sample_energy + self.lambda_cov * cov_diag
        print("reconst_loss",reconst_loss)
        print("sample_energy",self.lambda_energy * sample_energy)
        print("cov_diag",self.lambda_cov * cov_diag)

        return Variable(loss, requires_grad=True)

    def compute_energy(self, z, gamma, phi=None, mu=None, cov=None, sample_mean=True):
        """Computing the sample energy function"""
        if (phi is None) or (mu is None) or (cov is None):
            phi, mu, cov = self.compute_params(z, gamma)

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))

        eps = 1e-12
        cov_inverse = []
        det_cov = []
        cov_diag = 0
        for k in range(self.n_gmm):
            # print("cov[k]", cov[k].size())

            cov_k = cov[k] + (torch.eye(cov[k].size(-1)) * eps).to(self.device)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            det_cov.append((Cholesky.apply(cov_k.cpu() * (2 * np.pi)).diag().prod()).unsqueeze(0))
            cov_diag += torch.sum(1 / cov_k.diag())

        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov = torch.cat(det_cov).to(self.device)

        E_z = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        E_z = torch.exp(E_z)
        E_z = -torch.log(torch.sum(phi.unsqueeze(0) * E_z / (torch.sqrt(det_cov)).unsqueeze(0), dim=1) + eps)
        if sample_mean == True:
            E_z = torch.mean(E_z)
        return E_z, cov_diag

    def compute_params(self, z, gamma):
        """Computing the parameters phi, mu and gamma for sample energy function """
        # K: number of Gaussian mixture components
        # N: Number of samples
        # D: Latent dimension
        #  z = NxD
        # gamma = NxK

        # phi = D
        phi = torch.sum(gamma, dim=0) / gamma.size(0)

        # mu = KxD
        mu = torch.sum(z.unsqueeze(1) * gamma.unsqueeze(-1), dim=0)
        mu /= torch.sum(gamma, dim=0).unsqueeze(-1)
        # print("_--------------------------------")
        # print("mu", mu.size())
        # print("z", z.size())
        # print("gamma", gamma.size())
        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))
        # print("z_mu", z_mu.size())

        z_mu_z_mu_t = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        # print("z_mu_z_mu_t", z_mu_z_mu_t.size())

        # cov = K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_z_mu_t, dim=0)
        cov /= torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)
        # print("z_mu_z_mu_t", phi.size(), mu.size(), cov.size())
        return phi, mu, cov


class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        # print("a",a.size())
        l = torch.cholesky(a, False)
        ctx.save_for_backward(l)
        return l

    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s

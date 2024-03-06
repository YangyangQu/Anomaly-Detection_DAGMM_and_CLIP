import torch
from torch import optim
import numpy as np
from barbar import Bar

from model import DAGMM
from forward_step import ComputeLoss
from utils.utils import weights_init_normal
import os

from forward_step import ComputeLoss
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score
from sklearn.metrics import roc_auc_score


class TrainerDAGMM:
    """Trainer class for DAGMM."""

    def __init__(self, args, data, device):
        self.args = args
        self.train_loader, self.test_loader = data
        self.device = device

    def train(self):
        os.makedirs('models', exist_ok=True)

        """Training the DAGMM model"""
        self.model = DAGMM(self.args.n_gmm, self.args.latent_dim).to(self.device)
        self.model.apply(weights_init_normal)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.compute = ComputeLoss(self.model, self.args.lambda_energy, self.args.lambda_cov,
                                   self.device, self.args.n_gmm)

        self.model.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for x, _ in Bar(self.train_loader):
                x = x.float().to(self.device)
                optimizer.zero_grad()

                _, x_hat, z, gamma = self.model(x)

                loss = self.compute.forward(x, x_hat, z, gamma)
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                optimizer.step()

                total_loss += loss.item()
            print('Training DAGMM... Epoch: {}, Loss: {:.3f}'.format(
                epoch, total_loss / len(self.train_loader)))
            if epoch % 50 == 0:
                model_path = os.path.join('models', 'dagmm_model_epoch_{}.pt'.format(epoch))
                torch.save(self.model.state_dict(), model_path)

    def eval(self):
        """Testing the DAGMM model"""
        self.model = DAGMM(self.args.n_gmm, self.args.latent_dim).to(self.device)
        model_path = "models/dagmm_model_epoch_{}.pt".format(self.args.num_epochs - 50)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print('Testing...')
        compute = ComputeLoss(self.model, None, None, self.device, self.args.n_gmm)
        with torch.no_grad():
            N_samples = 0
            gamma_sum = 0
            mu_sum = 0
            cov_sum = 0
            # Obtaining the parameters gamma, mu and cov using the trainin (clean) data.
            for x, _ in self.train_loader:
                x = x.float().to(self.device)

                _, _, z, gamma = self.model(x)
                phi_batch, mu_batch, cov_batch = compute.compute_params(z, gamma)

                batch_gamma_sum = torch.sum(gamma, dim=0)
                gamma_sum += batch_gamma_sum
                mu_sum += mu_batch * batch_gamma_sum.unsqueeze(-1)
                cov_sum += cov_batch * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)

                N_samples += x.size(0)

            train_phi = gamma_sum / N_samples
            train_mu = mu_sum / gamma_sum.unsqueeze(-1)
            train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

            # Obtaining Labels and energy scores for train data
            energy_train = []
            labels_train = []
            for x, y in self.train_loader:
                x = x.float().to(self.device)

                _, _, z, gamma = self.model(x)
                sample_energy, cov_diag = compute.compute_energy(z, gamma, phi=train_phi,
                                                                 mu=train_mu, cov=train_cov,
                                                                 sample_mean=False)

                energy_train.append(sample_energy.detach().cpu())
                labels_train.append(y)
            energy_train = torch.cat(energy_train).numpy()
            labels_train = torch.cat(labels_train).numpy()

            # Obtaining Labels and energy scores for test data
            energy_test = []
            labels_test = []
            for x, y in self.test_loader:
                x = x.float().to(self.device)

                _, _, z, gamma = self.model(x)
                sample_energy, cov_diag = compute.compute_energy(z, gamma, train_phi,
                                                                 train_mu, train_cov,
                                                                 sample_mean=False)

                energy_test.append(sample_energy.detach().cpu())
                labels_test.append(y)
            energy_test = torch.cat(energy_test).numpy()
            labels_test = torch.cat(labels_test).numpy()

            scores_total = np.concatenate((energy_train, energy_test), axis=0)
            labels_total = np.concatenate((labels_train, labels_test), axis=0)

        threshold = np.percentile(scores_total, 100 - 20)
        pred = (energy_test > threshold).astype(int)
        gt = labels_test.astype(int)
        precision, recall, f_score, _ = prf(gt, pred, average='binary')
        print("Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(precision, recall, f_score))
        print('ROC AUC score: {:.2f}'.format(roc_auc_score(labels_total, scores_total) * 100))
        return labels_total, scores_total

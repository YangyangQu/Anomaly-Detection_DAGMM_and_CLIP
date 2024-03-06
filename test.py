import torch
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score

from forward_step import ComputeLoss
from barbar import Bar
from torch.utils.data import Dataset, DataLoader


def eval(model, dataloader_train, dataloader_test, device, n_gmm):
    """Testing the DAGMM model"""
    model.eval()
    print('Testing...')
    compute = ComputeLoss(model, None, None, device, n_gmm)
    dataloader_train = DataLoader(dataloader_train, batch_size=32, shuffle=True)
    dataloader_test = DataLoader(dataloader_test, batch_size=32, shuffle=True)

    with torch.no_grad():
        N_samples = 0
        gamma_sum = 0
        mu_sum = 0
        cov_sum = 0
        # Obtaining the parameters gamma, mu and cov using the trainin (clean) data.

        for x, _ in Bar(dataloader_train):
            x = x.float().to(device)
            _, _, z, gamma = model(x)
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

        for x, y in Bar(dataloader_train):
            x = x.float().to(device)
            _, _, z, gamma = model(x)
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

        for x, y in Bar(dataloader_test):
            x = x.float().to(device)
            _, x_hat, z, gamma = model(x)
            sample_energy, cov_diag = compute.compute_energy(z, gamma, train_phi,
                                                             train_mu, train_cov,
                                                             sample_mean=False)

            energy_test.append(sample_energy.detach().cpu())
            labels_test.append(y)
            print("labels_test shape:", y.shape)  # 打印所有标签的尺寸

        energy_test = torch.cat(energy_test).numpy()
        labels_test = torch.cat(labels_test).numpy()
        print("labels_train shape:", labels_train.shape)  # 打印所有标签的尺寸
        print("labels_test shape:", labels_test[:,0,0,0])  # 打印所有标签的尺寸

        scores_total = np.concatenate((energy_train, energy_test), axis=0)
        labels_total = np.concatenate((labels_train, labels_test), axis=0)



    threshold = np.percentile(scores_total, 100 - 20)
    pred = (energy_test > threshold).astype(int)
    gt = labels_test.astype(int)

    gt = gt[:,0,0,0]
    print("gt value:", gt)
    labels_total = labels_total[:,0,0,0]

    print("gt shape:", gt.shape)  # 打印真实标签的尺寸
    print("pred shape:", pred.shape)  # 打印预测标签的尺寸

    print("energy_train shape:", energy_train.shape)  # 打印训练集上的能量值的尺寸
    print("energy_test shape:", energy_test.shape)  # 打印测试集上的能量值的尺寸
    print("labels_total shape:", labels_total)  # 打印所有标签的尺寸
    print("scores_total shape:", scores_total)  # 打印所有能量值的尺寸

    precision, recall, f_score, _ = prf(gt, pred, average='micro')
    print("Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(precision, recall, f_score))
    # print('ROC AUC score: {:.2f}'.format(roc_auc_score(labels_total, scores_total) * 100))
    return labels_total, scores_total

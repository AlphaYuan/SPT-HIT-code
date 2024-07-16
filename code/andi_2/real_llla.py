import torch
from torch import nn, optim, autograd
from torch.nn import functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.utils.data
#import torchvision.transforms as transforms
#import torchvision.datasets as datasets
#from models import resnet_orig as resnet
#from models import hendrycks as resnet_oe
#from util.evaluation import get_auroc
#from util import hessian
from backpack import backpack, extend
from backpack.extensions import KFAC
from math import *
from tqdm import tqdm, trange
import numpy as np
import backpack.extensions as backpack_extensions


def get_hessian(device,model, train_loader, mnist=False):
    W = list(model.parameters())[-4]
    b = list(model.parameters())[-3]
    m, n = W.shape
    lossfunc = nn.CrossEntropyLoss()

    extend(lossfunc, debug=False)
    net = nn.ModuleList()

    #net.append(model.module.output_layer_m)
    #net.append(model.module.output_layer_a)
    #extend(net if not mnist else model.fc2, debug=False)
    extend(model.output_layer_m if not mnist else model.fc2, debug=False)

    #n_data = len(train_loader.dataset)
    #with backpack_extensions(KFAC([net])):
    with backpack(KFAC()):
        U, V = torch.zeros(m, m).to(device), torch.zeros(n, n).to(device)
        B = torch.zeros(m, m).to(device)

        for i, batch in tqdm(enumerate(train_loader)):
        # for i, (x, y) in enumerate(train_loader):
            #x, y = x.cuda(), y.cuda()
            X, targets_m, targets_a, padding_masks, IDs=batch


            model.zero_grad()
            predictions_m, predictions_a, _ ,_= model(X.to(device), padding_masks.to(device))
            targets=torch.reshape(targets_m,(len(targets_m.to(device)),))
            loss=lossfunc(predictions_m.to(device), targets.to(device))
            #loss = torch.reshape(loss, (len(loss), 1))

            #batch_loss = torch.sum(loss)
            #mean_loss = batch_loss / len(loss)
            loss.backward()

            #lossfunc(model(X), targets_m).backward()

            with torch.no_grad():
                # Hessian of weight
                U_, V_ = W.kfac
                B_ = b.kfac[0]

                rho = min(1-1/(i+1), 0.95)

                U = rho*U + (1-rho)*U_
                V = rho*V + (1-rho)*V_
                B = rho*B + (1-rho)*B_

    n_data = len(train_loader.dataset)

    M_W = W.t()
    M_b = b
    U = sqrt(n_data)*U
    V = sqrt(n_data)*V
    B = n_data*B

    return [M_W, M_b, U, V, B]


# @torch.no_grad()
def estimate_variance(device,var0, hessians, invert=True):
    if not invert:
        return hessians

    tau = 1/var0

    with torch.no_grad():
        M_W, M_b, U, V, B = hessians

    m, n = U.shape[0], V.shape[0]

    # Add priors
    U_ = U + torch.sqrt(tau)*torch.eye(m).to(device)
    V_ = V + torch.sqrt(tau)*torch.eye(n).to(device)
    B_ = B + tau*torch.eye(m).to(device)

    # Covariances for Laplace
    U_inv = torch.inverse(V_)  # Interchanged since W is transposed
    V_inv = torch.inverse(U_)
    B_inv = torch.inverse(B_)

    return [M_W, M_b, U_inv, V_inv, B_inv]


def gridsearch_var0(device,model, hessians, val_loader, ood_loader, interval, n_classes=10, lam=1):
    #targets = torch.cat([y for x, y in val_loader], dim=0).cuda()
    targets = torch.cat([targets_m for (x, targets_m, targets_a, padding_masks, IDs) in val_loader], dim=0)
    vals, var0s = [], []
    pbar = tqdm(interval)

    for var0 in pbar:
        M_W, M_b, U, V, B = estimate_variance(device,var0, hessians)

        preds = predict(device,val_loader, model, M_W, M_b, U, V, B, 10)
        preds_out = predict(device,ood_loader, model, M_W, M_b, U, V, B, 10)
        targets_1 = torch.reshape(targets, (len(targets.to(device)),))
        loss_in = F.nll_loss(torch.log(preds + 1e-8).to(device), targets_1.to(device))
        # loss_out = torch.mean(torch.sum(-1/n_classes * torch.log(preds_out + 1e-8), 1))
        loss_out = -torch.log(preds_out + 1e-8).mean()
        # loss_out = -(preds_out*torch.log(preds_out)).sum(1).mean()
        loss = loss_in + lam*loss_out

        vals.append(loss)
        var0s.append(var0)

        pbar.set_description(f'var0: {var0:.5f}, Loss-in: {loss_in:.3f}, Loss-out: {loss_out:.3f}, Loss: {loss:.3f}')

    best_var0 = var0s[np.argmin(vals)]

    return best_var0


@torch.no_grad()
def predict(device,dataloader, model, M_W, M_b, U, V, B, n_samples=100, delta=1, apply_softmax=True):
    py = []

    for batch in dataloader:
        #x, y = delta*x.cuda(), y.cuda()
        x, targets_m, targets_a, padding_masks, IDs=batch
        x= delta * x

        predictions_m, predictions_a, phi ,_= model(x.to(device), padding_masks.to(device))


        #phi = model.feature_extr(x)

        mu_pred = phi @ M_W + M_b
        Cov_pred = torch.diag(phi @ U @ phi.t()).view(-1, 1, 1) * V.unsqueeze(0) + B.unsqueeze(0)

        post_pred = MultivariateNormal(mu_pred, Cov_pred)

        # MC-integral
        py_ = 0

        for _ in range(n_samples):
            f_s = post_pred.rsample()
            py_ += torch.softmax(f_s, 1) if apply_softmax else f_s

        py_ /= n_samples

        py.append(py_)

    return torch.cat(py, dim=0)

#用于分类任务 计算交叉熵误差的损失函数  （数据类型不限于long）
class NoFussCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long().squeeze(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

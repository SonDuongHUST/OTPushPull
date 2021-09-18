import torch
import torch.nn as nn


class OTPushPullLoss(nn.Module):
    def __init__(self, lamda, alpha, epsilon=0.01):
        super(OTPushPullLoss, self).__init__()
        self.lamda = lamda
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, c, z, kantorovich_value_c, kantorovich_value_z):
        loss_Pc_Pz = self.entropic_wasserstein_distance(c, z, kantorovich_value_c)
        loss_Pz_Pcz = self.ot_pushpull_loss(c, z, kantorovich_value_z)
        loss = loss_Pc_Pz + self.alpha * loss_Pz_Pcz
        return loss

    @staticmethod
    def cosine_distances(x, y):  # x and y are normalized
        distances = torch.matmul(x, y.T)
        return distances

    def entropic_wasserstein_distance(self, c, z, kantorovich_value_c):
        distances = self.cosine_distances(c, z)
        exp_term = (- distances + kantorovich_value_c) / self.epsilon
        log_term = torch.mean(torch.exp(exp_term), 0)
        entropic_ws = torch.mean(kantorovich_value_c) + \
                      torch.mean(-self.epsilon * torch.log(log_term))
        return entropic_ws

    def ot_pushpull_loss(self, c, z, kantorovich_value_z):
        loss_zc = self.entropic_wasserstein_distance(z, c, kantorovich_value_z, self.epsilon)
        loss_zz = self.entropic_wasserstein_distance(z, z, kantorovich_value_z, self.epsilon)
        loss = self.lamda * loss_zc + (1 - self.lamda) * loss_zz
        return loss

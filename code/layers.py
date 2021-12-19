import torch.nn as nn
from interfaces import *

class DPLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear):
        super().__init__()
        self.weight = linear_layer.weight.detach()
        self.bias = linear_layer.bias.detach()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

    def forward(self, x):
        mat_lb = torch.cat((torch.cat((self.weight, self.bias.unsqueeze(1)), 1),
                            torch.cat((torch.zeros([self.weight.shape[1]]), torch.tensor([1])), 0).unsqueeze(0)), 0)
        mat_ub = mat_lb
        x.update_hist(mat_lb, mat_ub)
        x.backsubstitution(True)
        x.backsubstitution(False)
        return x

class DPSPU(nn.Module):
    def __init__(self, in_features):
        super(DPSPU, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        self.slope_l = torch.nn.Parameter(torch.zeros(in_features))
        self.slope_u = torch.nn.Parameter(torch.zeros(in_features))
        self.slope_l_use=torch.zeros(in_features)
        self.slope_u_use=torch.zeros(in_features)
        self.tanh= nn.Tanh()
    def diff_clamp(self,x,a,b):
        return self.tanh(x)*(b-a)/2+(b+a)/2
    def forward(self, x):
        # clip?
        # with torch.no_grad():
        ubb = x.ub.detach()
        lbb = x.lb.detach()
        spu_ub = SPU(ubb)
        spu_lb = SPU(lbb)
        g_ub = SPU_grad(ubb)
        g_lb = SPU_grad(lbb)
        mask_1, mask_2 = lbb.ge(0), ubb.le(0)
        mask_3 = ~(mask_1 | mask_2)
        a = (spu_ub - spu_lb) / (ubb - lbb + epsilon)
        self.slope_u_use=self.slope_u*1
        self.slope_l_use=self.slope_l*1
        self.slope_u_use[mask_1] = self.diff_clamp(self.slope_u[mask_1],a[mask_1], a[mask_1])
        self.slope_l_use[mask_1] = self.diff_clamp(self.slope_l[mask_1],g_lb[mask_1], g_ub[mask_1])

        self.slope_u_use[mask_2] =  self.diff_clamp(self.slope_u[mask_2],g_ub[mask_2], g_lb[mask_2])
        self.slope_l_use[mask_2] = self.diff_clamp(self.slope_l[mask_2],a[mask_2], a[mask_2])

        self.slope_u_use[mask_3] = self.diff_clamp(self.slope_u[mask_3],(torch.ones_like(a) * -0.25)[mask_3],
                                                           torch.maximum(torch.zeros_like(a), a)[mask_3])
        self.slope_l_use[mask_3] = self.diff_clamp(self.slope_l[mask_3],((spu_lb + 0.5) / (lbb + epsilon))[mask_3],
                                                            g_ub[mask_3])

        lb, ub = x.lb, x.ub
        l_bias = choose_bias(lb, ub, self.slope_l_use)
        u_bias = choose_bias(lb, ub, self.slope_u_use, False)
        mat_lb = torch.cat((torch.cat((torch.diag(self.slope_l_use), l_bias.unsqueeze(1)), 1),
                            torch.cat((torch.zeros_like(l_bias), torch.tensor([1])), 0).unsqueeze(0)), 0)
        mat_ub = torch.cat((torch.cat((torch.diag(self.slope_u_use), u_bias.unsqueeze(1)), 1),
                            torch.cat((torch.zeros_like(u_bias), torch.tensor([1])), 0).unsqueeze(0)), 0)
        x.update_hist(mat_lb, mat_ub)
        return x


class Validator(nn.Module):
    def __init__(self, in_features, true_label=None):
        super(Validator, self).__init__()
        self.in_features = in_features
        self.out_features = in_features - 1
        self.true_label = true_label

    def forward(self, x):
        ids = -1 * torch.diag(torch.ones(self.in_features - 1))
        true_label = torch.ones(self.in_features - 1).unsqueeze(1)
        zeros = torch.zeros_like(true_label)
        mat_lb = torch.cat((torch.cat((ids[:, :self.true_label], true_label, ids[:, self.true_label:], zeros), 1),
                            torch.cat((torch.zeros([self.in_features]), torch.tensor([1])), 0).unsqueeze(0)), 0)
        mat_ub = mat_lb
        x.update_hist(mat_lb, mat_ub)
        x.backsubstitution(True)
        # print("Output: ", x.lb.min())
        return x.lb

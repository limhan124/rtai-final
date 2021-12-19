import torch

epsilon = 2.22e-308

def SPU(x):
    y = torch.where(x.ge(0), torch.pow(x, 2) - 0.5, -1 * torch.sigmoid(x))
    return y

def SPU_grad(x):
    g = torch.where(x.ge(0), 2 * x, torch.mul(torch.sigmoid(x), (torch.sigmoid(x)-1)))
    return g

def inv_SPU(k):
    k = k.detach()
    x = torch.where(k.gt(-0.25) & k.lt(0),
                    torch.log(1 - torch.sqrt(1 + 4 * k)) - torch.log(1 + torch.sqrt(1 + 4 * k)),
                    k / 2)
    return x

def get_bias(lb, ub, k):
    # print(lb, ub, k)
    b_l = SPU(lb) - torch.mul(k, lb)
    b_u = SPU(ub) - torch.mul(k, ub)

    t_x = inv_SPU(k)
    mask_1 = torch.bitwise_and(lb < t_x, ub > t_x)
    b_t = torch.tensor([torch.nan] * len(lb))
    b_t[mask_1] = SPU(t_x[mask_1]) - torch.mul(k[mask_1], t_x[mask_1])

    mask_2 = torch.bitwise_and(lb < 0, ub > 0)
    b_0 = torch.tensor([torch.nan] * len(lb))
    b_0[mask_2] = -0.5
    return [b_l, b_u, b_0, b_t]

def choose_bias(lb, ub, k, lower=True):
    bs = get_bias(lb, ub, k)
    bias = torch.tensor([torch.inf] * len(lb))
    if lower:
        for b in bs:
            mask = b.lt(bias)
            bias[mask] = b[mask]
    else:
        bias = bias * -1
        for b in bs:
            mask = b.gt(bias)
            bias[mask] = b[mask]
    return bias


class X:
    def __init__(self, lb, ub):
        self.lb = lb.unsqueeze(0)
        self.ub = ub.unsqueeze(0)
        # mat_lb, mat_ub, is_input
        self.hist_mats = [[torch.cat((lb, torch.tensor([1])), 0), torch.cat((ub, torch.tensor([1])), 0), True]]

    def update_hist(self, mat_lb, mat_ub):
        self.hist_mats.append([mat_lb, mat_ub, False])

    def backsubstitution(self, lower=True):
        M = self.hist_mats[-1][0]
        if not lower:
            # mat_lb
            M = self.hist_mats[-1][1]
        for i in range(len(self.hist_mats) - 2, -1, -1):
            layer = self.hist_mats[i]
            pos = M * (M >= 0)
            neg = M * (M < 0)
            m_l = layer[0]
            m_u = layer[1]
            if not lower:
                m_l = layer[1]
                m_u = layer[0]
            if not layer[-1]:
                # m*m
                M = pos.matmul(m_l) + neg.matmul(m_u)
            else:
                # m*v
                if lower:
                    self.lb = (pos.matmul(m_l) + neg.matmul(m_u))[:-1]
                else:
                    self.ub = (pos.matmul(m_l) + neg.matmul(m_u))[:-1]

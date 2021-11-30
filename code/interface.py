# net: neural network
# inputs in the format of Tensor([[[..],[..],[..]]])
# interval_i=[x_i-eps,x_i+eps]
# true_label is stored in the test case file, if the result is not true_label, still not verified
import copy
import numpy as np
import torch
import os

epsilon = 2.22e-308
def get_lowers_and_uppers(flatten_inputs, eps):
    # paras
    # flatten_inputs: np.array
    # eps: float point
    lowers = []
    uppers = []
    for i in flatten_inputs[0]:
        lower = max(i.numpy() - eps, 0.0)
        upper = min(i.numpy() + eps, 1.0)
        lowers.append(lower)
        uppers.append(upper)
    return lowers, uppers


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def SPU(x):
    if x <= 0:
        return -1 * sigmoid(x)
    else:
        return x * x - 0.5
def root_N(x):
    if x<=0:
        return x
    else:
        return x**(2**(1/10))

def fuse(x):
    if x<=0:
        return -1 * sigmoid(x)
    else:
        return x-0.5

def SPU_Gradient(x):
    if x <= 0:
        k = sigmoid(x) * (sigmoid(x) - 1)
    else:
        k = 2 * x
    b = -k * x + SPU(x)
    return k, b

def Root_N_Gradient(x):
    if x <= 0:
        k = 1
    else:
        k = (2**(1/10))*x**(2**(1/10)-1)
    b = -k * x + root_N(x)
    return k, b

def line_between_two_p(lower_x, upper_x):
    lower_y = SPU(lower_x)
    upper_y = SPU(upper_x)
    k = (lower_y - upper_y) / (lower_x - upper_x + epsilon)
    b = lower_y - k * lower_x
    return k, b

def line_between_two_p_root_N(lower_x, upper_x):
    lower_y = root_N(lower_x)
    upper_y = root_N(upper_x)
    k = (lower_y - upper_y) / (lower_x - upper_x + epsilon)
    b = lower_y - k * lower_x
    return k, b

def line_between_two_p_fuse(lower_x, upper_x):
    lower_y = fuse(lower_x)
    upper_y = fuse(upper_x)
    k = (lower_y - upper_y) / (lower_x - upper_x + epsilon)
    b = lower_y - k * lower_x
    return k, b
def relaxation_between_two_p_root_N(lower_x, upper_x):
    #case 1
        # print('median',(lower_x+upper_x)/2)
    # return (1,0),(1,0)
    if lower_x >= 0:
        k_upper, b_upper = line_between_two_p_root_N(lower_x, upper_x)
        # print('mid',((lower_x**(-4/5)+upper_x**(-4/5))/2)**(-5/4))
        k_lower, b_lower = Root_N_Gradient(upper_x)
    elif upper_x <= 0:
        k_upper, b_upper = 1, 0
        k_lower, b_lower = 1, 0
    elif lower_x < 0 and upper_x > 0:
        if upper_x<1:
            k_upper, b_upper=1,0
            k_lower, b_lower = line_between_two_p_root_N(lower_x, upper_x)
        else:
            k_upper, b_upper = line_between_two_p_root_N(lower_x, upper_x)
            k_lower, b_lower = Root_N_Gradient(upper_x)
    # k_upper, b_upper = line_between_two_p_root_N(lower_x, upper_x)
    return (k_lower, b_lower), (k_upper, b_upper)

def relaxation_between_test(lower_x, upper_x):
    return(1,0),(1,0)

def relaxation_between_two_p_fuse(lower_x, upper_x):
    # case 1
    if lower_x >= 0:
        k_lower, b_lower = 1,-0.5
        k_upper, b_upper = 1,-0.5
    # case 2
    elif upper_x <= 0:
        k_lower, b_lower = line_between_two_p_fuse(lower_x, upper_x)
        k_upper = 0
        b_upper = max(fuse(lower_x), fuse(upper_x))
    # case 3
    elif lower_x < 0 and upper_x > 0:
        k, b = line_between_two_p_fuse(lower_x, upper_x)
        k_l, b_l = SPU_Gradient(lower_x)
        if k_l > k:
            k_upper = k_l
            b_upper = b_l
        else:
            k_upper = k
            b_upper = b
        # if |l| < |u|
        if (lower_x + upper_x) / 2 < 0:
            k_lower, b_lower = line_between_two_p_fuse(lower_x, 0)
        else:
            k_lower, b_lower = 0,-0.5
    return (k_lower, b_lower), (k_upper, b_upper)

def relaxation_between_two_p(lower_x, upper_x,mean_value):

    stat_max=mean_value[0]
    stat_min=mean_value[1]

    if lower_x<stat_min-2:
        lower_x+=2
    if upper_x>stat_max-2:
        upper_x-=2
    #case 1
    if lower_x >= 0:
        # print('median',(lower_x+upper_x)/2)

        k_lower, b_lower = SPU_Gradient((lower_x+upper_x)/2) ## the best tangent line
        k_upper, b_upper = line_between_two_p(lower_x, upper_x)
    # case 2
    elif upper_x <= 0:
        k_lower, b_lower = line_between_two_p(lower_x, upper_x)
        k_upper = 0
        b_upper = max(SPU(lower_x), SPU(upper_x))
    # case 3
    elif lower_x < 0 and upper_x > 0:
        k, b = line_between_two_p(lower_x, upper_x)
        k_l, b_l = SPU_Gradient(lower_x)
        if k_l > k:
            k_upper = k_l
            b_upper = b_l
        else:
            k_upper = k
            b_upper = b
        # if |l| < |u|
        if (lower_x+upper_x)/2 < 0:
            k_lower, b_lower = line_between_two_p(lower_x, 0)
        else:
            k_lower, b_lower = SPU_Gradient((lower_x+upper_x)/2)  ## the best tangent line
    return (k_lower, b_lower), (k_upper, b_upper)


def compute_SPU(lowers, uppers,huer,SPU_COUNTER):
    # paras
    # lowers,uppers: np.array
    '''
    Since the heuristic of our method is bounded by lines:
    We have four cases:
    case 1: l>0
    case 2: u<0
    case 3: l<0<u (there is a special case in case 3, tangent at l is needed)
    :return: k,b : slope and bias of line (scalar) y=kx+b
    if the line is horizontal, k=0
    '''
    if len(lowers) != len(uppers) or len(lowers) < 10:
        return None
    size = len(lowers) + 1
    lower_matrix = np.zeros((size, size), dtype=np.float64)
    upper_matrix = np.zeros((size, size), dtype=np.float64)
    for i in range(size - 1):
        lower = lowers[i]
        upper = uppers[i]
        (k_lower, b_lower), (k_upper, b_upper) = relaxation_between_two_p(lower, upper,huer[SPU_COUNTER][i])
        lower_matrix[i][i] = k_lower
        lower_matrix[i][size - 1] = b_lower
        upper_matrix[i][i] = k_upper
        upper_matrix[i][size - 1] = b_upper
    lower_matrix[size - 1][size - 1] = 1.0
    upper_matrix[size - 1][size - 1] = 1.0
    return torch.from_numpy(lower_matrix).to(torch.float64), torch.from_numpy(upper_matrix).to(torch.float64)

def compute_SPU_func(lowers, uppers,method=relaxation_between_two_p):
    # paras
    # lowers,uppers: np.array
    '''
    Since the heuristic of our method is bounded by lines:
    We have four cases:
    case 1: l>0
    case 2: u<0
    case 3: l<0<u (there is a special case in case 3, tangent at l is needed)
    :return: k,b : slope and bias of line (scalar) y=kx+b
    if the line is horizontal, k=0
    '''
    if len(lowers) != len(uppers) or len(lowers) < 10:
        return None
    size = len(lowers) + 1
    lower_matrix = np.zeros((size, size), dtype=np.float64)
    upper_matrix = np.zeros((size, size), dtype=np.float64)
    for i in range(size - 1):
        lower = lowers[i]
        upper = uppers[i]
        (k_lower, b_lower), (k_upper, b_upper) = method(lower, upper)
        lower_matrix[i][i] = k_lower
        lower_matrix[i][size - 1] = b_lower
        upper_matrix[i][i] = k_upper
        upper_matrix[i][size - 1] = b_upper
    lower_matrix[size - 1][size - 1] = 1.0
    upper_matrix[size - 1][size - 1] = 1.0
    return torch.from_numpy(lower_matrix).to(torch.float64), torch.from_numpy(upper_matrix).to(torch.float64)


def my_mv(m, vv_l, vv_u, upper=True):
    # paras
    # m: tensor_matrix
    # v_l, v_u: np.array
    reses = []
    for i in range(m.shape[0] - 1):
        v_l = copy.deepcopy(vv_l)
        v_u = copy.deepcopy(vv_u)
        line = m[i].numpy().reshape(1, -1)[0]
        if upper:
            v_u[line < 0] = v_l[line < 0]
            reses.append(np.dot(line, v_u))
        else:
            v_l[line < 0] = v_u[line < 0]
            reses.append(np.dot(line, v_l))
    return np.array(reses, dtype=np.float64)


def my_mm(m_1, mm_2_l, mm_2_u, upper=True):
    if not torch.equal(mm_2_l, mm_2_u):
        reses = []
        for i in range(m_1.shape[0] - 1):
            m_2_l = copy.deepcopy(mm_2_l)
            m_2_u = copy.deepcopy(mm_2_u)
            cur_line = m_1[i]
            line = cur_line.numpy().reshape(1, -1)[0]
            if upper:
                m_2_u[line < 0] = m_2_l[line < 0]
                length = len(line)
                reses.append(torch.mm(cur_line.reshape(1, -1).to(torch.float64), m_2_u.to(torch.float64))
                             .reshape(length))
            else:
                m_2_l[line < 0] = m_2_u[line < 0]
                length = len(line)
                reses.append(torch.mm(cur_line.reshape(1, -1).to(torch.float64), m_2_l.to(torch.float64))
                             .reshape(length))
        reses = torch.stack(reses)
        return torch.cat((reses, torch.cat((torch.zeros([m_2_l.shape[1] - 1]), torch.tensor([1])), 0)
                          .reshape(1, -1)), 0)
    else:
        # m_2 from affine, m_2_l = m_2_u
        return torch.mm(m_1.to(torch.float64), mm_2_l.to(torch.float64))


class Analyzer:
    '''
    The main analyzer
    '''

    def __init__(self, net, inputs, eps, true_label,heur):
        self.eps = eps
        self.net = net
        lowers, uppers = get_lowers_and_uppers(torch.flatten(inputs, start_dim=1), eps)
        # np.array
        self.input_lowers = [((i - 0.1307) / 0.3081) for i in lowers]
        self.input_lowers.append(1)
        self.input_lowers = np.array(self.input_lowers, dtype=np.float64)
        # np.array
        self.heur=heur
        self.input_uppers = [((i - 0.1307) / 0.3081) for i in uppers]
        self.input_uppers.append(1)
        self.input_uppers = np.array(self.input_uppers, dtype=np.float64)
        # np.arrays
        self.L = []
        # np.arrays
        self.U = []
        # tensor_matrixs
        self.A_L = []
        # tensor_matrixs
        self.A_U = []
        num_layers = len(net.state_dict()) // 2
        self.num_layers = 2 * num_layers + 1
        self.true_label = true_label
        self.SPU_COUNTER=0
    def backsubstitution(self):
        for i in range(self.num_layers):
            if i == 0:
                l = self.input_lowers
                u = self.input_uppers
                self.L.append(l[:-1])
                self.U.append(u[:-1])
            else:
                if i == self.num_layers - 1:
                    M = []
                    # 9 formular
                    for k in range(10):
                        if k == self.true_label:
                            continue
                        line = torch.zeros([11])
                        line[self.true_label] = 1
                        line[k] = -1
                        M.append(line)
                    M.append(torch.cat((torch.zeros([10]), torch.tensor([1])), 0))
                    M_L = torch.stack(M)
                    M_U = M_L
                    self.A_L.append(M_L)
                    self.A_U.append(M_U)
                    for j in range(len(self.A_L) - 1, -1, -1):
                        if j == 0:
                            lowers = self.input_lowers
                            uppers = self.input_uppers
                            l = my_mv(M_L, lowers, uppers, False)
                            u = my_mv(M_U, lowers, uppers, True)
                            self.L.append(l)
                            self.U.append(u)
                            if i == self.num_layers - 1:
                                return l, u
                        else:
                            m_l = self.A_L[j - 1]
                            m_u = self.A_U[j - 1]
                            M_L = my_mm(M_L, m_l, m_u, False)
                            M_U = my_mm(M_U, m_l, m_u, True)
                elif i % 2 == 1:
                    weight = self.net.state_dict()["layers.{}.weight".format(i + 1)]
                    bias = self.net.state_dict()["layers.{}.bias".format(i + 1)].reshape(-1, 1)
                    length = weight.shape[1]
                    M_L = torch.cat((torch.cat((weight, bias), 1),
                                     torch.cat((torch.zeros([length]), torch.tensor([1])), 0).reshape(1, -1)), 0)
                    M_U = M_L
                    self.A_L.append(M_L)
                    self.A_U.append(M_U)
                    for j in range(len(self.A_L) - 1, -1, -1):
                        if j == 0:
                            lowers = self.input_lowers
                            uppers = self.input_uppers
                            l = my_mv(M_L, lowers, uppers, False)
                            u = my_mv(M_U, lowers, uppers, True)
                            self.L.append(l)
                            self.U.append(u)
                            if i == self.num_layers - 1:
                                return l, u
                        else:
                            m_l = self.A_L[j - 1]
                            m_u = self.A_U[j - 1]
                            M_L = my_mm(M_L, m_l, m_u, False)
                            M_U = my_mm(M_U, m_l, m_u, True)
                else:
                    # ----- compute_SPU receive two np.array 1d and return a tensor matrix -----


                    spu_l = self.L[len(self.L)- 1]
                    spu_u = self.U[len(self.L) - 1]

                    M_L, M_U = compute_SPU(spu_l, spu_u,self.heur,SPU_COUNTER=self.SPU_COUNTER)
                    self.SPU_COUNTER+=1
                    self.A_L.append(M_L)
                    self.A_U.append(M_U)
                    for j in range(len(self.A_L) - 1, -1, -1):
                        if j == 0:
                            lowers = self.input_lowers
                            uppers = self.input_uppers
                            l = my_mv(M_L, lowers, uppers, False)
                            u = my_mv(M_U, lowers, uppers, True)
                            self.L.append(l)
                            self.U.append(u)
                            if i == self.num_layers - 1:
                                return l, u
                        else:
                            m_l = self.A_L[j - 1]
                            m_u = self.A_U[j - 1]
                            M_L = my_mm(M_L, m_l, m_u, False)
                            M_U = my_mm(M_U, m_l, m_u, True)




    def test(self):
        net = {
            "layers.2.weight": torch.tensor([[1, 1], [1, -1]], dtype=torch.float64),
            "layers.2.bias": torch.tensor([0, 0], dtype=torch.float64),
            "layers.4.weight": torch.tensor([[1, 1], [1, -1]], dtype=torch.float64),
            "layers.4.bias": torch.tensor([-0.5, 0], dtype=torch.float64),
            "layers.6.weight": torch.tensor([[-1, 1], [0, 1]], dtype=torch.float64),
            "layers.6.bias": torch.tensor([3, 0], dtype=torch.float64)
        }
        for i in range(7):
            if i == 0:
                l = np.array([-1, -1, 1], dtype=np.float64)
                u = np.array([1, 1, 1], dtype=np.float64)
                self.L.append(l)
                self.U.append(u)
            else:
                if i == 6:
                    M_L = torch.tensor([[1, -1, 0], [0, 0, 1]], dtype=torch.float64)
                    M_U = M_L
                if i % 2 == 1:
                    weight = net["layers.{}.weight".format(i + 1)]
                    bias = net["layers.{}.bias".format(i + 1)].reshape(-1, 1)
                    length = weight.shape[1]
                    M_L = torch.cat((torch.cat((weight, bias), 1),
                                     torch.cat((torch.zeros([length]), torch.tensor([1])), 0).reshape(1, -1)), 0)
                    M_U = M_L
                else:
                    if i == 2:
                        M_L = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=torch.float64)
                        M_U = torch.tensor([[0.5, 0, 1], [0, 0.5, 1], [0, 0, 1]], dtype=torch.float64)
                    if i == 4:
                        M_L = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=torch.float64)
                        M_U = torch.tensor([[5 / 6, 0, 5 / 12], [0, 0.5, 1], [0, 0, 1]], dtype=torch.float64)
                self.A_L.append(M_L)
                self.A_U.append(M_U)
                for j in range(i - 1, -1, -1):
                    if j == 0:
                        lowers = self.L[0]
                        uppers = self.U[0]
                        l = my_mv(M_L, lowers, uppers, False)
                        u = my_mv(M_U, lowers, uppers, True)
                        self.L.append(l)
                        self.U.append(u)
                        if i == self.num_layers - 1:
                            return l, u
                    else:
                        m_l = self.A_L[j - 1]
                        m_u = self.A_U[j - 1]
                        M_L = my_mm(M_L, m_l, m_u, False)
                        M_U = my_mm(M_U, m_l, m_u, True)

    def verify(self, margin):
        l, u = self.backsubstitution()
        return min(l) > 0

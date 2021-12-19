import argparse
import networks
from networks import FullyConnected
from torch import optim
from layers import *

DEVICE = 'cpu'
INPUT_SIZE = 28
num_iter = 10000000000
# num_iter = 1000


def build_verify_network(net, true_label, inputs):
    ordered_layers = [module for module in net.modules()
                      if type(module) not in [networks.FullyConnected, nn.Sequential]]
    verify_layers = []
    for layer in ordered_layers:
        in_features = verify_layers[-1].out_features if len(verify_layers) > 0 else inputs
        if type(layer) == networks.Normalization:
            pass
        elif type(layer) == nn.Flatten:
            pass
        elif type(layer) == nn.Linear:
            verify_layers.append(DPLinear(layer))
        else:
            verify_layers.append(DPSPU(in_features))
    verify_layers.append(Validator(verify_layers[-1].out_features, true_label=true_label))
    return nn.Sequential(*verify_layers)


def loss_fn(res):
    # print(res)
    return torch.sqrt(-res).sum()


def analyze(net, inputs, eps, true_label):
    pixel_values = inputs.view(-1)
    mean = 0.1307
    sigma = 0.3081
    lb = (pixel_values - eps).clamp(0, 1)
    ub = (pixel_values + eps).clamp(0, 1)
    lb = (lb - mean) / sigma
    ub = (ub - mean) / sigma

    verify_net = build_verify_network(net, true_label, len(pixel_values))

    opt = optim.Adam(verify_net.parameters(), lr=0.5)
    index = 0
    for i in range(num_iter):
        opt.zero_grad()
        # need to create new DeepPoly each iter
        x = X(lb, ub)
        verify_result = verify_net(x)
        if (verify_result > 0).all():
            return True
        if index >= 9:
            index = 0
        if verify_result[index] > 0:
            index += 1
            continue
        if i == num_iter - 1:
            return False
        loss = loss_fn(verify_result[index])
        loss.backward()
        # for name, para in verify_net.named_parameters():
        #     print(name, para.grad)
        if (verify_result != verify_result).sum() != 0:
            return False
        opt.step()


def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepPoly relaxation')
    parser.add_argument('--net',
                        type=str,
                        required=True,
                        help='Neural network architecture which is supposed to be verified.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])


    if args.net.endswith('fc1'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
    elif args.net.endswith('fc2'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
    elif args.net.endswith('fc3'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net.endswith('fc4'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
    elif args.net.endswith('fc5'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 10]).to(DEVICE)
    else:
        assert False

    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()

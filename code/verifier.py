import argparse
import torch
from networks import FullyConnected
from interface import *

DEVICE = 'cpu'
INPUT_SIZE = 28

def analyze(net, inputs, eps, true_label,heur):


    analyzer = Analyzer(net, inputs, eps, true_label,heur)
    # res = analyzer.test()
    res = analyzer.verify(0)
    return res

def create_input_batch(input,eps,batch_num=10000):
    input_lower=(input-eps).clamp(0,1)
    input_upper=(input+eps).clamp(0,1)
    N,C,H,W=input.shape
    random_num=torch.rand(batch_num,C,H,W)
    return input_lower*random_num+(1-random_num)*input_upper

def inference_separate(input_batch,net):
    '''
    NFL + SL + SL + SL
    Net: 3 + 2 + 2 + 2

    :param input_batch:
    :param net:
    :return:
    '''
    num_spu=(len(net.layers)-3)//2
    SPU_INPUT_HUER=[]
    fea=net.layers[:3](input_batch)
    SPU_INPUT_HUER.append(torch.cat([torch.unsqueeze(fea.max(dim=0)[0],1),torch.unsqueeze(fea.min(dim=0)[0],1)],1))
    base=3
    for i in range(num_spu-1):
        fea=net.layers[base:base+2*i](fea)
        SPU_INPUT_HUER.append(torch.cat([torch.unsqueeze(fea.max(dim=0)[0],1),torch.unsqueeze(fea.min(dim=0)[0],1)],1))
        base=base+2*i
    return SPU_INPUT_HUER
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

    input_batch=create_input_batch(inputs,eps,100000)
    # print('input_batch',input_batch.shape)
    with torch.no_grad():
        heur=inference_separate(input_batch,net)


    if analyze(net, inputs, eps, true_label,heur):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    # if analyze():
    #     print('verified')
    # else:
    #     print('not verified')
    main()
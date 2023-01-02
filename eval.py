import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F

from model import PreActResNet18
from utils import load_dataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--opt', default='SAM', type=str)
    parser.add_argument('--eps', default=1,  type=int)
    parser.add_argument('--step', default=5, type=int)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--adv', action='store_true')
    parser.add_argument('--all', action='store_true')
    return parser.parse_args()

def attack_pgd(model, x, y, eps, alpha, n_iters):
    delta = torch.zeros_like(x).to(x.device)
    delta.uniform_(-eps, eps)
    delta = torch.clamp(delta, 0-x, 1-x)
    delta.requires_grad = True
    for _ in range(n_iters):
        output = model(x+delta)
        loss = F.cross_entropy(output, y)
        loss.backward()
        grad = delta.grad.detach()
        d = torch.clamp(delta + alpha * torch.sign(grad), min=-eps, max=eps)
        d = torch.clamp(d, 0 - x, 1 - x)
        delta.data = d
        delta.grad.zero_()
    
    return delta.detach()

def adv_eval(filename):
    model = torch.load(filename)
    model.to(device)
    model.eval()
    
    _, test_loader = load_dataset(dataset)
    
    clean, robust, total = 0, 0, 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        clean += (output.max(1)[1] == y).float().sum().item()
        
        delta = attack_pgd(model, x, y, eps, alpha, args.step)
        output = model(x+delta)
        robust += (output.max(1)[1] == y).float().sum().item()
        
        total += len(y)
    
    print(f'{dataset} {opt}{"_adv" if args.adv else ""}:\t clean {clean/total*100:.2f} robust {robust/total*100:.2f}')

if __name__ == '__main__':
    args = get_args()
    dataset = args.dataset
    alpha = args.alpha / 255.
    eps = args.eps / 255.
    opt = args.opt
    device = f'cuda:{args.device}'
    
    if not args.all:
        filename = f'models/{dataset}_{opt}{"_adv" if args.adv else ""}.pth'
        adv_eval(filename)
    else:
        for dataset in ["cifar10", "cifar100"]:
            for adv in ["", "_adv"]:
                for opt in ["SAM", "SGD"]:
                    filename = f'models/{dataset}_{opt}{adv}.pth'
                    adv_eval(filename)
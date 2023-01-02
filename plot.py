import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', type=str)
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    dataset = args.dataset
    results = [] #ST, SAM, AT, AT+SAM
    heads = ["ST","SAM","AT","AT+SAM"]
    if dataset == 'cifar10':
        adv_res = [34.26,40.99,81.71,81.66]
    else:
        adv_res = [13.69,17.84,55.50,55.64]
    for adv in ["", "_adv"]:
        for opt in ['SGD', 'SAM']:
            csv_name = f'logs/{dataset}_{opt}{adv}.csv'
            data = pd.read_csv(csv_name).to_numpy() # epoch, train loss, acc, test loss, acc
            results.append(data)
    
    results = np.stack(results)
    
    # test loss
    # plot
    for i in range(4):
        plt.plot(np.arange(100),results[i,:,3])
    plt.title(f'{dataset} test loss', fontsize=20)
    plt.grid()
    plt.legend(["ST","SAM","AT","AT+SAM"], fontsize=16)
    plt.savefig(f'report_figs/{dataset}_test_loss.png', dpi=200)
    plt.clf()
    
    # test acc
    # plot
    for i in range(4):
        plt.plot(np.arange(100),results[i,:,4])
    plt.title(f'{dataset} test acc', fontsize=20)
    plt.grid()
    plt.legend(["ST","SAM","AT","AT+SAM"], fontsize=16)
    plt.savefig(f'report_figs/{dataset}_test_acc.png', dpi=200)
    plt.clf()
    
    # result
    for i in range(4):
        print(f'{heads[i]} & {results[i,-1,4]*100:.2f} & {adv_res[i]} \\\\')
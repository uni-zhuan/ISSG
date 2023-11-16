import numpy as np
import matplotlib.pyplot as plt
from utils import TargetDistribution, ProposalDistribution
import pandas as pd


def compute_importance_weights(samples, target, proposal):
    '''计算归一化的重要性权重'''
    weights = target.pdf(samples) / proposal.pdf(samples)
    weights /= np.sum(weights)  # 归一化
    return weights


def plot_distributions(samples, weights, target, proposal):
    '''绘制目标分布和建议分布'''
    x = np.linspace(np.min(samples), np.max(samples), 1000)
    plt.plot(x, target.pdf(x), label=f'Target: {target.name}')
    plt.plot(x, proposal.pdf(x), label=f'Proposal: {proposal.name}')
    plt.hist(samples, bins=30, density=True, alpha=0.5, color='g', 
             weights=weights/np.sum(weights), label='Weighted samples')
    plt.legend()
    plt.show()

def generate_dataset(target, proposal, n_samples):
    '''生成一个通过重要性采样得到的数据集'''
    # 生成样本
    samples = proposal.generate_samples(n_samples)
    # 计算权重
    weights = compute_importance_weights(samples, target, proposal)

    # 对于那些重要性权重特别高的样本，进行重复采样
    high_weight_indices = weights > np.percentile(weights, 75)  # 可以根据实际情况调整阈值
    samples = np.concatenate([samples, samples[high_weight_indices]])
    weights = np.concatenate([weights, weights[high_weight_indices]])

    # 丢弃那些重要性权重特别低的样本
    low_weight_indices = weights < np.percentile(weights, 25)  # 可以根据实际情况调整阈值
    samples = samples[~low_weight_indices]
    weights = weights[~low_weight_indices]

    # 返回数据集
    return samples, weights


def main():
    # 创建目标分布和建议分布, 这边需要完善怎么对应
    target = TargetDistribution('norm', 0, 1)  # 假设目标分布是标准正态分布
    proposal = ProposalDistribution('uniform', -3, 6)  # 假设建议分布是在[-3, 3]上的均匀分布

    # 生成数据集
    n_samples = 1000
    samples, weights = generate_dataset(target, proposal, n_samples)

    # 创建DataFrame
    df = pd.DataFrame({
        'Sample': samples,
        'Weight': weights
    })

    # 保存为CSV文件
    df.to_csv('dataset.csv', index=False)

    # 绘制目标分布和建议分布
    plot_distributions(samples, weights, target, proposal)


if __name__ == '__main__':
    main()

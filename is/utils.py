from scipy import stats
import numpy as np

class Distribution:
    def __init__(self, name, *params):
        self.name = name
        self.params = params
        self.dist = getattr(stats, name)

    def pdf(self, x):
        return self.dist.pdf(x, *self.params)

    def rvs(self, size):
        return self.dist.rvs(*self.params, size=size)


class TargetDistribution(Distribution):
    def calculate_expected_value(self, samples, weights):
        '''计算期望值'''
        return np.sum(samples * weights) / np.sum(weights)


class ProposalDistribution(Distribution):
    def generate_samples(self, n_samples):
        '''从建议分布中生成样本'''
        return self.rvs(n_samples)

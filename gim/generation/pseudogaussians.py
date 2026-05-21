'''
Adapted from https://github.com/XuandongZhao/PRC-Watermark, modified by PRC-estimator authors.
MIT License

Copyright (c) 2024 Xuandong Zhao, modified by PRC-estimator authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import torch
from scipy.special import erf
from scipy.linalg import orth
import numpy as np


def sample(codeword, basis=None):
    
    codeword_np = codeword.numpy()
    pseudogaussian_np = codeword_np * np.abs(np.random.randn(*codeword_np.shape))
    pseudogaussian = torch.from_numpy(pseudogaussian_np).to(dtype=torch.float64)
    if basis is None:
        return pseudogaussian
    return pseudogaussian @ basis.T


def recover_posteriors(z, basis=None, variances=None):
    if variances is None:
        default_variance = 1.5
        denominators = np.sqrt(2 * default_variance * (1+default_variance)) * torch.ones_like(z)
    elif type(variances) is float:
        denominators = np.sqrt(2 * variances * (1 + variances))
    else:
        denominators = torch.sqrt(2 * variances * (1 + variances))

    if basis is None:
        return erf(z / denominators)
    else:
        return erf((z @ basis) / denominators)

def random_basis(n):
    gaussian = torch.randn(n, n, dtype=torch.double)
    return orth(gaussian)

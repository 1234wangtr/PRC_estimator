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
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.special import binom, lambertw
from ldpc import bp_decoder
import sys
import galois

GF = galois.GF(2)

def apply_channel_probs(x, channel_probs):
    e = GF(np.random.binomial(1, channel_probs))
    return x + e


def boolean_row_reduce(A, print_progress=False):
    n, k = A.shape
    A_rr = A.copy()
    perm = np.arange(n)
    for j in range(k):
        idxs = j + np.nonzero(A_rr[j:, j])[0]
        if idxs.size == 0:
            print("The given matrix is not invertible")
            return None
        A_rr[[j, idxs[0]]] = A_rr[[idxs[0], j]]  
        (perm[j], perm[idxs[0]]) = (perm[idxs[0]], perm[j])  
        A_rr[idxs[1:]] += A_rr[j]
        if print_progress and (j%5==0 or j+1==k):
            sys.stdout.write(f'\rDecoding progress: {j + 1} / {k}')
            sys.stdout.flush()
    if print_progress: print()
    return perm[:k]


def str_to_bin(string):
    bin_str = ''.join(format(i, '08b') for i in bytearray(string, encoding ='utf-8'))
    return [int(b) for b in bin_str]

def bin_to_str(bin_list):
    bin_str = ''.join(map(str, bin_list))
    byte_array = bytearray(int(bin_str[i:i+8], 2) for i in range(0, len(bin_str), 8) if bin_str[i:i+8]!='00000000')
    return byte_array.decode('utf-8')










def KeyGen(n, message_length=512, false_positive_rate=1e-9, t=3, g=None, r=None, noise_rate=None):
    
    num_test_bits = int(np.ceil(np.log2(1 / false_positive_rate)))
    secpar = int(np.log2(binom(n, t)))
    if g is None: g = secpar
    
    
    if noise_rate is None: noise_rate = 1 - 2 ** (-secpar / g ** 2)
    
    
    
    k = message_length + g + num_test_bits
    if r is None: r = n - k - secpar
    

    
    generator_matrix = GF.Random((n, k))

    
    row_indices = []
    col_indices = []
    data = []
    for row in range(r):
        chosen_indices = np.random.choice(n - r + row, t - 1, replace=False)
        chosen_indices = np.append(chosen_indices, n - r + row)
        row_indices.extend([row] * t)
        col_indices.extend(chosen_indices)
        data.extend([1] * t)
        generator_matrix[n - r + row] = generator_matrix[chosen_indices[:-1]].sum(axis=0)
    parity_check_matrix = csr_matrix((data, (row_indices, col_indices)))

    
    max_bp_iter = int(np.log(n) / np.log(t))

    
    one_time_pad = GF.Random(n)
    test_bits = GF.Random(num_test_bits)

    
    permutation = np.random.permutation(n)
    generator_matrix = generator_matrix[permutation]
    one_time_pad = one_time_pad[permutation]
    parity_check_matrix = parity_check_matrix[:, permutation]

    encoding_key = (generator_matrix, one_time_pad, test_bits, g, noise_rate)
    decoding_key = (generator_matrix, parity_check_matrix, one_time_pad, false_positive_rate, noise_rate, test_bits, g, max_bp_iter, t)
    
    
    return encoding_key, decoding_key






def Encode(encoding_key, message=None):
    generator_matrix, one_time_pad, test_bits, g, noise_rate = encoding_key
    
    n, k = generator_matrix.shape

    if message is None:
        payload = np.concatenate((test_bits, GF.Random(k - len(test_bits))))
    else:
        assert len(message) <= k-len(test_bits)-g, "Message is too long"
        payload = np.concatenate((test_bits, GF.Random(g), GF(message), GF.Zeros(k-len(test_bits)-g-len(message))))

    error = GF(np.random.binomial(1, noise_rate, n))
    
    
    x=payload @ generator_matrix.T

    return 1 - 2 * torch.tensor(x + one_time_pad + error, dtype=float), x, one_time_pad, error








def Detect(decoding_key, posteriors, false_positive_rate=None):
    generator_matrix, parity_check_matrix, one_time_pad, false_positive_rate_key, noise_rate, test_bits, g, max_bp_iter, t = decoding_key
    if false_positive_rate is not None:
        fpr = false_positive_rate
    else:
        fpr = false_positive_rate_key

    posteriors = (1 - 2 * noise_rate) * (1 - 2 * np.array(one_time_pad, dtype=float)) * posteriors.numpy(force=True)
    posteriors_array = []
    
    for i in range(len(posteriors)):
        if posteriors[i]>0:
            posteriors_array.append(0)
        else:
            posteriors_array.append(1)
    r = parity_check_matrix.shape[0]
    Pi = np.prod(posteriors[parity_check_matrix.indices.reshape(r, t)], axis=1)
    log_plus = np.log((1 + Pi) / 2)
    log_minus = np.log((1 - Pi) / 2)
    log_prod = log_plus + log_minus

    const = 0.5 * np.sum(np.power(log_plus, 2) + np.power(log_minus, 2) - 0.5 * np.power(log_prod, 2))
    threshold = np.sqrt(2 * const * np.log(1 / fpr)) + 0.5 * log_prod.sum()
    
    return posteriors_array, log_plus.sum() >= threshold








def Decode(decoding_key, posteriors, print_progress=False, max_bp_iter=None):
    generator_matrix, parity_check_matrix, one_time_pad, false_positive_rate_key, noise_rate, test_bits, g, max_bp_iter_key, t = decoding_key
    if max_bp_iter is None:
        max_bp_iter = max_bp_iter_key

    posteriors = (1 - 2 * noise_rate) * (1 - 2 * np.array(one_time_pad, dtype=float)) * posteriors.numpy(force=True)
    channel_probs = (1 - np.abs(posteriors)) / 2
    x_recovered = (1 - np.sign(posteriors)) // 2

    
    if print_progress:
        print("Running belief propagation...")
    bpd = bp_decoder(parity_check_matrix, channel_probs=channel_probs, max_iter=max_bp_iter, bp_method="product_sum")
    x_decoded = bpd.decode(x_recovered)

    
    bpd_probs = 1 / (1 + np.exp(bpd.log_prob_ratios))
    confidences = 2 * np.abs(0.5 - bpd_probs)

    
    confidence_order = np.argsort(-confidences)
    ordered_generator_matrix = generator_matrix[confidence_order]
    ordered_x_decoded = x_decoded[confidence_order]

    
    
    

    
    top_invertible_rows = boolean_row_reduce(ordered_generator_matrix, print_progress=print_progress)
    if top_invertible_rows is None:
        return None

    
    if print_progress:
        print("Solving linear system...")
    a = ordered_x_decoded[top_invertible_rows]
    
    recovered_string = np.linalg.solve(ordered_generator_matrix[top_invertible_rows], GF(a.astype(int)))

    if not (recovered_string[:len(test_bits)] == test_bits).all():
        return None
    return np.array(recovered_string[len(test_bits) + g:])

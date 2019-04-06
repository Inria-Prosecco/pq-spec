from __future__ import print_function

from math import log
from pqsec import optimize_attack, svp_classical, svp_quantum, svp_plausible, primal_cost, dual_cost
from failure_prob_pke import exact_failure_prob_pke
from discrete_distr import dgauss, std_modulo, pdf_product
from approx_distr import approximate_dgauss


def estimate_cost(q, n, max_m, s, attacks):
  """Finds attack costs (classical, quantum, plausible quantum) for an LWE instance.

  Args:
    q: LWE modulus.
    n: LWE dimension.
    max_m: maximum number of samples
    s: standard deviation of the error distribution
    attacks: A vector specifying types of attacks to consider (svp_classical,
        svp_quantum, svp_plausible)

  Returns:
    A triple of log_2 of the costs of three attacks: classical, quantum, and
    plausible quantum.
    Infinity if analysis is not run.
  """
  r = [float('inf')] * 3
  for cost in [primal_cost, dual_cost]:
    for i, attack in enumerate(attacks):
      _, _, c = optimize_attack(q, n, max_m, s, cost, attack)
      r[i] = min(r[i], c)
  return r


def is_secure(qlog, n, sigma, m_bar, n_bar):
  max_m = n + max(m_bar, n_bar)
  costs_gauss_single_shot = estimate_cost(2 ** qlog, n, max_m, sigma,
                                          attacks=[svp_classical, svp_quantum, svp_plausible])

  # Taking into account that there are m_bar + n_bar simultaneous instances that all must hold.
  costs_gauss = [c - log(m_bar + n_bar) / log(2) for c in costs_gauss_single_shot]

  return costs_gauss[0] >= 272 and costs_gauss[1] >= 149 and costs_gauss[2] >= 149


def approximate_and_compute_failure_pr(qlog, n, sigma, m_bar, n_bar, agree_bits, w):
  samples = 2 * (m_bar + n_bar) * n + m_bar * n_bar
  _, d, a = approximate_dgauss(sigma, samples, 149, None, 16, quiet=True)
  sym_distr = pdf_product(d, {+1: .5, -1: .5})
  return exact_failure_prob_pke(sym_distr, 2 ** qlog, n, w, agree_bits)


def main():
  sigmas = [1.3, 1.35, 1.4, 1.45, 1.5]
  ns = range(1312, 1600, 16)

  agree_bits = 256
  qlog = 16
  w = 4
  m_bar, n_bar = 8, 8

  print('n & bandwidth &' + ' & '.join(['{:.2f}'.format(sigma) for sigma in sigmas]) + r' \\')

  # true_sigmas = [std_modulo(dgauss(sigma), 2**qlog) for sigma in sigmas]
  # print('n & bandwidth &' + ' & '.join(['{:.2f}'.format(sigma) for sigma in true_sigmas]) + r' \\')

  for n in ns:
    print(n, '&', (m_bar * n + m_bar * n_bar) * qlog / 8, end='')
    for sigma in sigmas:
      if is_secure(qlog, n, sigma, m_bar, n_bar):
        failure_pr = approximate_and_compute_failure_pr(qlog, n, sigma, m_bar, n_bar, agree_bits, w)
        if failure_pr == 0:
          s = r' & $0$ '
        else:
          s = r' & $2^{{-{}}}$'.format(int(log(1 / failure_pr) / log(2)))
        print(s, end='')
      else:
        print(' & ---', end='')
    print(r'\\')


if __name__ == '__main__':
  main()

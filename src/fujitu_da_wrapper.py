from __future__ import print_function

import copy
import json
import requests
import numpy as np


class QPoly(object):
    """Quadratic Polynomial class
       Arguments:
           n:                 The number of variables that can be handled by this QPoly.
       Attributes:
           array:             The numpy array showing this QPoly.
           constant:          The constant value of this QPoly.
    """

    def __init__(self, n=1024):
        self.array = np.zeros((n, n), dtype=int)
        self.constant = 0
        self._size = n

    def add_term(self, c, i, j=None):
        """Add a term 'c * x_i * x_j' to this QPoly"""
        if j is None:
            j = i
        if i >= self._size or j >= self._size:
            raise RuntimeError('wrong var number')
        if i > j:
            self.array[j][i] += c
        else:
            self.array[i][j] += c

    def add_constant_term(self, c):
        """Add a constant term 'c' to this QPoly"""
        self.constant += c

    def power(self, p=2):
        """Raise this QPoly to the second power"""
        diag = np.diag(self.array)
        if np.count_nonzero(self.array - np.diag(diag)) > 0 or p != 2:
            raise RuntimeError('not quadratic')
        a = np.outer(diag, diag)
        self.array = np.triu(a, k=1) + np.triu(a.T) + \
            (2 * self.constant * np.diag(diag))
        self.constant = self.constant ** 2

    def multiply_quadratic_binary_polynomial(self, poly):
        """Multiply this QPoly with a Quadratic Polynomial 'poly'"""
        diag0 = np.diag(self.array)
        diag1 = np.diag(poly.array)
        if diag0.size != diag1.size:
            raise RuntimeError('wrong array size')
        if np.count_nonzero(self.array - np.diag(diag0)) > 0 or np.count_nonzero(poly.array - np.diag(diag1)) > 0:
            raise RuntimeError('not quadratic')
        a = np.outer(diag0, diag1)
        self.array = np.triu(a, k=1) + np.triu(a.T) + (self.constant *
                                                       np.diag(diag1)) + (poly.constant * np.diag(diag0))
        self.constant *= poly.constant

    def multiply_by_factor(self, f):
        """Multiply all terms in this QPoly by a constant value 'f'"""
        self.array *= f
        self.constant *= f

    def sum(self, p):
        """Add a Quadratic Polynomial 'p' to this QPoly"""
        if self._size != p._size:
            raise RuntimeError('wrong array size')
        self.array += p.array
        self.constant += p.constant

    def build_polynomial(self):
        """Make a copy of itself"""
        return copy.deepcopy(self)

    def export_dict(self):
        """Convert this QPoly to a dictionary"""
        cells = np.where(self.array != 0)
        ts = [{"coefficient": float(self.array[i][j]), "polynomials": [
            int(i), int(j)]} for i, j in zip(cells[0], cells[1])]
        if self.constant != 0:
            ts.append({"coefficient": float(self.constant), "polynomials": []})
        return {'binary_polynomial': {'terms': ts}}

    def reset(self):
        """Clear this QPoly"""
        self.array.fill(0)
        self.constant = 0

    def eval(self, conf):
        """Evaluate this Poly with a configuration 'conf'"""
        if self._size < len(conf):
            raise RuntimeError('wrong configuration size')
        val = self.constant
        for i, c in enumerate(conf):
            for j, d in enumerate(conf[i:]):
                if c and d:
                    val += self.array[i][j + i]
        return val

    def remove_var(self, var):
        if var < 0 or self._size <= var:
            raise RuntimeError('wrong var number')
        self.array[:, var] = 0
        self.array[var, :] = 0


class SolverResponse(object):
    """Solver Response class
       Attributes:
           response:          The raw data which is a response of requests.
           answer_mode:       The distribution of solutions. When 'HISTOGRAM' is set, get_solution_list() returns a histogram of solutions.
    """
    class AttributeSolution(object):
        def __init__(self, obj):
            self.obj = obj

        def __getattr__(self, key):
            if key in self.obj:
                return self.obj.get(key)
            else:
                raise AttributeError(key)

        def keys(self):
            return self.obj.keys()

    def __init__(self, response):
        solutions = response.json()[u'qubo_solution'][u'solutions']
        self.answer_mode = 'RAW'
        self.response = response
        self._solutions = [self.AttributeSolution(d) for d in solutions]
        self._solution_histogram = []
        lowest_energy = None
        for sol in solutions:
            if lowest_energy is None or lowest_energy > sol.get(u'energy'):
                lowest_energy = sol.get(u'energy')
                self.minimum_solution = self.AttributeSolution(sol)
        for i, d in enumerate(solutions):
            if i == solutions.index(d):
                self._solution_histogram.append(copy.deepcopy(d))
            else:
                for s in self._solution_histogram:
                    if s[u'configuration'] == d[u'configuration']:
                        s[u'frequency'] += 1
                        break
        self._solution_histogram = sorted([self.AttributeSolution(
            d) for d in self._solution_histogram], key=lambda x: x.energy)

    def get_minimum_energy_solution(self):
        """Get a minimum energy solution"""
        return self.minimum_solution

    def get_solution_list(self):
        """Get all solution"""
        if self.answer_mode == 'HISTOGRAM':
            return self._solution_histogram
        else:
            return self._solutions


class Solver_fujitsu(object):
    """
        Digital Annealer Solver class
        @Shirai modified 2019 09 14
        Attributes:
            rest_url: Digital Annealer Web API address and port 'http://<address>:<port>'.
            timing: To show the time (milliseconds) for the minimization.
            anneal_time: To show the time spent by the DA hardware to solve the problem in ms.
    """

    def __init__(self):
        self.rest_url = 'https://api.jp-east-1.digitalannealer.global.fujitsu.com'
        self.access_key = None
        self.proxies = None
        self.rest_headers = {'content-type': 'application/json'}
        self.cpu_time = 0
        self.queue_time = 0
        self.solve_time = 0
        self.total_elapsed_time = 0
        self.anneal_time = 0

    def minimize(self, poly_dic):
        """Find the minimum value of a Quadratic Polynomial 'poly' and return a object of SolverResponse class"""
        request = poly_dic
        headers = self.rest_headers
        headers['X-DA-Access-Key'] = self.access_key
        response = requests.post(self.rest_url + '/v1/qubo/solve',
                                 json.dumps(request), headers=headers, proxies=self.proxies)
        if response.ok:
            j = response.json()
            if j[u'qubo_solution'].get(u'timing'):
                self.cpu_time = j[u'qubo_solution'][u'timing'][u'cpu_time']
                self.queue_time = j[u'qubo_solution'][u'timing'][u'queue_time']
                self.solve_time = j[u'qubo_solution'][u'timing'][u'solve_time']
                self.total_elapsed_time = j[u'qubo_solution'][u'timing'][u'total_elapsed_time']
                self.anneal_time = j[u'qubo_solution'][u'timing'][u'detailed'][u'anneal_time']
            if j[u'qubo_solution'][u'result_status']:
                return SolverResponse(response)
            raise RuntimeError('result_status is false.')
        else:
            raise RuntimeError(response.text)


if __name__ == "__main__":
    import doctest
    import sys
    sys.exit(doctest.testmod(verbose=True, optionflags=doctest.ELLIPSIS)[0])

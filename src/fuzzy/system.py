"""
Fuzzy system
"""
import logging

import numpy as np
import numpy.typing as npt

from .rule import Rule

logger = logging.getLogger(__name__)

class FuzzySystem:
    """ Equidistant sets """
    def __init__(self, intervals: list, discretization: list, output_interval: tuple, output_discretization: int) -> None:

        self._intervals = intervals
        self._discretization = discretization

        assert len(intervals) >= 1 and len(discretization) >= 1, "Length of a intervals and discretization has to be GEC 1"
        assert len(intervals) == len(discretization), "Intervals and discretization has to be of a same shape"
        assert all([d > 1 for d in discretization]), "All discretizations has to be GEC 2"

        self._rules = []
        self._rules_metadata = {}
        
        # cached properties
        self._sets_centers = [np.linspace(i[0], i[1], d) for i, d in zip(intervals, discretization)] # this is cached property
        self._diffs = [np.mean(np.diff(centers)) for centers, d in zip(self.centers, discretization)]

        self._output_set_centers = np.linspace(output_interval[0], output_interval[1], output_discretization)
        self._output_diffs = np.mean(np.diff(self._output_set_centers))

        if True: # type fuzzy sets linear
            self._sigmas_left = [np.full(shape=(d,), fill_value=3*c) for c, d in zip(self.diffs, discretization)]
            self._sigmas_right = [np.full(shape=(d,), fill_value=3*c) for c, d in zip(self.diffs, discretization)]

            self._output_sigmas_left = np.full(shape=(output_discretization,), fill_value=self._output_diffs)
            self._output_sigmas_right = np.full(shape=(output_discretization,), fill_value=self._output_diffs)

    @property
    def centers(self):
        return self._sets_centers
    
    @property
    def sigmas_left(self):
        return self._sigmas_left

    @property
    def sigmas_right(self):
        return self._sigmas_right

    @property
    def diffs(self):
        return self._diffs

    @property
    def ndim(self):
        return len(self._intervals)
    
    @property
    def rules(self):
        return self._rules
    
    def add_rules(self, rules):
        self._rules = rules
    
    def fuzzifier(self, x: npt.NDArray):
        values = []
        # logger.debug(f"\n\n\nWORKING ON X={x}")
        for i in range(self.ndim):
            # logger.debug(f" ..... \n \n centers {self.centers[i]} \n left {self.sigmas_left[i]} \n right {self.sigmas_right[i]}\n ......")
            is_left = (x[i] < self.centers[i]) & ( x[i] >= self.centers[i] - self.sigmas_left[i])
            is_right = (x[i] >= self.centers[i]) & (x[i] < self.centers[i] + self.sigmas_right[i])
            left = 1 - (1.0 / self.sigmas_left[i]) * (self.centers[i] - x[i])
            right = 1 - (1.0 / self.sigmas_right[i]) * (x[i] - self.centers[i])
            # logger.debug(f" ..... \n left {is_left.astype(int)} -> {left} \n right {is_right.astype(int)} -> {right} \n ......")
            values.append(is_left * left + is_right * right)
        return values
    
    def output_fuzzifier(self, y: float):
        is_left = (y < self._output_set_centers) & (y >= self._output_set_centers - self._output_sigmas_left)
        is_right = (y >= self._output_set_centers) & (y < self._output_set_centers + self._output_sigmas_right)
        left = 1 - (1.0 / self._output_sigmas_left) * (self._output_set_centers - y)
        right = 1 - (1.0 / self._output_sigmas_right) * (y - self._output_set_centers)
        return (is_left * left + is_right * right)
    
    def rule_fullfillment(self, rule_lhs: tuple, fuzzifier_output: list) -> float:
        # assume ndim TODO
        prod = 1.0
        for i, fset_index in enumerate(rule_lhs):
            prod *= fuzzifier_output[i][fset_index]
        return prod

    def inference(self, fuzzifier_output: list):
        # product inference engine
        inferenced = []
        for rule in self.rules:
            # implication ((1,2,5), 3)
            lhs = rule.index_x
            rhs = rule.index_y

            rule_fullfilment = self.rule_fullfillment(lhs, fuzzifier_output)
            inferenced.append((rule_fullfilment, rhs))
        return inferenced
    
    def trust_by_rule(self, fuzzifier_output: list):
        trust = 0.0
        for rule in self.rules:
            # implication ((1,2,5), 3)
            lhs = rule.index_x
            rhs = rule.index_y
            rule_fullfilment = self.rule_fullfillment(lhs, fuzzifier_output)
            
            trust += rule.counter[rhs] * rule_fullfilment
        return trust
    
    def defuzzifier(self, inference_output):
        # list of (rule_fullfillment: float, output fuzzy set index: int)

        num = 0.0
        denum = 0.0
        for i, iout in enumerate(inference_output):
            r, fset_index = iout
            num += r * self._output_set_centers[fset_index]
            denum += r
        
        return num / denum

    def value_to_rule(self, x: npt.NDArray, y: float):
        fuzzified = self.fuzzifier(x)
        lhs = tuple([int(np.argmax(v)) for v in fuzzified])
        rhs = np.argmin(np.abs(self._output_set_centers - y))
        # logger.debug(f"   {y} -- > {rhs} ({self._output_set_centers}, {np.abs(self._output_set_centers - y)})")
        # rule = Rule(lhs, rhs, None, x, y)
        rule = Rule(lhs)
        rule.increase_counter(rhs)
        return rule
    
    def d_rule(self, rule: Rule):
        prod = 1.0
        # logger.debug(f"Calculating Drule for rule {rule}")
        for ix, memberships in zip(rule.index_x, self.fuzzifier(rule.origin_x)):
            # logger.debug(f"    {ix} -> {memberships}")
            prod *= memberships[ix]
        
        
        output = self.output_fuzzifier(rule.origin_y)
        # logger.debug(f"    (OUTPUT) {rule.index_y} -> {output}")
        prod *= output[rule.index_y]
        return prod
    
    def add_rule_io_pair(self, x, y):
        new_rule = self.value_to_rule(x, y)        
        # check if rule exists
        exists = False
        for i, r in enumerate(self.rules):
            if r.index_x == new_rule.index_x:
                self.rules[i] += new_rule
                exists = True
                break
        if not exists: # New rule HAS to be created
            logger.debug(f"Adding new rule {new_rule}")
            self._rules.append(new_rule)            
            
    def values_to_rules(self):
        pass

    def predict(self, x):
        return self.defuzzifier(self.inference(self.fuzzifier(x)))
    
    def trust(self, x) -> float:
        fuzzifier_output = self.fuzzifier(x)
        trust = 0.0
        for rule in self.rules:
            # implication ((1,2,5), 3)
            lhs = rule.index_x
            rhs = rule.index_y
            rule_fullfilment = self.rule_fullfillment(lhs, fuzzifier_output)
            
            trust += rule.counter[rhs] * rule_fullfilment
        return trust
    
    def confusion(self, x) -> float:
        fuzzifier_output = self.fuzzifier(x)
        confusion = 0.0
        for rule in self.rules:
            # implication ((1,2,5), 3)
            lhs = rule.index_x
            rhs = rule.index_y
            
            rule_fullfilment = self.rule_fullfillment(lhs, fuzzifier_output)
            
            c = sum([v for k, v in rule.counter.items() if k != rhs])
            confusion += c * rule_fullfilment
        return confusion


        






    

    

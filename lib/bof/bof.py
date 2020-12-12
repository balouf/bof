def default_preprocessor(txt):
    return txt.strip().lower()


class FactorTree:
    def __init__(self, txt_list=None, auto_update=False, preprocessor=None, n_range=5):
        """
        Maintain a Tree of factor of a given corpus.

        Parameters
        ----------
        txt_list
        auto_update
        preprocessor
        n_range

        Attributes
        ----------
        """
        self.nodes = [dict()]
        self.edges = [dict()]
        self.txt_list = []
        self.txt_dict = dict()
        self.factor_list = [""]
        self.factor_dict = {"": 0}
        self.factors = []
        self.n = 1
        self.m = 0
        self.auto_update = auto_update
        if preprocessor is None:
            preprocessor = default_preprocessor
        self.preprocessor = preprocessor
        self.n_range = n_range
        if txt_list is not None:
            self.add_txt_list_to_tree(txt_list)

    def add_txt_list_to_tree(self, txt_list):
        for txt in txt_list:
            self.add_txt_to_tree(txt)

    def add_txt_to_tree(self, txt):
        txt = self.preprocessor(txt)
        length = len(txt)

        # Empty factor my friend!
        self.nodes[0][self.m] = len(txt) + 1
        self.factors.append(1)

        for start in range(length):
            node = 0
            end = min(start + self.n_range, length) if self.n_range else length
            for letter in txt[start:end]:
                n_node = self.edges[node].setdefault(letter, self.n)
                if n_node == self.n:
                    self.edges.append(dict())
                    self.nodes.append(dict())
                    fac = self.factor_list[node] + letter
                    self.factor_list.append(fac)
                    self.factor_dict[fac] = self.n
                    self.n += 1
                node = n_node
                d = self.nodes[node]
                if d.setdefault(self.m, 0) == 0:
                    self.factors[self.m] += 1
                d[self.m] += 1
        self.txt_list.append(txt)
        self.txt_dict[txt] = self.m
        self.m += 1

    def common_factors(self, txt):
        txt = self.preprocessor(txt)
        index = self.txt_dict.get(txt)
        if not index:
            if not self.auto_update:
                return self.common_factors_external(txt)
            index = self.m
            self.add_txt_to_tree(txt)
        return self.common_factors_internal(index)

    def common_factors_internal(self, i):
        buffer = {0}
        res = [0] * self.m
        while buffer:
            node = buffer.pop()
            if i in self.nodes[node]:
                for j in self.nodes[node]:
                    res[j] += 1
                buffer.update(self.edges[node].values())
        return res, self.factors[i]

    def common_factors_external(self, txt):
        buffer = {0}
        res = [0] * self.m
        new_tree = FactorTree([txt], preprocessor=self.preprocessor, n_range=self.n_range)
        while buffer:
            node = buffer.pop()
            target = self.factor_dict.get(new_tree.factor_list[node])
            if target is not None:
                for j in self.nodes[target]:
                    res[j] += 1
                buffer.update(new_tree.edges[node].values())
        return res, new_tree.factors[0]

    def joint_complexity(self, txt, bias=.5):
        common_factors, autofactors = self.common_factors(txt)
        biased_factors = [2 * bias * autofactors + 2 * (1 - bias) * f for f in self.factors]
        return [0 if f - common_factor_number == 0 else
                (f - 2 * common_factor_number) /
                (f - common_factor_number)
                for f, common_factor_number in zip(biased_factors, common_factors)]

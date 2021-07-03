# -*- coding: utf-8 -*-
from package.utils import decode_bio_tags
from collections import Counter


class Score:
    """ Score BIO-style NER
    """

    def __init__(self):
        super(Score, self).__init__()
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self):
        """ Compute the score
        """

        info = {}
        origins = Counter([t for t, _, _ in self.origins])
        founds = Counter([t for t, _, _ in self.founds])
        rights = Counter([t for t, _, _ in self.rights])

        origin_all = 0
        found_all = 0
        right_all = 0
        for t, count in origins.items():
            found = founds.get(t, 0)
            right = rights.get(t, 0)
            info[t] = self._score(count, found, right)
            origin_all += count
            found_all += found
            right_all += right

        return self._score(origin_all, found_all, right_all), info

    @staticmethod
    def _score(origin, found, right):
        """ Score Utils
        """
        p = 0 if found == 0 else (right / found)
        r = 0 if origin == 0 else (right / origin)
        f1 = 0. if r + p == 0 else (2 * p * r) / (p + r)
        return {'p': p, 'r': r, 'f1': f1}

    def update(self, input, target):
        """ Update the result
        """
        for y_pred, y_target in zip(input, target):
            y_pred = decode_bio_tags(y_pred)
            y_target = decode_bio_tags(y_target)

            self.origins.extend(y_target)
            self.founds.extend(y_pred)
            self.rights.extend([y for y in y_pred if y in y_target])

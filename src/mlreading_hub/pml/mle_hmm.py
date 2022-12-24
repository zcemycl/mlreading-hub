from collections import defaultdict

import numpy as np


class HMM:
    def __init__(self):
        self.params = defaultdict(dict)
        self.o2id, self.id2o = dict(), dict()
        self.q2id, self.id2q = dict(), dict()
        self.ct, self.ce = dict(), dict()

    def _setDistinctObservations(self, distO):
        self.o = len(distO)
        for i, o in enumerate(distO):
            self.o2id[o] = i
            self.id2o[i] = o

    def _setDistinctHiddens(self, distQ):
        self.q = len(distQ)
        A = np.random.uniform(0, 1, (self.q, self.q))
        A /= np.sum(A, axis=1, keepdims=1)
        B = np.random.uniform(0, 1, (self.q, self.o))
        B /= np.sum(B, axis=1, keepdims=1)
        pi = np.random.uniform(0, 1, self.q)
        pi /= np.sum(pi)
        self.A = defaultdict(dict)
        self.B = defaultdict(dict)
        self.pi = {}
        for i, q in enumerate(distQ):
            self.q2id[q] = i
            self.id2q[i] = q
            for j, q_ in enumerate(distQ):
                self.A[i][j] = A[i][j]
            for k in range(self.o):
                self.B[i][k] = B[i][k]
            self.pi[i] = pi[i]

    def setDistinctHiddensAndObservations(self, distO, distQ):
        self._setDistinctObservations(distO)
        self._setDistinctHiddens(distQ)

    def setSpecificEmit(self, qSym, emitDict):
        assert sum(emitDict.values()) == 1, "Sum of probability is not 1"
        for i in self.B[self.q2id[qSym]].keys():
            # assert in dict
            self.B[self.q2id[qSym]][i] = emitDict.get(self.id2o[i], 0)
        assert (
            sum(self.B[self.q2id[qSym]].values()) == 1
        ), "Sum of probability is not 1"

    def setSpecificTransit(self, qSym, tranDict):
        assert sum(tranDict.values()) == 1, "Sum of probability is not 1"
        for i in self.A[self.q2id[qSym]].keys():
            self.A[self.q2id[qSym]][i] = tranDict.get(self.id2q[i], 0)
        assert (
            sum(self.A[self.q2id[qSym]].values()) == 1
        ), "Sum of probability is not 1"

    def setInitial(self, initDict):
        assert sum(initDict.values()) == 1, "Sum of probability is not 1"
        for i in self.pi.keys():
            # assert in dict
            self.pi[i] = initDict.get(self.id2q[i], 0)
        assert sum(self.pi.values()) == 1, "Sum of probability is not 1"

    def computeLikelihood(self, Os):
        raise NotImplementedError(
            "You need to implement function1 when you inherit from Model"
        )

    def decode(self, Os):
        raise NotImplementedError(
            "You need to implement function1 when you inherit from Model"
        )

    def learn(self, Qs, Os):
        raise NotImplementedError(
            "You need to implement function1 when you inherit from Model"
        )


def likelihood(self, Os):
    Length = len(Os)
    alpha = np.zeros((self.q, Length))
    for t, o in enumerate(Os):
        for i in range(self.q):
            if t == 0:
                alpha[i, t] = self.pi[i] * self.B[i][self.o2id[o]]
            else:
                for j in range(self.q):
                    alpha[i, t] += alpha[j, t - 1] * self.A[j][i]
                alpha[i, t] *= self.B[i][self.o2id[o]]
    return sum(alpha[:, -1]), alpha


def decode(self, Os):
    Length = len(Os)
    V = np.zeros((self.q, Length))
    bt = [0] * Length
    for t, o in enumerate(Os):
        for i in range(self.q):
            if t == 0:
                V[i, t] = self.pi[i] * self.B[i][self.o2id[o]]
            else:
                for j in range(self.q):
                    V[i, t] = max(V[i, t], V[j, t - 1] * self.A[j][i])
                V[i, t] *= self.B[i][self.o2id[o]]
        bt[t] = self.id2q[np.argmax(V[:, t])]
    P_ = max(V[:, -1])
    return P_, bt


def samples(self, length):
    Qs, Os = [], []
    for i in range(length):
        if i == 0:
            q = np.random.choice(self.q, 1, p=list(self.pi.values()))
        else:
            q = np.random.choice(self.q, 1, p=list(self.A[q[0]].values()))
        o = np.random.choice(self.o, 1, p=list(self.B[q[0]].values()))
        Qs.append(self.id2q[q[0]])
        Os.append(self.id2o[o[0]])
    return Qs, Os


def learn(self, Qs, Os):
    Length = len(Qs)
    if Length == 0:
        return
    for i, (q, o) in enumerate(zip(Qs, Os)):
        q_ = self.q2id[q]
        o_ = self.o2id[o]
        if i == 0:
            self.ct[None] = self.ct.get(None, 0) + 1
            self.ct[(None, q_)] = self.ct.get((None, q_), 0) + 1
        if i != Length - 1:
            q_1 = self.q2id[Qs[i + 1]]
            self.ct[q_] = self.ct.get(q_, 0) + 1
            self.ct[(q_, q_1)] = self.ct.get((q_, q_1), 0) + 1
        self.ce[q_] = self.ce.get(q_, 0) + 1
        self.ce[(q_, o_)] = self.ce.get((q_, o_), 0) + 1
    for i in range(self.q):
        self.pi[i] = self.ct.get((None, i), 0) / self.ct.get(None, 0)
        for j in range(self.q):
            self.A[i][j] = (self.ct.get((i, j), 0) + 1) / (
                self.ct.get(i, 0) + self.q
            )
        for o in range(self.o):
            self.B[i][o] = (self.ce.get((i, o), 0) + 1) / (
                self.ce.get(i, 0) + self.o
            )


def add_patch(obj):
    obj.computeLikelihood = lambda x: likelihood(obj, x)
    obj.decode = lambda x: decode(obj, x)
    obj.samples = lambda x: samples(obj, x)
    obj.learn = lambda x, y: learn(obj, x, y)
    return obj

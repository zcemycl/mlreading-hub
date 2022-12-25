from collections import defaultdict

import numpy as np
from tqdm import tqdm


class HMM:
    def __init__(self):
        self.params = defaultdict(dict)
        self.o2id, self.id2o = dict(), dict()
        self.q2id, self.id2q = dict(), dict()

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


def likelihood_backwards(self, Os):
    Length = len(Os)
    beta = np.zeros((self.q, Length))
    for t in range(Length - 1, -1, -1):
        # o = Os[t]
        for i in range(self.q):
            if t == Length - 1:
                beta[i, t] = 1
            else:
                for j in range(self.q):
                    o_tp1 = self.o2id[Os[t + 1]]
                    tmp = beta[j, t + 1] * self.A[i][j]
                    tmp *= self.B[j][o_tp1]
                    beta[i, t] += tmp

    P = 0
    for j in range(self.q):
        P += self.pi[j] * self.B[j][self.o2id[Os[0]]] * beta[j, 0]
    return P, beta


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


def e_step(self, Os):
    Length = len(Os)
    l, alpha = self.computeLikelihood(Os)
    lB, beta = self.likelihood_backwards(Os)
    alphabeta = alpha * beta
    sum_alphabeta = np.sum(alphabeta, axis=0)
    gamma = alphabeta / sum_alphabeta
    xi = np.zeros((Length, self.q, self.q))
    for t in range(Length - 1):
        o_tp1 = self.o2id[Os[t + 1]]
        for i in range(self.q):
            for j in range(self.q):
                tmp = alpha[i, t] * beta[j, t + 1]
                tmp *= self.A[i][j] * self.B[j][o_tp1]
                xi[t, i, j] = tmp / sum_alphabeta[t]
    return l, gamma, xi


def m_step(self, Os, gamma, xi):
    atop = np.sum(xi[:-1, :, :], axis=0)
    abottom = np.sum(atop, axis=1, keepdims=1)
    Os = np.array(Os)
    btop = np.zeros((self.o, self.q))
    for o in range(self.o):
        tmp = gamma[:, Os == self.id2o[o]].sum(axis=1)
        btop[o, :] = tmp
    bbottom = gamma.sum(axis=1)
    return atop, abottom, btop, bbottom


def update(self, A, B, pi):
    for i in range(self.q):
        for j in range(self.q):
            self.A[i][j] = A[i, j]
        self.pi[i] = pi[i]
        for o in range(self.o):
            self.B[i][o] = B[o, i]


def learn(self, Osamples, iters=10):
    ls = []
    N = len(Osamples)
    for _ in tqdm(range(iters)):
        tmp, at, ab, bt, bb, pt = 0, 0, 0, 0, 0, 0
        for Os in Osamples:
            l, gamma, xi = self.e_step(Os)
            tmp += np.log(l)
            atop, abottom, btop, bbottom = self.m_step(Os, gamma, xi)
            at += atop
            ab += abottom
            bt += btop
            bb += bbottom
            pt += gamma[:, 0]
        ls.append(tmp)
        pi = pt / N
        A = at / ab
        B = bt / bb
        self.update(A, B, pi)
    return ls


def add_patch(obj):
    obj.computeLikelihood = lambda x: likelihood(obj, x)
    obj.decode = lambda x: decode(obj, x)
    obj.likelihood_backwards = lambda x: likelihood_backwards(obj, x)
    obj.samples = lambda x: samples(obj, x)
    obj.e_step = lambda x: e_step(obj, x)
    obj.m_step = lambda x, y, z: m_step(obj, x, y, z)
    obj.update = lambda x, y, z: update(obj, x, y, z)
    obj.learn = lambda x, y: learn(obj, x, y)
    return obj

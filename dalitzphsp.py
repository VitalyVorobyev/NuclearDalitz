""" Dalitz plot phase space description """

import numpy as np

def Kallen(Msq, m1sq, m2sq):
    """ Kallen's triangle function """
    return np.clip((Msq - m1sq - m2sq)**2 - 4*m1sq*m2sq, 0, a_max=None)

def two_body_momentum(Msq, m1sq, m2sq):
    """ """
    return 0.5 * np.sqrt(Kallen(Msq, m1sq, m2sq) / Msq)


def Kibble(s, s3, s2, m1sq, m2sq, m3sq):
    """ If inside Dalitz phase space """
    return np.abs(2. * s3 * (s2 - m1sq - m3sq) - (s3 + m1sq - m2sq) * (s - s3 - m3sq)) <\
           np.sqrt(Kallen(s, m3sq, s3) * Kallen(s3, m1sq, m2sq))


class DalitzPhsp():
    """ A Dalitz phase space tool. Only required features implemented """

    def __init__(self, M, A, B, C):
        """ Constructor. Args:
             - M: mother particle mass
             - A, B, C: daughter particles masses
        """
        self.da = np.array([A, B, C])
        self.daSq = self.da**2
        self.setM(M)

    def setM(self, M):
        """ Set new mother particle mass """
        self.mo = M
        self.moSq = self.mo**2
        self.msqsum = self.mo**2 + self.daSq.sum()
        self.mACsqRange  = np.array([(self.da[0] + self.da[2])**2, (M - self.da[1])**2])
        self.mABsqRange  = np.array([(self.da[0] + self.da[1])**2, (M - self.da[2])**2])
        self.mBCsqRange  = np.array([(self.da[1] + self.da[2])**2, (M - self.da[0])**2])

    def linspaceAB(self, bins=250):
        """ """
        return np.linspace(self.mABsqRange[0], self.mABsqRange[1], bins)

    def linspaceAC(self, bins=250):
        """ """
        return np.linspace(self.mACsqRange[0], self.mACsqRange[1], bins)

    def linspaceBC(self, bins=250):
        """ """
        return np.linspace(self.mBCsqRange[0], self.mBCsqRange[1], bins)

    def mgridABAC(self, b1, b2=None):
        """ """
        if b2 is None:
            b2 = b1
        dAB = (self.mABsqRange[1] - self.mABsqRange[0]) / b1
        dAC = (self.mACsqRange[1] - self.mACsqRange[0]) / b2
        return (np.meshgrid(self.linspaceAB(b1), self.linspaceAC(b2)), dAB*dAC)

    def mgridACBC(self, b1, b2=None):
        """ """
        if b2 is None:
            b2 = b1
        dAC = (self.mACsqRange[1] - self.mACsqRange[0]) / b1
        dBC = (self.mBCsqRange[1] - self.mBCsqRange[0]) / b2
        return (np.meshgrid(self.linspaceAC(b1), self.linspaceBC(b2)), dBC*dAC)

    def eB_AB(self, mABsq):
        """ E(B) in the (AB) frame """
        return 0.5 * (mABsq - self.daSq[0] + self.daSq[1]) / np.sqrt(mABsq)

    def eA_AB(self, mABsq):
        """ E(A) in the (AB) frame """
        return 0.5 * (mABsq - self.daSq[1] + self.daSq[0]) / np.sqrt(mABsq)

    def eC_AB(self, mABsq):
        """ E(C) in the (AB) frame """
        return 0.5 * (self.moSq - mABsq - self.daSq[2]) / np.sqrt(mABsq)

    def eC_AC(self, mACsq):
        """ E(C) in the (AC) frame """
        return 0.5 * (mACsq - self.daSq[0] + self.daSq[2]) / np.sqrt(mACsq)

    def eB_AC(self, mACsq):
        """ E(B) in the (AC) frame """
        return 0.5 * (self.moSq - mACsq - self.daSq[1]) / np.sqrt(mACsq)

    def mBCsqLims(self, eB, eC):
        """ m^2(BC) limits from E(B) and E(C) """
        pB, pC = np.sqrt(eB**2 - self.daSq[1]), np.sqrt(eC**2 - self.daSq[2])
        return ((eB + eC)**2 - (pB + pC)**2,
                (eB + eC)**2 - (pB - pC)**2)

    def mACsqLims(self, eA, eC):
        """ m^2(BC) limits from E(A) and E(C) """
        pA, pC = np.sqrt(eA**2 - self.daSq[0]), np.sqrt(eC**2 - self.daSq[2])
        return ((eA + eC)**2 - (pA + pC)**2,
                (eA + eC)**2 - (pA - pC)**2)

    def mACsqLimAB(self, mABsq):
        """ m^2(AC) limits from m^2(AB) """
        return self.mACsqLims(self.eA_AB(mABsq), self.eC_AB(mABsq))

    def mBCsqLimAB(self, mABsq):
        """ m^2(BC) limits from m^2(AB) """
        return self.mBCsqLims(self.eB_AB(mABsq), self.eC_AB(mABsq))

    def mBCsqLimAC(self, mACsq):
        """ m^2(BC) limits from m^2(AC) """
        return self.mBCsqLims(self.eB_AC(mACsq), self.eC_AC(mACsq))

    def inPhspABBC(self, mABsq, mBCsq):
        mBCsqL = self.mBCsqLimAB(mABsq)
        return (mBCsqL[1] > mBCsq) & (mBCsqL[0] < mBCsq)

    def inPhspABAC(self, mABsq, mACsq):
        mACsqL = self.mACsqLimAB(mABsq)
        return (mACsqL[1] > mACsq) & (mACsqL[0] < mACsq)

    def inPhspACBC(self, mACsq, mBCsq):
        mBCsqL = self.mBCsqLimAC(mACsq)
        return (mBCsqL[1] > mBCsq) & (mBCsqL[0] < mBCsq)

    def mZsq(self, mXsq, mYsq):
        """ The third Dalitz variable from the other two """
        return self.msqsum - mXsq - mYsq

    def kine(self, ma, mbcsq):
        return ((self.mo - ma)**2 - mbcsq) / (2. * self.mo)

    def KineA(self, mBCsq):
        """ Kinetic energy of A from mBCsq """
        return self.kine(self.da[0], mBCsq)

    def KineB(self, mACsq):
        """ Kinetic energy of B from mACsq """
        return self.kine(self.da[1], mACsq)

    def KineC(self, mABsq):
        """ Kinetic energy of C from mABsq """
        return self.kine(self.da[2], mABsq)

    @staticmethod
    def iKineFramejk(mMo, mi, mjksq):
        """ i'th kine energy in the (jk) frame """
        return ((mMo - mi)**2 - mjksq) / (2. * mMo)

    def mijsq(self, mMosq, miksq, mjksq):
        """ m(ij)^2 given m(ik)^2 and m(jk)^2 """
        return mMosq + self.daSq.sum() - miksq - mjksq

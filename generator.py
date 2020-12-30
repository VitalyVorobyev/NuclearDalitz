import numpy as np

def generate_phsp(phsp, neve, seed=0, chunkSize=10**5):
    """ Generates Dalitz variables uniformly """
    mABsqLo, mABsqHi = phsp.mABsqRange
    mACsqLo, mACsqHi = phsp.mACsqRange

    rng = np.random.default_rng(seed=seed)
    sample = np.empty((0, 2))
    while sample.size < neve:
        chunk = rng.uniform(low=[mABsqLo, mACsqLo], high=[mABsqHi, mACsqHi], size=(chunkSize, 2))
        mask = phsp.inPhspACBC(chunk[:,0], chunk[:,1])
        sample = np.row_stack((sample, chunk[mask]))

    return sample[:neve]

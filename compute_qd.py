import numpy as np
import sys

class Compute_qd():

    @staticmethod
    def compute_qd(Ak):
        eps = sys.float_info.epsilon
        # ENTROPY = []
        # for i in range(Ak.shape[1]):
        # sample = Ak[:,i,:,:]
        hist, bin_edges = np.histogram(Ak, bins=100)
        aa = hist/sum(hist)+eps
        en = - sum(aa*np.log(aa))

        return en
        # ENTROPY.append(en)
        # return ENTROPY


#%%
if __name__ == "__main__":
    sample = np.random.randint(256, size=(60000,114, 114))
    entropy = compute_qd(sample)
    print(entropy)
# %%

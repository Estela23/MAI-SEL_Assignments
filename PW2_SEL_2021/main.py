import math
import random


def runif(M):
    n = random.randint(1, M)
    print(n)


NT = [1, 10, 25, 50, 75, 100]   # Number of trees desired

n_feat = 17    # Number of features

# Number of random features used in the splitting of the nodes (RF)
F = {1, 3, int(math.log(n_feat + 1, 2)), math.floor(math.sqrt(n_feat))}

# Number of random features used in the splitting of the nodes (DF)
F = {int(n_feat/4), int(n_feat/2), int(3*n_feat/4), runif(n_feat)}






import math
import numpy as np
import collections

data = np.array([1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5])
data_collection = collections.Counter(data)
print(data_collection)

def emtropy(collection):
    total = sum(collection.values())
    ent = 0
    for count in collection.values():
        # for i to j -= 𝑝(𝑎𝑗)𝑙𝑜𝑔2(𝑝(𝑎𝑗))
        # print(count)
        p = count / total
        ent -= p * math.log2(p)
        # print(ent)
    return ent

if __name__ == "__main__":
    print(emtropy(data_collection))

#ans: 2.1219

#ans: Counter({np.int64(4): 8, np.int64(1): 4, np.int64(3): 4, np.int64(2): 2, np.int64(5): 2})
# 2.121928094887362
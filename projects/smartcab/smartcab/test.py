import numpy as np
import random

dic = {     '0': 0,
            '1': 0,
            '2': 0,
            '3': 0  }

vals=np.array(dic.values())
keys=list(dic.keys())
indexes = np.where(vals==vals.max())[0]
i = random.choice(indexes)
print(int(keys[i]))
import numpy as np
import matplotlib
matplotlib.rcParams["backend"] = "TkAgg"
import matplotlib.pyplot as plt

r = np.load("reward_history.npy").tolist()
plt.plot(r)
plt.show()

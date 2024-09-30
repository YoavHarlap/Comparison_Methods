import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 2*np.pi, 1000000)
y = np.sin(x)+np.cos(60*x)
print(y)
plt.plot(x,y,label ="sin")
plt.legend()



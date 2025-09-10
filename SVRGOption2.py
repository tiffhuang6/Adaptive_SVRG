import numpy as np
import matplotlib.pyplot as plt
import random

def stochGradient(x, n):
    list = np.zeros(n)
    sum = 0
    for i in range(n):
        list[i] = (1 / n) * (sum + x[i])
        sum += x[i]

    return list

def summation(n,w,x,y):
    sum = 0
    for i in range(n):
        newTerm = 2 * (w[i] * x[i]) - y[i]
        sum += newTerm

    return sum

def sumVector(x,n):
    sum = 0
    for i in range(n):
        sum += x[i]
    return sum



w_tilda = np.random.randn(50)
sGradient = stochGradient(w_tilda, 50)
w = np.zeros(50)
x = np.random.randn(50)
y = [i + 1 for i in range(50)]
m = 100
n = 50
LR = 3


for i in range(n): #k
    w[i] = w_tilda[i]
    mu = (1/n) * summation(n,w_tilda,x,y)
    for j in range(m): #t
        index = random.randint(1, n - 1)
        w[index] = w[index - 1] - LR * (2 * (w[index - 1]
        * x[index - 1]) - y[index - 1]) - (2 * (w_tilda[index] * x[index]) - y[index]) + mu

    if (i + 1 < n):
        w_tilda[i + 1] = (1/m) * sumVector(w,n)

plt.xlabel("Indices")
plt.ylabel("w value")

plt.plot(w, color="red")
plt.plot(sGradient, color="green")
plt.show()








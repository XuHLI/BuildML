import numpy as np 
import scipy as sp 
import matplotlib.pyplot as plt
# load data 
data = np.loadtxt('web_traffic.tsv')

# split data
x = data[:,0]
y = data[:,1]

# an error function
def error(f,x,y):
	return sp.sum((f(x)-y)**2)

# polynomial fit
fp1, res, rank, sv, rcond = sp.polyfit(x,y,2,full=True)

print('Model parameters: %s'% fp1)
# print(res)

f1 = sp.poly1d(fp1)
# print(error(f1,x,y))

# generate x values for plotting
fx = sp.linspace(0,x[-1],1000)
plt.scatter(x,y)
plt.title('Web traffic over the last month')
plt.xlabel('Time')
plt.ylabel('Hits/hour')
plt.xticks([w*7*24 for w in range(10)],
	['week %i'% w for w in range(10)])
plt.autoscale(tight=True)
plt.grid()
plt.plot(fx,f1(fx),linewidth=4,color='r')
plt.legend(['d=%i'%f1.order],loc='upper left')
plt.show()
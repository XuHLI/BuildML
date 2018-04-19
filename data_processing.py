import scipy as sp 
import matplotlib.pyplot as plt
import os

DATA_DIR = ''
CHART_DIR = ''

# read data from file web_traffic.tsv using genfromtxt
data = sp.genfromtxt('data/web_traffic.tsv', delimiter='\t')

# list the properties of the data
print('data.shape, data.ndim:', data.shape, data.ndim)

# print the first 10 rows
# print(data[:10])

# Separate the data into two columns
x = data[:,0]
y = data[:,1]

# remove nan 
x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

print([w*6*24 for w in range(10)],['week %i'% w for w in range(10)])

# plot scatter (x,y)
plt.scatter(x,y)
plt.title('Web traffic over the last month')
plt.xlabel('Time')
plt.ylabel('Hits/hour')
plt.xticks([w*7*24 for w in range(10)],
	['week %i'% w for w in range(10)])
plt.autoscale(tight=True)
plt.grid()
plt.show()

plt.savefig(os.path.join(CHART_DIR,'web_traffic.png'))
sp.savetxt(os.path.join(DATA_DIR,'web_traffic.tsv'),list(zip(x,y)),
	delimiter='\t',fmt='%s')


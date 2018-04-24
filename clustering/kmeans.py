from scipy.stats import norm
import scipy as sp
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

seed = 2
sp.random.seed(seed) # reproduce the data

xw1 = norm(loc=0.3, scale=.15).rvs(20)
yw1 = norm(loc=0.3, scale=.15).rvs(20)

xw2 = norm(loc=0.7, scale=.15).rvs(20)
yw2 = norm(loc=0.7, scale=.15).rvs(20)

xw3 = norm(loc=0.2, scale=.15).rvs(20)
yw3 = norm(loc=0.8, scale=.15).rvs(20)

x = sp.append(sp.append(xw1,xw2),xw3)
y = sp.append(sp.append(yw1,yw2),yw3)

def plot_clustering(x,y,title,ymax=None,xmax=None,km=None):
	# plt.figure(num=None,figsize=(8,6))
	if km:
		plt.scatter(x,y,s=50,c=km.predict(list(zip(x,y))))
	else:
		plt.scatter(x,y,s=50)

	
	plt.title(title)
	plt.xlabel('Occurrence word 1')
	plt.ylabel('Occurrence word 2')

	plt.autoscale(tight=True)
	plt.xlim(xmin=0,xmax=1)
	plt.ylim(ymin=0,ymax=1)
	plt.grid(True, linestyle='-',color='0.75')
	
	return plt

num_clusters = 3

# plt.figure(num=None,figsize=(8,6))
i = 1
plot_clustering(x,y,title="Vectors")
# plt.show()
plt.close()

i+=1


mx, my = sp.meshgrid(sp.arange(0,1,0.001),sp.arange(0,1,0.001))

km = KMeans(init='random',n_clusters=num_clusters,verbose=1,n_init=1,
	max_iter=1, random_state=seed)

km.fit(sp.array(list(zip(x,y))))

Z = km.predict(sp.c_[mx.ravel(),my.ravel()]).reshape(mx.shape)
plot_clustering(x,y,"Clustering iteration 1", km=km)
plt.imshow(Z,interpolation='nearest',
	extent=(mx.min(),mx.max(),my.min(),my.max()),
	cmap = plt.cm.Blues,
	aspect='auto', origin='lower')
c1a,c1b,c1c = km.cluster_centers_
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],marker='x',
	linewidth=2,s=100,color='black')
# plt.show()

i+=1
km = KMeans(init='random',n_clusters=num_clusters,verbose=1,n_init=1,
	max_iter=10, random_state=seed)

km.fit(sp.array(list(zip(x,y))))

Z = km.predict(sp.c_[mx.ravel(),my.ravel()]).reshape(mx.shape)
plot_clustering(x,y,"Clustering iteration 2", km=km)
plt.imshow(Z,interpolation='nearest',
	extent=(mx.min(),mx.max(),my.min(),my.max()),
	cmap = plt.cm.Blues,
	aspect='auto', origin='lower')
c2a,c2b,c2c = km.cluster_centers_
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],marker='x',
	linewidth=2,s=100,color='black')

plt.gca().add_patch(
	plt.Arrow(c1a[0],c1a[1],c2a[0]-c1a[0],c2a[1]-c1a[1], width=0.1))
plt.gca().add_patch(
	plt.Arrow(c1b[0],c1b[1],c2b[0]-c1b[0],c2b[1]-c1b[1], width=0.1))

plt.show()



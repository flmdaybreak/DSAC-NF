from pylab import *

def f(x,y): return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)
mu=0

mu1=2
sigma=2
n = 10
sampleNo=90000
q = np.random.normal(mu,sigma,sampleNo)
p = np.random.normal(mu1,sigma,sampleNo)
X,Y = np.meshgrid(p,q)
q1 = q.reshape(300)
p1 = p.reshape(300)
c=q1-p1
plt.pcolormesh()
print(x)
y = np.linspace(-3,3,n)

print(X)
contourf(X, Y, f(X,Y), 8, alpha=.75, cmap='jet')
#C = contour(X, Y, f(X,Y), 8, colors='black', linewidth=.5)
plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

plt.show()
#-*- coding:utf-8 -*-

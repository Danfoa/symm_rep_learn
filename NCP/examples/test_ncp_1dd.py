
#%%


import numpy as np
import torch
import pylab as pl
import torch.nn.functional

#%% create data
n = 5000

sig = 0.3
x = 10*np.random.rand(n,1)
#x = 1.5*np.random.randn(n,1)+5
x_grid = np.linspace(0,10,100)[:,None]


y = 2*np.cos(x)**2+x- np.exp(x/5)  + sig*(x/5+0.1)*np.random.randn(n,1)
y_true_grid = 2*np.cos(x_grid)**2+x_grid- np.exp(x_grid/5)  

pl.figure(3,(4,4))
pl.clf()
pl.scatter(x[:,0],y,alpha=0.5)
pl.plot(x_grid,y_true_grid,color='C1')
pl.xlabel('X')
pl.ylabel('Y')
pl.show()
#%%

# torch model for u

class NetU(torch.nn.Module):

    def __init__(self, r=10,n_hidden=64):
        super(NetU, self).__init__()
        self.fc1 = torch.nn.Linear(1, n_hidden[0])
        self.fc2 = torch.nn.Linear(n_hidden[0], n_hidden[1])
        self.fc3 = torch.nn.Linear(n_hidden[1], n_hidden[2])
        self.fc4 = torch.nn.Linear(n_hidden[2], n_hidden[3])
        self.fc5 = torch.nn.Linear(n_hidden[3], n_hidden[4])
        self.fc6 = torch.nn.Linear(n_hidden[4], r)

    def forward(self, x):
        activation = torch.tanh
        x = activation(self.fc1(x))
        x = activation(self.fc2(x))
        x = activation(self.fc3(x))
        x = activation(self.fc4(x))
        x = activation(self.fc5(x))
        x = self.fc6(x)
        return x

# torch model for v

class NetV(torch.nn.Module):

    def __init__(self, r=10,n_hidden=64):
        super(NetV, self).__init__()
        self.fc1 = torch.nn.Linear(1, n_hidden[0])
        self.fc2 = torch.nn.Linear(n_hidden[0], n_hidden[1])
        self.fc3 = torch.nn.Linear(n_hidden[1], n_hidden[2])
        self.fc4 = torch.nn.Linear(n_hidden[2], n_hidden[3])
        self.fc5 = torch.nn.Linear(n_hidden[3], n_hidden[4])
        self.fc6 = torch.nn.Linear(n_hidden[4], r)
        

    def forward(self, x):
        activation = torch.tanh
        x = activation(self.fc1(x))
        x = activation(self.fc2(x))
        x = activation(self.fc3(x))
        x = activation(self.fc4(x))
        x = activation(self.fc5(x))
        x = self.fc6(x)
        return x


# torch model for operator

class NetP(torch.nn.Module):

    def __init__(self,u,v, r=10):
        super(NetP, self).__init__()
        self.u = u
        self.v = v
        self.U = torch.eye(r)
        self.V = torch.eye(r)
        self.log_sigma = torch.nn.Parameter(torch.zeros(r))
        self.sigma_ = torch.zeros(r)
        self.regression = False

    def sigma(self):
        return torch.exp(-self.log_sigma**2)

    def forward(self, x, y):
        U = self.u(x)
        V = self.v(y)
        return 1+torch.sum(U[:,None,:]*V[None,:,:]*self.sigma()[None,None,:],2)

    def joint(self, x, y):
        U = self.u(x)
        V = self.v(y)
        return 1+torch.mv(U*V,self.sigma())
        #return 1+torch.sum(U*V*self.sigma[None,:],1)

    def predict(self, xnew, y,f=None):
        if f is None:
            f = lambda x: x
        U = torch.mm(self.u(xnew),self.U)
        V = torch.mm(self.v(y),self.V)
        if self.regression is True:
            sigma_ = self.sigma_
        else:
            sigma_ = self.sigma()
        y2 = f(y)
        P = 1+torch.sum(U[:,None,:]*V[None,:,:]*sigma_[None,None,:],2)
        return torch.mm(P,y2)/y2.shape[0]

    def predict_max(self, xnew, y):
        U = torch.mm(self.u(xnew),self.U)
        V = torch.mm(self.v(y),self.V)
        if self.regression is True:
            sigma_ = self.sigma_
        else:
            sigma_ = self.sigma()
        P = 1+torch.sum(U[:,None,:]*V[None,:,:]*sigma_[None,None,:],2)
        return y[P.argmax(1)]


def loss_ncp(x,y,xp,yp,p):
    # this one is now correct, there was a typo in formula in Eq (6)
    return torch.mean(p.joint(xp,yp)**2)-2*torch.mean(p.joint(x,y))+1

def loss_ncp2(x,y,xp,yp,p):
    # this was not correctly implementd, 
    # in the paper forumulas (10) and (11) are correct, 
    # but as stated in the algorithm
    # one needs to average over batch_size^2 points
    # e.g. u(x_i)^T Sigma v(y_j') 
    # or (u(x_i)-u(x_j')^T Sigma (v(y_i) - v(y_j'))
    U = p.u(x)
    V = p.v(y)
    Up = p.u(xp)
    Vp = p.v(yp)
    s = p.sigma() # * 0 + 1
    # Product of marginals
    t1 = torch.mean((torch.einsum('ik,jk,k->ij',U,Vp,s)+1)**2)/2
    t2 = torch.mean((torch.einsum('ik,jk,k->ij',Up,V,s)+1)**2)/2
    # Joint distribution
    t4 = - torch.mean(torch.einsum('ik,ik,k->i',Up,Vp,s))
    return t1+t2+(2*t4)-1

def loss_ncp3(x,y,ncp):
    fx = ncp.u(x)  # n x r
    hy = ncp.v(y)  # n x r

    n_samples = x.shape[0]
    fx_centered = fx - fx.mean(dim=0, keepdim=True)
    hy_centered = hy - hy.mean(dim=0, keepdim=True)

    E = torch.einsum('nr,nc->rc', fx_centered, hy_centered) / n_samples

    Vx = torch.einsum('nr,nc->rc', fx_centered, fx_centered) / (n_samples - 1) + 1e-6 * torch.eye(fx.shape[1])
    Vy = torch.einsum('nr,nc->rc', hy_centered, hy_centered) / (n_samples - 1) + 1e-6 * torch.eye(hy.shape[1])

    A = torch.einsum('ab,bc,cd,de->ae', E, Vy, torch.transpose(E, 0, 1), Vx)

    return - 2 * torch.sum(E**2) + torch.trace(A)


def reg_term(x,y,xp,yp,p):
    # this was not correctly implementd, 
    # in the paper forumulas (10) and (11) are correct, 
    # but as stated in the algorithm
    # one needs to average over batch_size^2 points
    # e.g. u(x_i)^T v(y_j') 
    # or (u(x_i)-u(x_j')^T (v(y_i) - v(y_j'))
    U = p.u(x)
    V = p.v(y)
    Up = p.u(xp)
    Vp = p.v(yp)
    s = p.sigma()
    t1_u = torch.mean((torch.einsum('ik,jk->ij',U,Up)+1)**2)
    t1_v = torch.mean((torch.einsum('ik,jk->ij',V,Vp)+1)**2)
    t2_u = - torch.einsum('ik,ik->',U,U)/U.shape[0] - torch.einsum('ik,ik->',Up,Up)/Up.shape[0]
    t2_v = - torch.einsum('ik,ik->',V,V)/V.shape[0] - torch.einsum('ik,ik->',Vp,Vp)/Vp.shape[0]
    #t1 = torch.mean(torch.sum(U*Up,1)**2)+torch.mean(torch.sum(V*Vp,1)**2)
    #t2 = - torch.mean(torch.sum((U-Up),1))-torch.mean(torch.sum((V-Vp),1))
    return t1_u+t1_v+t2_u+t2_v+(U.shape[1] + V.shape[1]-2)


#%% optimization

# create models
r = 20
n_hidden=[32,64,32,64,32]
reg = r*1e-1
u = NetU(r,n_hidden=n_hidden)
v = NetV(r,n_hidden=n_hidden)
p = NetP(u,v,r=r)

x_t = torch.tensor(x).float()
y_t = torch.tensor(y).float()


# optimizer
optimizer = torch.optim.Adam(p.parameters(), lr=5e-3)

decayRate = 0.99
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

# training
niter = 5000
batch_size = n//4
losses = []
losses_ncp = []

x_batch = x_t
y_batch = y_t

for i in range(niter):
    optimizer.zero_grad()

    if batch_size<n:
        perm_ = torch.randperm(n)
        x_batch = x_t[perm_[:batch_size]]
        y_batch = y_t[perm_[:batch_size]]

    if False:  # loss NCP Eq (6)
        xp = x_batch[torch.randperm(batch_size)]
        yp = y_batch[torch.randperm(batch_size)]
        loss_ncp_ = loss_ncp(x_batch,y_batch,xp,yp,p)
    elif False:
        # loss NCP 2 Eq (10)
        perm_ = torch.randperm(batch_size)
        xp = x_batch[perm_]
        yp = y_batch[perm_]
        loss_ncp_ = loss_ncp2(x_batch,y_batch,xp,yp,p) 
    else:
        # loss NCP 3 Eq (10)
        xp = x_batch[torch.randperm(batch_size)]
        yp = y_batch[torch.randperm(batch_size)]
        loss_ncp_ = loss_ncp3(x_batch,y_batch,ncp=p)

    losses_ncp.append(loss_ncp_)

    loss = loss_ncp_ + reg*reg_term(x_batch,y_batch,xp,yp,p)
    loss.backward()

    losses.append(loss.item())

    optimizer.step()
    #my_lr_scheduler.step()
    print(i,f" Total loss {loss.item():0.2f} | NCP loss {loss_ncp_:0.2f}")

pl.plot(losses)
pl.show()

#%%

# predict
ypred = p.predict(x_t,y_t).detach().numpy()
ypred2= p.predict_max(x_t,y_t).detach().numpy()


pl.figure(3,(4,4))
pl.clf()


pl.figure(3,(4,4))
pl.clf()
pl.scatter(x[:,0],y,alpha=0.5,color='C0',label='Data')
pl.plot(x_grid,y_true_grid,color='C1',label='True E[Y|X]')
pl.scatter(x,ypred,alpha=0.1,color='C2', label='Pred E[Y|X]')
pl.xlabel('X')
pl.ylabel('Y')
pl.plot(x_grid,y_true_grid,color='C1')
pl.title('Prediction E[Y|X]')
pl.legend()
pl.show()

pl.figure(4)
pl.clf()
pl.scatter(y,ypred,alpha=0.5)
#pl.scatter(y,ypred2,alpha=0.5)
ax = pl.axis()
pl.plot([0,18],[0,18],color='C2')
pl.axis(ax)
pl.xlabel('True y')
pl.ylabel("Pred Y")
pl.show()
print(np.mean((ypred-y)**2))
#print(np.mean((ypred2-y)**2))


# #%% plot eigenfunctions

# U = u(x_t).detach().numpy()
# V = v(y_t).detach().numpy()

# pl.figure(3,(10,10))

# for i in range(r):
#     pl.subplot(4,4,i+1)
#     pl.scatter(x[:,0],x[:,1],c=U[:,i],alpha=0.5)
#     pl.title('u'+str(i))

# pl.figure(4,(10,10))

# for i in range(r):
#     pl.subplot(4,4,i+1)
#     pl.scatter(y,V[:,i],alpha=0.5)
#     pl.title('v'+str(i))
#     pl.xlabel('y')
#     pl.ylabel('v')


#%% plot distribution for 1 sample
i = 1

nbins = 20
y_grid = np.linspace(0,6,nbins)
y_grid_t = torch.tensor(y_grid)

x_new= x_t[i:i+1]


def f(y):
    return 1.0*(y<=y_grid_t[None,:])

cdf = p.predict(x_new,y_t,f).detach().numpy().ravel()
df = np.concatenate(([0,],np.diff(cdf)))

# pl.figure(3,(10,5))

# pl.subplot(1,2,1)
# pl.scatter(x[:,0],x[:,1],c=y,alpha=0.5)
# pl.scatter(x_new[:,0],x_new[:,1],c='r',s=100)

pl.subplot(1,2,2)
pl.hist(y,bins=20,alpha=0.5,density=True, label='P(Y)')
pl.plot(y_grid-(y_grid[1]-y_grid[0])/2,df,label='P(Y|X)')
#pl.plot(y_grid,cdf,alpha=.5)
lims = pl.axis()
pl.plot([ypred[i],ypred[i]],[0,1],label='Y pred')
pl.plot([y[i],y[i]],[0,1],label='Y')
pl.axis(lims)
pl.title('P(Y|X)')
pl.legend()
pl.show()

#%% plot quantiles


nbins = 100
y_grid = np.linspace(0,6,nbins)
y_grid_t = torch.tensor(y_grid)

ngrid= len(x_grid)
x_grid_t = torch.tensor(x_grid).float()

def f(y):
    return 1.0*(y<=y_grid_t[None,:])

y_grid_pred = p.predict(x_grid_t,y_t).detach().numpy().ravel()

cdf = p.predict(x_grid_t,y_t,f).detach().numpy()
cdf= cdf/cdf.max(1,keepdims=True)

q20 = [np.searchsorted(cdf[i],0.2) for i in range(ngrid)]
q80 = [np.searchsorted(cdf[i],0.8) for i in range(ngrid)]


pl.figure(3,(4,4))
pl.clf()
pl.scatter(x[:,0],y,alpha=0.5,color='C0',label='Data')
pl.plot(x_grid,y_true_grid,color='C1',label='True E[Y|X]')
pl.scatter(x_grid,y_grid_pred,alpha=0.5,color='C2', label='Pred E[Y|X]')
pl.fill_between(x_grid.ravel(),y_grid[q20],y_grid[q80],alpha=0.2,color='C2',label='60% CI')
pl.xlabel('X')
pl.ylabel('Y')
pl.plot(x_grid,y_true_grid,color='C1')
pl.title('Prediction E[Y|X]')
pl.legend()
pl.show()
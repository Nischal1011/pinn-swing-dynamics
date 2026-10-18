# Throwaway feasibility prototype (Oct 8). Run from repo root: python notebooks/prototype_forward_pinn.py <lambda> <max_iters>
import sys, time, numpy as np, scipy.io as s, torch
from scipy.stats import qmc
torch.set_default_dtype(torch.float64); torch.manual_seed(1234); np.random.seed(1234)
lam=float(sys.argv[1]); iters=int(sys.argv[2])
D=s.loadmat('reference/misyris2019_original/Data/swingEquation_inference.mat'); t=D['t'].ravel(); x=D['x'].ravel(); U=D['usol']  # (P,t)
P,T=np.meshgrid(x,t,indexing='ij'); Xs=np.c_[P.ravel(),T.ravel()]; us=U.ravel()[:,None]
# paper's boundary selection: t=0 line, P=min traj, P=max traj
bnd=np.vstack([np.c_[x,np.zeros_like(x),U[:,0]], np.c_[np.full_like(t,x[0]),t,U[0]], np.c_[np.full_like(t,x[-1]),t,U[-1]]])
bnd=bnd[np.random.choice(len(bnd),40,replace=False)]
lb=np.array([0.08,0.]); ub=np.array([0.18,20.])
Xf=np.vstack([lb+(ub-lb)*qmc.LatinHypercube(d=2,seed=1234).random(8000), bnd[:,:2]])
tt=lambda a: torch.tensor(a)
Xu,uu=tt(bnd[:,:2]),tt(bnd[:,2:]); Pf=tt(Xf[:,:1]); Tf=tt(Xf[:,1:]).requires_grad_()
lbt,ubt=tt(lb),tt(ub)
layers=[2]+[10]*5+[1]; mods=[]
for a,b in zip(layers[:-1],layers[1:]):
    L=torch.nn.Linear(a,b); torch.nn.init.xavier_normal_(L.weight); torch.nn.init.zeros_(L.bias); mods+=[L,torch.nn.Tanh()]
net=torch.nn.Sequential(*mods[:-1])
f=lambda X: net(2*(X-lbt)/(ubt-lbt)-1)
def loss():
    l=((f(Xu)-uu)**2).mean()
    if lam>0:
        d=f(torch.cat([Pf,Tf],1)); dt=torch.autograd.grad(d.sum(),Tf,create_graph=True)[0]; dtt=torch.autograd.grad(dt.sum(),Tf,create_graph=True)[0]
        l=l+lam*((0.4*dtt+0.15*dt+0.2*torch.sin(d)-Pf)**2).mean()
    return l
opt=torch.optim.LBFGS(net.parameters(),lr=1,max_iter=iters,max_eval=int(iters*1.25),history_size=50,tolerance_grad=1e-9,tolerance_change=1e-15,line_search_fn='strong_wolfe')
def closure(): opt.zero_grad(); l=loss(); l.backward(); return l
t0=time.time(); opt.step(closure); el=time.time()-t0
with torch.no_grad(): up=f(tt(Xs)).numpy()
print(f'lam={lam} iters={opt.state[opt._params[0]]["n_iter"]} time={el:.1f}s loss={loss().item():.2e} relL2={np.linalg.norm(up-us)/np.linalg.norm(us):.3e}')

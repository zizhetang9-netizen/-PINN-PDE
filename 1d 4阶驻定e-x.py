import torch
import math
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import random
import time
torch.set_printoptions(precision=8)

torch.set_default_dtype(torch.float64)

x = torch.linspace(0, 1, 100, requires_grad=True)[:, None]

y_true = torch.exp(-x)

model = nn.Sequential(
    nn.Linear(1,10),nn.Sigmoid(),
    nn.Linear(10,1)
    )
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
loss_f = nn.MSELoss(reduction='mean')

max_abs1 = 1000
def autograd(y,x):
    dif = torch.autograd.grad(y,x,grad_outputs=torch.ones_like(y),create_graph=True)[0]
    return dif
start = time.time()
for i in range(5000):
    y = model(x)
    dif1 = autograd(y,x)
    dif2 = autograd(dif1,x)
    dif3 = autograd(dif2,x)
    dif4 = autograd(dif3,x)
    dif_f = dif4 + dif2 - 6*y*dif1**2-3*y**2*dif2 + 2*dif2**2 + 2*dif1*dif3
    loss1 =loss_f(dif_f,2*torch.exp(-x)-9*torch.exp(-3*x)+4*torch.exp(-2*x)) #负指数方程平方
    # loss1 = loss_f(dif_f,-6*torch.cos(x)**2*(torch.sin(x)+2)+3*(torch.sin(x)+2)**2*torch.sin(x)+2*torch.sin(x)**2-2*torch.cos(x)**2) #真解为2+sinx
    # loss1 = loss_f(dif_f,-6*(torch.cos(x))*torch.sin(x)**2+3*(torch.cos(x))**2*torch.cos(x)+2*torch.cos(x)**2-2*torch.sin(x)**2) # 真解为cosx
    # loss1 = loss_f(dif_f,-6*torch.cos(x)**2*(torch.sin(x))+3*(torch.sin(x))**2*torch.sin(x)+2*torch.sin(x)**2-2*torch.cos(x)**2) #真解为sinx
    loss2 = ((y[0]-1)**2+(y[99]-math.exp(-1))**2)+((dif1[0]+1) **2+(dif1[99]+math.exp(-1)))**2# 负指数0+1
    # loss2 = 0.9*((y[0]-1)**2+(y[99]-math.exp(-1))**2)+0.1*((dif2[0]-1)**2+(dif2[99]-math.exp(-1))**2)# 负指数 0+2
    # loss2 = 0.9*((y[0]-1)**2+(y[99]-math.exp(-1))**2)+0.1*((dif3[0]+1)**2+(dif3[99]+math.exp(-1))**2)# 负指数 0+3
    # loss2 = (dif1[0]+1)**2+(dif1[99]+math.exp(-1))**2+(dif2[0]-1)**2+(dif2[99]-math.exp(-1))**2# 负指数 1+2
    # loss2 = (dif1[0]+1)**2+(dif1[99]+math.exp(-1))**2+(dif3[0]+1)**2+(dif3[99]+math.exp(-1))**2# 负指数 1+3
    # loss2 = (dif2[0]-1)**2+(dif2[99]-math.exp(-1))**2+(dif3[0]+1)**2+(dif3[99]+math.exp(-1))**2# 负指数 2+3
    # loss2 =((y[0]-3)**2+(y[99]-1)**2)+((dif1[0]-0)**2+(dif1[99]-0)**2)# cosx+2 0阶和1阶导作为边界
    # loss2 =((y[0]-1)**2+(y[99]+1)**2)+((dif1[0]-0)**2+(dif1[99]-0)**2)# cosx 0阶和1阶导作为边界
    # loss2 = (y[0]-1)**2+(y[99]+1)**2 +(dif2[0]+1)**2+(dif2[99]-1)**2 #   cosx 0阶和2阶
    # loss2 = ((y[0]-1)**2+(y[99]+1)**2) +((dif3[0]-1)**2+(dif3[99]+1)**2) # cosx  0和3阶
    # loss2 = (dif1[0]-0)**2+(dif1[99]-0)**2+(dif2[0]+1)**2+(dif2[99]-1)**2  # cosx 1阶和2阶
    # loss2 = (dif1[0]-0)**2+(dif1[99]-0)**2+(dif3[0]-1)**2+(dif3[99]+1)**2 # cosx 1阶导和3阶导
    # loss2 = (dif2[0]+1)**2+(dif2[99]-1)**2+(dif3[0]-1)**2+(dif3[99]+1)**2  #cosx 2阶导和3阶导
    # loss2 = ((y[0]-0)**2+(y[99]-0)**2)+((dif1[0]-1)**2+(dif1[99]+1)**2)# sinx +2 0阶和1阶导作为边界
    loss = loss1 + loss2
    abs_error = torch.abs(y_true - model(x))
    max_abs = torch.max(abs_error, 0)[0]
    # min_abs = torch.min(abs_error, 0)[0]
    # relative_error = torch.abs(abs_error / y_true)
    # mean_re = torch.mean(relative_error)
    # relative_error = torch.abs (torch.abs(y_true - model(x))/y_true)
    # max_re = torch.max(relative_error , 0)[0]  #最大相对误差
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if max_abs < max_abs1:
        max_abs1 = max_abs
        torch.save(model.state_dict(), 'model2.pth')

    if (i+1)%50 == 0:
        print(f'loss={loss},loss1={loss1},loss2={loss2}')
end = time.time()
time = end-start
new_model=nn.Sequential(nn.Linear(1,10),nn.Sigmoid(),nn.Linear(10,1))
new_model.load_state_dict(torch.load('model2.pth'))
y_t = new_model(x)
abs_error = torch.abs(y_true-y_t)
max_abs = torch.max(abs_error,0)[0]
min_abs = torch.min(abs_error,0)[0]
relative_error= torch.abs(abs_error/y_true)
max_relative = torch.max(relative_error,0)[0]
min_relative = torch.min(relative_error,0)[0]
mean_re = torch.mean(relative_error)
l2 = torch.norm(y_true-y_t,p=2)
print(f'最大绝对误差={max_abs},最小绝对误差={min_abs}，最大相对误差={max_relative},最小相对误差={min_relative },相对误差平均值={mean_re}，L2误差={l2}')
print(f'运行时间：{time}')

fig=plt.figure()
plt.plot(x.detach().numpy(), y_t.detach().numpy(), linewidth=1, color="g", )
plt.plot(x.detach().numpy(), y_true.detach().numpy(), linewidth=1, color="b", )
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['Neural network solution','analytic solution'],loc='best')    #加图列

fig2 = plt.figure("绝对误差")          #画出每一点的绝对误差
plt.plot(x.detach().numpy(),abs_error.detach().numpy())
plt.xlabel('x')
plt.ylabel('absolute error')
plt.ticklabel_format(style='sci',scilimits=(-1,2),axis='y')
fig3 = plt.figure("相对误差")          #画出每一点的相对误差
plt.plot(x.detach().numpy(),relative_error.detach().numpy())
plt.xlabel('x')
plt.ylabel('relative error')
plt.ticklabel_format(style='sci',scilimits=(-1,2),axis='y')
plt.show()

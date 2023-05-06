import numpy as np
from math import e
import random
import matplotlib.pyplot as plt
import json
import matplotlib as mpt
mpt.rcParams['font.sans-serif'] = ['SimHei']
import math
from scipy import integrate


N = 50000
n = int(N/4)
destination=int(N/4)
c = -1

betas=np.linspace(0.1,3,50)
alphas =np.linspace(0.1,3,30)
sum_prob,E_length={},{}

for beta in betas:
        sum_prob[beta]=0
        for i in range(1,2*n):
                sum_prob[beta]+=i**(-beta)
        E_length[beta]=0
        for i in range(1,2*n):
                E_length[beta]+=i**(1-beta)
        E_length[beta]=E_length[beta]/sum_prob[beta]


def distance(index1, index2):
    if abs(index1-index2) >= n:
        return N-abs(index1-index2)
    else:
        return abs(index1-index2)


def get_shortcut_dest(current, beta):
    total_prob = sum_prob[beta]
    u = random.random()*total_prob
    dis, cdf = 0, 0
    while cdf < u:
        dis += 1
        cdf += dis**(-beta)
    pon = random.choice([-1, 1])
    return current+pon*dis


def walk(current_index, lambda0, restriction, beta):
    num = np.random.poisson(lambda0)
    if num == 0:
        if current_index > destination:
            return current_index-1
        else:
            return current_index+1
    elif num == 1 or c==1:
        shortcut_dest=get_shortcut_dest(current_index,beta)
        lattice_dest=current_index+int((destination-current_index)/abs(destination-current_index))
        if distance(shortcut_dest,destination)<distance(lattice_dest,destination):
            return shortcut_dest
        else:
            return lattice_dest
    else:
        dest_list=[]
        for _ in range(num):
            dest=get_shortcut_dest(current_index,beta)
            dest_list.append(dest)
        lattice_dest=current_index+int((destination-current_index)/abs(destination-current_index))
        dest_list.append(lattice_dest)
        distances=[distance(node,destination) for node in dest_list]
        return dest_list[np.argmin(distances)]

def select_beta_opt(alpha):
    beta_opt_=np.max([np.min([3-alpha,2]),1])
    beta_opt=betas[np.argmin(abs(np.array(betas)-beta_opt_))]
    return beta_opt

def walk_one_time(alpha,beta,betao):
    gamma = N**alpha
    current_node = 0
    path_length = 0
    while distance(current_node, destination) > 0:
        path_length += 1
        if gamma > 0:
            lambda0 = np.min([gamma/(E_length[beta]*N),gamma/(E_length[betao]*N)])
        next_node = walk(current_node,lambda0,gamma,beta)
        current_node = next_node
    return  path_length

start = 0
times = 50

alpha_beta_T={}

for alpha in alphas:
        alpha_beta_T[alpha]={}
        alpha_beta_T[alpha]['*']=[]
        for beta in betas:
                alpha_beta_T[alpha][beta]=[]
        beta_opt=select_beta_opt(alpha)
        for _ in range(times):
                T_beta=[]
                for beta in betas:
                        a=walk_one_time(alpha,beta,beta_opt)
                        T_beta.append(a)
                        alpha_beta_T[alpha][beta].append(a)
                beta_best=betas[np.argmin(T_beta)]
                if alpha_beta_T[alpha][beta_best][-1] == alpha_beta_T[alpha][beta_opt][-1]:
                        beta_best=beta_opt
                alpha_beta_T[alpha]['*'].append(beta_best)
        mean_T=[]
        for beta in betas:
                mean_T.append(np.mean(np.array(alpha_beta_T[alpha][beta])))
                alpha_beta_T[alpha][beta].append(mean_T[-1])
        alpha_beta_T[alpha]['**']=betas[np.argmin(mean_T)]
        if alpha_beta_T[alpha][alpha_beta_T[alpha]['**']][-1]>alpha_beta_T[alpha][beta_opt][-1]-10:
                alpha_beta_T[alpha]['**']=beta_opt
        print(str(alpha*100/3)+ '% is done, the best beta for '+str(alpha)+' is '+ str(alpha_beta_T[alpha]['**']))
        tf=open('temp_alpha_beta_T_1D.json','w')
        json.dump(alpha_beta_T,tf)
        tf.close()


N=50000
size = [N,N]
d=int(N/4)
betas=np.linspace(1.5,3.5,40)
alphas =np.linspace(0.1,3,30)
c=-1

sum_prob,E_length={},{}

for beta in betas:
        sum_prob[beta]=0
        for i in range(1,2*d):
                sum_prob[beta]+=i**(1-beta)
        E_length[beta]=0
        for i in range(1,2*d):
                E_length[beta]+=i**(2-beta)
        E_length[beta]=E_length[beta]/sum_prob[beta]


def distance(coordinate):
    x,y=coordinate
    return abs(x-d)+abs(y-d)


def transformer(coordinate):
    x,y=coordinate
    if x<0:
        x+=N
    elif x>=N:
        x-=N
    if y<0:
        y+=N
    elif y>=N:
        y-=N
    return (x,y)


def walk_on_lattice(current):
    x,y=current
    dests=[(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
    dest_list=[transformer(ele) for ele in dests]
    distances,choose_nodes=[distance(node) for node in dest_list],[]
    for node in dest_list:
        if distance(node)==min(distances):
            choose_nodes.append(node)
    return random.choice(choose_nodes)


def walk_on_shortcut(current,beta):
    x,y=current
    u0,prob=np.random.uniform(),0
    for l in range(1,2*d):
        prob+=l**(1-beta)
        if prob>=u0*sum_prob[beta]:
            length=l
            break

    u1,u2=random.choice([-1,1]),random.choice([-1,1])
    if length<=d:
        length_list=list(range(length))
    else:
        length_list=list(range(length-d,d))
    dx=random.choice(length_list)
    dy=length-dx
    return transformer((x+u1*dx,y+u2*dy))


def walk(current_node, lambda0,beta):
    num=np.random.poisson(lambda0)
    if num == 0:
        return walk_on_lattice(current_node)
    elif num == 1 or c==1:
        next1=walk_on_shortcut(current_node,beta)
        next2=walk_on_lattice(current_node)
        if distance(next1)>distance(next2):
            return next2
        else:
            return next1
    elif num > 1:
        dest_list = []
        for _ in range(num):
            dest=walk_on_shortcut(current_node,beta)
            dest_list.append(dest)
        distances = [distance(dest_) for dest_ in dest_list]
        next1 = dest_list[np.argmin(distances)]
        next2 = walk_on_lattice(current_node)
        if distance(next1)>distance(next2):
            return next2
        else:
            return next1

def select_beta_opt(alpha):
    beta_opt_=np.max(np.array([np.min([3-alpha,2]),1])+1)
    beta_opt=betas[np.argmin(abs(np.array(betas)-beta_opt_))]
    return beta_opt

def walk_one_time(alpha,beta,betao):
    current_node = [0,0]
    path_length = 0
    lambda0 =np.min([N**alpha / (E_length[beta]*N**2),N**alpha / (E_length[betao]*N**2)])
    while distance(current_node) > 0:
        path_length += 1
        next_node = walk(current_node,lambda0,beta)
        current_node = next_node
    return  path_length

start = 0
times = 50

alpha_beta_T={}

for alpha in alphas[17:]:
        alpha_beta_T[alpha]={}
        alpha_beta_T[alpha]['*']=[]
        for beta in betas:
                alpha_beta_T[alpha][beta]=[]
        beta_opt=select_beta_opt(alpha)
        for _ in range(times):
                T_beta=[]
                for beta in betas:
                        a=walk_one_time(alpha,beta,beta_opt)
                        T_beta.append(a)
                        alpha_beta_T[alpha][beta].append(a)
                beta_best=betas[np.argmin(T_beta)]
                if alpha_beta_T[alpha][beta_best][-1] == alpha_beta_T[alpha][beta_opt][-1]:
                        beta_best=beta_opt
                alpha_beta_T[alpha]['*'].append(beta_best)
        mean_T=[]
        for beta in betas:
                mean_T.append(np.mean(np.array(alpha_beta_T[alpha][beta])))
                alpha_beta_T[alpha][beta].append(mean_T[-1])
        alpha_beta_T[alpha]['**']=betas[np.argmin(mean_T)]
        if alpha_beta_T[alpha][alpha_beta_T[alpha]['**']][-1]>alpha_beta_T[alpha][beta_opt][-1]-2:
                alpha_beta_T[alpha]['**']=beta_opt
        print(str(alpha*100/3)+ '% is done, the best beta for '+str(alpha)+' is '+ str(alpha_beta_T[alpha]['**']))
        tf=open('temp_alpha_beta_T_2Dcopy.json','w')
        json.dump(alpha_beta_T,tf)
        tf.close()

g={}

df1=open('10000_l.json','r')
f1=json.load(df1)
g.update(f1)

df2=open('10000_m.json','r')
f2=json.load(df2)
g.update(f2)

df3=open('10000_g.json','r')
f3=json.load(df3)
g.update(f3)

k=np.linspace(1.1,4,30)
v=list(g.values())
g={("%.1f"%k[i]):v[i] for i in range(len(k))}

g.keys()
f1['1.5'][list(f1['1.5'].keys())[10]]

wf=open('beta_vs_alpha_2D.json','w')
f={}

df1=open('temp_alpha_beta_T_2D.json','r')
f1=json.load(df1)
f.update(f1)

df2=open('temp_alpha_beta_T_2Dcopy.json','r')
f2=json.load(df2)
f.update(f2)

k=np.linspace(0.1,3,30)
v=list(f.values())
f={("%.1f"%k[i]):v[i] for i in range(len(k))}

f_=json.dumps(f)
wf.write(f_)
wf.close()

def plot_theory(alpha,xmin=0,xmax=61):
    x=np.linspace(0,3,61)
    y=np.zeros(61)
    beta=3-alpha
    for i in range(61):
        if x[i]<1:
            y[i]=1-(alpha-1)/(2-x[i])
        if x[i]>=1 and x[i]<beta:
            y[i]=1-(alpha-1)
        if x[i]>=beta:
            y[i]=beta*(1-1/x[i])
    plt.plot(x[xmin:xmax],y[xmin:xmax],label='理论值')

alpha='2.0'
k=list((g[alpha].keys()))[1:-1]
v=np.array([g[alpha][i][-1] for i in k])
k=[float(i) for i in k]
plt.scatter(k,v/4850,label='模拟值,n=10000',color='darkblue',alpha=0.5,marker='s')

kk=list((gg[alpha].keys()))[1:-1]
vv=np.array([gg[alpha][i][-1] for i in kk])
kk=[float(i) for i in kk]
plt.scatter(kk,vv/25000,label='模拟值,n=50000',color='darkblue')


plt.plot([3,3],[0.65,max((np.log(v)/np.log(5000))[:30])],color='grey',linestyle='--')
plt.xlabel(r'$\beta$',labelpad=-12,x=1)
plt.ylabel(r'$\log T/\log n$',rotation=0,labelpad=-12,y=1.02)
plt.title('二维网络能量参数'+r'$\alpha$'+'='+alpha+'时步长T与生成参数'+r'$\beta$'+'的关系')

rst=[]
betas=np.linspace(0.1,3,100)
for beta in betas:
    if beta<=2:
        if beta!=1:
            def func(x):
                return 1/((np.power(2,beta-1)-1)/(2*(beta-1))*np.power(x,2-beta)+1)
            a,_=integrate.quad(func,0,1)
            rst.append(a)
        else:
            def func(x):
                return 1/(np.power(x,2-beta)+1)
            a,_=integrate.quad(func,0,1)
            rst.append(a)
    else:
        def func(x):
            return 1/(np.exp((2-beta)/(beta-1))/(2*(beta-1))+1)
        a,_=integrate.quad(func,0,1)
        rst.append(a)

plt.plot(np.linspace(1.1,4,100),rst)
plt.legend(loc=3)
plt.show()

=[float(i) for i in g.keys()]
v=np.array([g[i][str(g[i]['**'])][-1] for i in g.keys()])/5000
plt.scatter(k,v,label='模拟值,n='+r'$10^4$')

plt.scatter(np.linspace(0,1,11),np.ones(11),color='#1f77b4')

alphas=np.linspace(-1,3,100)
n=2500
rst=[]
for alpha in alphas:
    if alpha<1:
        def func(x):
            return 1
    if alpha>2:
        def func(x):
            return 1/(np.power(n,1)*np.power(x,1)+1)
    else:
        def func(x):
            return 1/(np.power(n,alpha-1)*np.power(x,alpha-1)*(np.power(2,2-alpha)-1)/(2*(2-alpha))+1)
    a,_=integrate.quad(func,0,1)
    rst.append(a)
plt.plot(alphas+1,rst,label='理论值,n='+r'$10^4$',color='darkblue')

alphas=np.linspace(-1,3,100)
n=25000
rst=[]
for alpha in alphas:
    if alpha<1:
        def func(x):
            return 1
    if alpha>2:
        def func(x):
            return 1/(np.power(n,1)*np.power(x,1)+1)
    else:
        def func(x):
            return 1/(np.power(n,alpha-1)*np.power(x,alpha-1)*(np.power(2,2-alpha)-1)/(2*(2-alpha))+1)
    a,_=integrate.quad(func,0,1)
    rst.append(a)
plt.plot(alphas+1,rst,label='理论值,n='+r'$10^5$',color='darkblue',alpha=0.75)

alphas=np.linspace(-1,3,100)
n=250000
rst=[]
for alpha in alphas:
    if alpha<1:
        def func(x):
            return 1
    if alpha>2:
        def func(x):
            return 1/(np.power(n,1)*np.power(x,1)+1)
    else:
        def func(x):
            return 1/(np.power(n,alpha-1)*np.power(x,alpha-1)*(np.power(2,2-alpha)-1)/(2*(2-alpha))+1)
    a,_=integrate.quad(func,0,1)
    rst.append(a)
plt.plot(alphas+1,rst,label='理论值,n='+r'$10^6$',color='darkblue',alpha=0.5)

alphas=np.linspace(-1,3,100)
n=2500000
rst=[]
for alpha in alphas:
    if alpha<1:
        def func(x):
            return 1
    if alpha>2:
        def func(x):
            return 1/(np.power(n,1)*np.power(x,1)+1)
    else:
        def func(x):
            return 1/(np.power(n,alpha-1)*np.power(x,alpha-1)*(np.power(2,2-alpha)-1)/(2*(2-alpha))+1)
    a,_=integrate.quad(func,0,1)
    rst.append(a)
plt.plot(alphas+1,rst,label='理论值,n='+r'$10^7$',color='darkblue',alpha=0.25)


plt.plot([0,2,2,3],[1,1,0,0],label='极限值',color='grey',linestyle='--')
plt.legend()

plt.xlabel(r'$\alpha$',labelpad=-12,x=1)
plt.ylabel(r'$T/n$',rotation=0,labelpad=-12,y=1.02)
plt.title('二维网络导航步长T与能量参数'+r'$\alpha$'+'的关系及极限相变')
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 08:52:49 2022

@author: Zao
"""

from a_mps import MPS
import a_mps
from b_model import TFIModel
from b2_model import TFIM2
import c_tebd
import numpy as np
import matplotlib.pyplot as plt

sx = np.array([[0,1],[1,0]])
sz =np.array([[1,0],[0,-1]])
L=14
m = a_mps.init_spinup_MPS(L)

'''
def q_ab(m, op):
    print(m.site_expectation_value(op))
    return

def init_spinright_MPS(L):
    B = np.zeros([1, 2, 1], np.float)
    B[0, 0, 0] = 1./np.sqrt(2.)
    B[0, 1, 0] = 1./np.sqrt(2.)
    S = np.ones([1], np.float)
    Bs = [B.copy() for i in range(L)]
    Ss = [S.copy() for i in range(L)]
    return MPS(Bs, Ss)

m1 = init_spinright_MPS(L)


print('for spin up initial MPS')
print('the site expectation values are')
print('op=sz')
q_ab(m, sz)
print('op=sx')
q_ab(m, sx)
print('for spin right initial MPS')
print('the site expectation values are')
print('op=sz')
q_ab(m1, sz)
print('op=sx')
q_ab(m1, sx)


J=1
for g in [0.5,1.,1.5]:
    ising = TFIModel(L, J, g)
    print('when g={}'.format(g))
    print('spinup energy is {}'.format(ising.energy(m)))
    print('spinright energy is {}'.format(ising.energy(m1)))
'''

def correlation(op_x, op_y, psi, i, j):
    theta_i = psi.get_theta1(i) # vL, i, vR
    op_theta_i = np.tensordot(op_x, theta_i, axes=[1, 1])   # i, i*; vL, i, vR -> i, vL, vR
    result = np.tensordot(theta_i.conj(), op_theta_i, [[0,1],[1,0]])   # vL*, i*, vR*; i, vL, vR ->vR*, vR
    for index in range(i+1,j):
        bj = psi.Bs[index]  # vL, i, vR
        cell = np.tensordot(bj.conj(), bj, [[1],[1]])   
        result = np.tensordot(result, cell, [[0,1],[0,2]])  # vR*, vR
    Bj = psi.Bs[j]  #vL, i, vR
    op_Bj = np.tensordot(op_y, Bj, axes=[1, 1]) #i, vL, vR
    final_cell = np.tensordot(Bj.conj(), op_Bj, [[1,2],[0,2]])  # vL*, vL
    #print(result.shape, final_cell.shape)
    result = np.tensordot(result, final_cell, [[0,1],[0,1]])
    #print(result)
    return result


def plot_cor(i, g, L):
    j_list = np.arange(i+1, L) - i
    _, psi, _=c_tebd.example_TEBD_gs_finite(L, 1, g)
    site_exp = psi.site_expectation_value(sx)
    result = []
    c_j = []
    for j in j_list:
        result.append(correlation(sx, sx, psi, i, i+j))
        c_j.append(result-site_exp[i]*site_exp[i+j])
    plt.plot(j_list, result, label='g={}'.format(g))
    return result, c_j


g_list = [0.3,0.5,0.8,0.9,1.,1.1,1.2,1.5]
L = 30
phase_transition = {}
c_j = {}
for g in g_list:
    p,c=plot_cor(int(L/4), g, L)
    phase_transition[g] = p
    c_j[g] = c
plt.legend()
plt.title('Correlation v.s. dist')
plt.xlabel('distance')
plt.ylabel('correlation')
plt.savefig('correlation.png')
plt.clf()

phase = []
for g in g_list:
    phase.append(phase_transition[g][int(L/2)])
plt.plot(g_list, phase)
plt.xlabel('g')
plt.ylabel('magnetization sq')
plt.title('phase_transistion')
plt.savefig('phase.png')
plt.clf()

for g in g_list:
    x = np.arange(len(c_j[g]))+1
    y = np.log(c_j[g][-1])
    z = np.polyfit(x, y, 1)
    plt.plot(x, y, label='g={}'.format(g))
    print('when g={0},possible zeta is {1}'.format(g,z[0]))
plt.legend()
plt.xlabel('distance')
plt.ylabel('log of correlation')
plt.title('log_corr v.s. dist')
plt.savefig('log_corr_dist.png')
plt.clf()
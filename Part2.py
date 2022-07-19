# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 15:41:46 2022

@author: Zao
"""

from a_mps import MPS
import a_mps
from b_model import TFIModel
from b2_model import TFIM2
import c_tebd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import fourier_transform as ft

id = np.array([[1,0],[0,1]])
sz = np.array([[1,0],[0,-1]])
sx = np.array([[0,1],[1,0]])
sy = np.array([[0,0-1j], [0+1j, 0]])

def calc_U_bonds_realtime(model, dt):
    """Given a model, calculate ``U_bonds[i] = expm(-dt*model.H_bonds[i])``.

    Each local operator has legs (i out, (i+1) out, i in, (i+1) in), in short ``i j i* j*``.
    Note that no imaginary 'i' is included, thus real `dt` means imaginary time evolution!
    """
    H_bonds = model.H_bonds
    d = H_bonds[0].shape[0]
    U_bonds = []
    for H in H_bonds:
        H = np.reshape(H, [d * d, d * d])
        U = expm(-dt * H * 1j)
        U_bonds.append(np.reshape(U, [d, d, d, d]))
    return U_bonds
    
def get_Lp_Rp(psi, psi_0):
    L = psi.L
    Lp = []
    Rp = []
    Lp.append(np.tensordot(psi_0.get_theta1(0).conj(), psi.get_theta1(0), [[0,1],[0,1]])) #vR*, vR
    for i in range(1,L-1):
        cell = np.tensordot(psi_0.Bs[i].conj(), psi.Bs[i], [[1],[1]]) #vL*, vR*, vL, vR
        Lp.append(np.tensordot(Lp[-1], cell, [[0,1],[0,2]]))
    Rp.append(np.tensordot(psi_0.Bs[-1].conj(), psi.Bs[-1], [[1,2],[1,2]]))    #vL*, vL
    for i in range(2,L):
        cell = np.tensordot(psi_0.Bs[L-i].conj(), psi.Bs[L-i], [[1],[1]]) #vL*, vR*, vL, vR
        Rp.insert(0, np.tensordot(cell, Rp[0], [[1,3],[0,1]]))  #vL*, vL
    return Lp, Rp

def get_correlation(psi_origin, L, psi, chi_max, eps):
    cor_list = []
    Lp, Rp = get_Lp_Rp(psi, psi_origin)
    
    theta_j0 = np.tensordot(sy, psi.get_theta1(0), [[1],[1]]) # i, vL, vR
    first_cell = np.tensordot(psi_origin.get_theta1(0).conj(), theta_j0, [[0,1], [1,0]]) #vR*, vR
    cor_list.append(np.tensordot(first_cell, Rp[0], [[0,1],[0,1]]))
    
    for j in range(1, L-1):
        Bj =  np.tensordot(sy, psi.Bs[j], [[1],[1]]) # i, vL, vR
        cell_j = np.tensordot(psi_origin.Bs[j].conj(), Bj, [[1], [0]]) #vL*, vR*, vL, vR
        cell_mit_r = np.tensordot(cell_j, Rp[j], [[1,3], [0,1]])    #vL*, vL
        cor_list.append(np.tensordot(cell_mit_r, Lp[j-1], [[0,1],[0,1]]))
        
    final_bj = np.tensordot(sy, psi.Bs[-1], [[1],[1]]) # i, vL, vR
    final_cell = np.tensordot(psi_origin.Bs[-1].conj(), final_bj, [[1,2],[0,2]]) #vL*, vL
    cor_list.append(np.tensordot(Lp[-1], final_cell, [[0,1],[0,1]]))
    return np.array(cor_list)

def TEBD_realtime(psi, model, t, chi_max, eps, dt, psi_origin, E):
    U_bonds = calc_U_bonds_realtime(model, dt)
    steps = int(t/dt)
    ent_list = []
    cor_list = []
    for s in range(steps):
        '''
        for k in [0,1]:
            for i in range(k, N, 2):
                c_tebd.update_bond(psi, i, U_bonds[i], chi_max, eps)
                '''
        c_tebd.run_TEBD(psi, U_bonds, 1, chi_max, eps)
        ent_list.append(psi.entanglement_entropy())
        cor_list.append(get_correlation(psi_origin, psi.L, psi, chi_max, eps)*np.exp(E*s*dt*1j))
    return ent_list, cor_list

def time_evolution_correlation(L, g, h, t, dt):
    chi_max=100
    eps=1.e-10
    E, psi, model =c_tebd.example_TEBD_gs_finite_2(L, 1, g, h)
    psi_origin = psi.copy()
    psi.Bs[int(L/2)] = np.transpose(np.tensordot(sy, psi.Bs[int(L/2)], [[1],[1]]), (1,0,2))# vL, i, vR
    ent_list, cor_list = TEBD_realtime(psi, model, t, chi_max, eps, dt, psi_origin, E)
    return np.array(ent_list), np.array(cor_list)


def get_dsf_manul(cor_arr, dt, L):
    k_list = np.arange(-3.1, 3.1, 0.2)
    w_list = np.arange(-1, 10.5, 0.2)
    xi = int(L/2)
    sigma = 0.4
    N=cor_arr.shape[0]
    skw = []
    for w in w_list:
        sw = []
        for k in k_list:
            s = 0
            for j in range(L):
                for tn in range(N):
                    s+=np.exp(1j*(w*dt*tn-k*(j-xi)))*cor_arr[tn, j]*np.exp(-0.5*(tn/sigma/N)**2)
            sw.append(s)
        skw.append(sw)
    return skw, k_list, w_list

                
def get_dsf(cor_arr, dt, L):
    xi = int(L/2)
    cor = np.zeros(cor_arr.shape)
    cor[:,L-xi:] =cor_arr[:,:xi]
    cor[:,:L-xi] =cor_arr[:,xi:]
    Ctk = []
    for tn in range(cor_arr.shape[0]):
        data_space = cor[tn,:]
        _, Ck = ft.fourier_space(data_space)
        Ctk.append(Ck)
    Ctk = np.array(Ctk)
    print(Ctk.shape)
    Skw = np.zeros(Ctk.shape)
    freq = []
    for k in range(Ctk.shape[1]):
        data_space = Ctk[:,k]
        w, Shifted = ft.fourier_time(data_space, dt)
        Skw[:, k] = Shifted
        freq.append(w)
    return Skw, freq

from scipy import special

def airy(Swk, freqs, L, g, h):
    n_k = Swk.shape[1]
    start = np.where(freqs>3.5)[0][0]
    end = np.where(freqs<4.5)[0][-1]
    data = np.real(Swk[start:end+1, n_k//2])
    max1 = start + np.argmax(data[:len(data)//2])
    max2 = start + len(data)//2 + np.argmax(data[len(data)//2:])

    ai_neg_zeros = -special.ai_zeros(6)[0]
    z1 = ai_neg_zeros[0]
    z2 = ai_neg_zeros[1]
    m1 = freqs[max1]
    m2 = freqs[max2]

    c = (m1-m2)/(z1-z2)
    m0 = (m1 - c*z1)/2

    masses = 2*m0 + c*ai_neg_zeros

    plt.figure()
    #n_k = Swk.shape[1]
    plt.plot(freqs, np.real(Swk[:,n_k//2]))
    plt.vlines(masses, ymin=0, ymax=500, color='red')
    plt.xlabel(r'$\omega$')
    plt.ylabel('intensity')
    plt.xlim(0,6)
    plt.title('Bound states at k=0')
    plt.savefig('airy_L={}_g={}_h={}.png'.format(L,g,h))
    plt.clf()

L_list = [81]
g_list = [0.2]
h=0.1
t=140
dt=0.01
for L in L_list:
    for g in g_list:
        ent_arr, cor_arr = time_evolution_correlation(L, g, h, t, dt)
        plt.pcolor(np.real(cor_arr))
        plt.xlabel('Site index')
        plt.ylabel('tn')
        plt.title('Corr_L={}_g={}_h={}'.format(L,g,h))
        plt.savefig('Corr_L={}_g={}_h={}.png'.format(L,g,h))
        plt.clf()
        plt.pcolor(ent_arr)
        plt.xlabel('Site index')
        plt.ylabel('tn')
        plt.title('ent_L={}_g={}_h={}'.format(L,g,h))
        plt.savefig('ent_L={}_g={}_h={}.png'.format(L,g,h))
        plt.clf()
        s_kw, k, w = get_dsf_manul(cor_arr, dt, L)
        s_kw = np.array(s_kw)
        k = np.array(k)
        w=np.array(w)
        plt.pcolor(k,w,(np.real(s_kw)))
        plt.xlabel('k')
        plt.ylabel('\u03C9')
        plt.title('DSF_L={}_g={}_h={}'.format(L,g,h))
        plt.savefig('DSF_L={}_g={}_h={}.png'.format(L,g,h))
        plt.clf()
        airy(s_kw, w, L, g, h)

    
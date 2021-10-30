#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 16:30:29 2018

@author: Saehong Park
"""
th=1e-6
import numpy as np
from scipy import linalg, matrix
import scipy.io as sio
from scipy.interpolate import CubicSpline #import interp1d
from scipy.interpolate import interpolate
from scipy import integrate
import scipy.sparse as sps
import ipdb
import pdb
from sys import path
from casadi import *
from gym_dfn.envs.ParamFile_LCO2 import *

import casadi as ca


def dae_dfn_casadi_pade(x, z, u, p):

  #==============================================================================
  # Parse out states
  #==============================================================================

  Ncsn = p['PadeOrder'] * (p['Nxn']-1)
  Ncsp = p['PadeOrder'] * (p['Nxp']-1)

  Nce = p['Nxn'] - 1 + p['Nxs'] -1 + p['Nxp'] - 1 # p.Nx - 3
  Nx = Nce # p.Nx - 3
  Nc = Ncsn + Ncsp + Nce

  Nn = p['Nxn']-1
  Ns = p['Nxs']-1
  Np = p['Nxp']-1
  Nnp = Nn + Np

  # Solid concentraion (c_s_n)
  c_s_n = x[range(Ncsn)]
  c_s_n=if_else(c_s_n<=(1-th)*p['c_s_n_max'],c_s_n,(1-th)*p['c_s_n_max'])

  c_s_p = x[range(Ncsn,Ncsn+Ncsp)]
  c_s_p=if_else(c_s_p<=(1-th)*p['c_s_p_max'],c_s_p,(1-th)*p['c_s_p_max'])
  

  c_s_n_mat = ca.reshape(c_s_n,(p['PadeOrder'],p['Nxn']-1)) # --- Pade
  c_s_p_mat = ca.reshape(c_s_p,(p['PadeOrder'],p['Nxn']-1))

  # Electrolyte concentration
  c_e = x[range(Ncsn+Ncsp,Nc)]

  # Temperature [given]
  T1 = x[-1]

  # Solid potential
  phi_s_n = z[range(Nn)]
  phi_s_p = z[range(Nn,Nnp)]

  # Electrolyte current
  i_en = z[range(Nnp,Nnp+Nn)]
  i_ep = z[range(Nnp+Nn, 2*Nnp)]

  # Electrolyte potential
  phi_e = z[range(2*Nnp, 2*Nnp + Nx +2)]

  # Molar flux
  jn = z[range(2*Nnp + Nx +2,  2*Nnp + Nx +2 + Nn)]
  jp = z[range(2*Nnp + Nx +2 + Nn, 2*Nnp + Nx +2 + Nn + Np)]

  # Input
  Cur = u

  # Temperature-dependent parameters
  p['D_s_n'] = p['D_s_n0'] * np.exp(p['E_Dsn'] / p['R'] * (1/p['T_amb'] - 1/T1))
  p['D_s_p'] = p['D_s_p0'] * np.exp(p['E_Dsp'] / p['R'] * (1/p['T_amb'] - 1/T1))
  p['k_n'] = p['k_n0'] * np.exp(p['E_kn'] / p['R'] * (1/p['T_amb'] - 1/T1))
  p['k_p'] = p['k_p0'] * np.exp(p['E_kp'] / p['R'] * (1/p['T_amb'] - 1/T1))

  #==============================================================================
  # Li diffusion in solid phase: c_s(x,r,t) -- Pade approx 3
  #==============================================================================

  A_csn, B_csn, A_csp, B_csp, C_csn, C_csp = c_s_mats(p)

  # Anode LTI system
  c_s_n_dot_mat = mtimes(A_csn,c_s_n_mat) + mtimes(B_csn,jn.T)
  y_csn = mtimes(C_csn,c_s_n_mat)

  # Anode parse LTI outputs
  c_ss_n_mat = y_csn[0,:].T
  c_ss_n_mat=if_else(c_ss_n_mat<=(1-th)*p['c_s_n_max'],c_ss_n_mat,(1-th)*p['c_s_n_max'])

  #pdb.set_trace()
  c_avg_n = y_csn[1,:].T
  c_s_n_dot = ca.reshape(c_s_n_dot_mat, (Ncsn,1))

  # Cathode LTI system
  c_s_p_dot_mat = mtimes(A_csp,c_s_p_mat) + mtimes(B_csp,jp.T)
  y_csp = mtimes(C_csp,c_s_p_mat)

  # Cathode parse LTI outputs
  c_ss_p_mat = y_csp[0,:].T
  c_ss_p_mat=if_else(c_ss_p_mat<=(1-th)*p['c_s_p_max'],c_ss_p_mat,(1-th)*p['c_s_p_max'])

  c_avg_p = y_csp[1,:].T
  c_s_p_dot = ca.reshape(c_s_p_dot_mat, (Ncsp,1) )


  #==============================================================================
  # Li diffusion in electrolyte phase: c_e(x,t)
  #==============================================================================
  arg = 1

  # Call system matrices
  M1n, M2n, M3n, M4n, M5n, M1s, M2s, M3s, M4s, M1p, M2p, M3p, M4p, M5p, C_ce = c_e_mats(p)

  # compute boundary conditions
  c_e_bcs = mtimes(C_ce, c_e) #CASADI change

  # c_en, c_es, c_ep # i.e., Nxn=10, Nxs=5, Nxp=10
  c_en = c_e[range(p['Nxn']-1)] # range(9) = 0,1,...,8
  c_es = c_e[range(p['Nxn']-1 , p['Nxn']-1 + p['Nxs']-1)] # range(9,13) = 9,10,...,12
  c_ep = c_e[range(p['Nxn']-1 + p['Nxs']-1 , p['Nxn']-1 + p['Nxs']-1 + p['Nxp']-1)] # range(13, 22) = 13,...,21
  c_ex = vertcat(c_e_bcs[0],c_en,c_e_bcs[1],c_es,c_e_bcs[2],c_ep,c_e_bcs[3]) # concatenate with boundary conditions.
  ce0n = c_e_bcs[0]
  cens = c_e_bcs[1]
  cesp = c_e_bcs[2]
  ce0p = c_e_bcs[3]

  D_en,dD_en = electrolyteDe(c_en, T1, arg)
  D_es,dD_es = electrolyteDe(c_es, T1, arg)
  D_ep,dD_ep = electrolyteDe(c_ep, T1, arg)

  # ADD BRUGGEMAN RELATION % Apr.22 2016 by Saehong Park
  D_en_eff = D_en * p['epsilon_e_n']**(p['brug']-1)
  dD_en_eff = dD_en * p['epsilon_e_n']**(p['brug']-1);

  # DO same s,p
  D_es_eff = D_es * p['epsilon_e_s']**(p['brug']-1);
  dD_es_eff = dD_es * p['epsilon_e_s']**(p['brug']-1);

  D_ep_eff = D_ep * p['epsilon_e_p']**(p['brug']-1);
  dD_ep_eff = dD_ep * p['epsilon_e_p']**(p['brug']-1);


  # Compute derivative
  c_en_dot = dD_en_eff * (mtimes(M1n,c_en) + mtimes(M2n,c_e_bcs[range(0,2)]))**2 \
      + D_en_eff * (mtimes(M3n,c_en) + mtimes(M4n,c_e_bcs[range(0,2)])) \
      + mtimes(M5n,jn)

  c_es_dot = dD_es_eff * (mtimes(M1s,c_es) + mtimes(M2s,c_e_bcs[range(1,3)]))**2 \
      + D_es_eff * (mtimes(M3s,c_es) + mtimes(M4s,c_e_bcs[range(1,3)]))

  c_ep_dot = dD_ep_eff * (mtimes(M1p,c_ep) + mtimes(M2p,c_e_bcs[range(2,4)]))**2 \
      + D_ep_eff * (mtimes(M3p,c_ep) + mtimes(M4p,c_e_bcs[range(2,4)])) \
      + mtimes(M5p,jp)

  # Assemble c_e_dot
  c_e_dot = vertcat(c_en_dot, c_es_dot, c_ep_dot)


  #==============================================================================
  # Potential in solid phase: phi_s(x,t)
  #==============================================================================
  # semi-explicit form 0=g()

  # Call system matrices
  F1_psn, F1_psp, F2_psn, F2_psp, G_psn, G_psp, C_psn, C_psp, D_psn, D_psp = phi_s_mats(p)

  i_enn = vertcat(0, i_en, Cur)
  i_epp = vertcat(Cur, i_ep, 0)

  phi_sn_dot = mtimes(F1_psn,phi_s_n) + mtimes(F2_psn,i_enn) + mtimes(G_psn,Cur)
  phi_sp_dot = mtimes(F1_psp,phi_s_p) + mtimes(F2_psp,i_epp) + mtimes(G_psp,Cur)

  # Terminal voltage
  phi_s_n_bcs = mtimes(C_psn,phi_s_n) + mtimes(D_psn,Cur)
  phi_s_p_bcs = mtimes(C_psp,phi_s_p) + mtimes(D_psp,Cur)

  Volt = phi_s_p_bcs[1] - phi_s_n_bcs[0]

  #==============================================================================
  # Electrolyte current: i_e(x,t)
  #==============================================================================
  # semi-explicit form 0=g()

  F1_ien, F1_iep, F2_ien, F2_iep, F3_ien, F3_iep = i_e_mats(p)

  # Electrolyte Current: i_e(x,t)
  i_en_dot = mtimes(F1_ien,i_en) + mtimes(F2_ien,jn) + mtimes(F3_ien,Cur)
  i_ep_dot = mtimes(F1_iep,i_ep) + mtimes(F2_iep,jp) + mtimes(F3_iep,Cur)

  # Electrolyte current across all three regions
  i_e_in = vertcat(i_en, Cur*np.ones((p['Nxs']+1,1)), i_ep)


  #==============================================================================
  # Potential in electrolyte phase: phi_e(x,t)
  #==============================================================================
  # semi-explicit form 0=g()

  # Electrolyte conductivity
  kappa_ref = electrolyteCond(c_ex, arg)

  # Adjustment for Arrhenius
  kappa = kappa_ref * np.exp(p['E_kappa_e']/p['R']*(1/p['T_amb'] - 1/T1))

  kappa0 = kappa[0]                   # BC1
  kappa_n = kappa[range(1,p['Nxn'])]
  kappa_ns = kappa[p['Nxn']]
  kappa_s = kappa[p['Nxn']+1 : p['Nxn']+2 + p['Nxs']-2]
  kappa_sp = kappa[p['Nxn']+2+p['Nxs']-2]
  kappa_p = kappa[p['Nxn']+2+p['Nxs']-1 : -1]
  kappaN = kappa[-1]                  # BC2

  # Effective conductivity
  kappa_eff0 = kappa0 * p['epsilon_e_n'] **(p['brug'])
  kappa_eff_n = kappa_n * p['epsilon_e_n'] **(p['brug'])
  kappa_eff_ns = kappa_ns * ((p['epsilon_e_n']+p['epsilon_e_s'])/2) ** (p['brug'])
  kappa_eff_s = kappa_s * p['epsilon_e_s'] **(p['brug'])
  kappa_eff_sp = kappa_sp * ((p['epsilon_e_s']+p['epsilon_e_p'])/2) **(p['brug'])
  kappa_eff_p = kappa_p * p['epsilon_e_p'] **(p['brug'])
  kappa_effN = kappaN * p['epsilon_e_p'] **(p['brug'])


  # Form into vector
  kappa_eff = vertcat(kappa_eff_n,kappa_eff_ns,kappa_eff_s,kappa_eff_sp,kappa_eff_p)

  Kap_eff = ca.diag(kappa_eff)

  # Diffusional conductivity - electrolyteAct

  dactivity = electrolyteAct(c_ex,T1,p)
  dActivity0 = dactivity[0]
  dActivity_n = dactivity[1:p['Nxn']]
  dActivity_ns = dactivity[p['Nxn']]
  dActivity_s = dactivity[p['Nxn']+1 : p['Nxn']+2 + p['Nxs']-2]
  dActivity_sp = dactivity[p['Nxn']+2 + p['Nxs']-2]
  dActivity_p = dactivity[p['Nxn']+2 + p['Nxs']-1 : -1]
  dActivityN = dactivity[-1]

  dActivity = vertcat(dActivity_n,dActivity_ns,dActivity_s,dActivity_sp,dActivity_p)
  bet = (2*p['R']*T1) / (p['Faraday']) * (p['t_plus']-1) * (1+ dActivity)

  bet_mat = ca.diag(bet)

  # Modified effective conductivity
  Kap_eff_D = mtimes(bet_mat,Kap_eff)

  # Call system matrices
  M1_pe, M2_pe, M3_pe, M4_pe, C_pe = phi_e_mats(p)

  M2_pe[0,0] = M2_pe[0,0] * kappa_eff0
  M2_pe[-1,-1] = M2_pe[-1,-1] * kappa_effN

  F1_pe = mtimes(Kap_eff,M1_pe) + mtimes(M2_pe,C_pe)

  F2_pe = M3_pe
  F3_pe = mtimes(Kap_eff_D,M4_pe)


  # Algebraic eqns
  phi_e_dot = mtimes(F1_pe,phi_e) + mtimes(F2_pe,i_e_in) + mtimes(F3_pe,np.log(c_ex))


  #==============================================================================
  # Butler-Volmer equations: jn_dot, jp_dot
  #==============================================================================
  # semi-explicit form 0=g()

  aFRT = (p['alph'] * p['Faraday']) / (p['R'] * T1)

  # Exchange current density : i_0^{\pm}
  i_0n, i_0p = exch_cur_dens(p,c_ss_n_mat,c_ss_p_mat,c_e)

  # Equilibrium potential : U^{\pm}(c_ss_mat)
  theta_n = c_ss_n_mat / p['c_s_n_max']
  theta_p = c_ss_p_mat / p['c_s_p_max']
  Unref = refPotentialAnode_casadi(theta_n)
  Upref = refPotentialCathode_casadi(theta_p)

  # Overpotential : \eta
  eta_n = phi_s_n - phi_e[0:Nn] - Unref - p['Faraday']*p['R_f_n']*jn
  eta_p = phi_s_p - phi_e[Nn+Ns+2 : ] - Upref - p['Faraday']*p['R_f_p']*jp


  # Algebraic eqns (semi-explict form)
  jn_dot = 2/p['Faraday'] * i_0n * sinh(aFRT * eta_n) - jn
  jp_dot = 2/p['Faraday'] * i_0p * sinh(aFRT * eta_p) - jp

  # check1 = eta_p
  #==============================================================================
  # Temperature: T_dot
  #==============================================================================
  ## Thermal dynamics [1-state]

  SOC_n_avg = sum1(c_avg_n)/c_avg_n.shape[0]/p['c_s_n_max']
  SOC_p_avg = sum1(c_avg_p)/c_avg_p.shape[0]/p['c_s_p_max']

  avg_Unb = refPotentialAnode_casadi(SOC_n_avg)
  avg_Upb = refPotentialCathode_casadi(SOC_p_avg)

  Q_dot = fabs(Cur*(Volt-(avg_Upb-avg_Unb))) # fabs: absolute
  T1_dot = 1/(p['mth']*p['C_p']) * (Q_dot - 1/p['R_th']*(T1 - p['T_amb']))
    


  #==============================================================================
  # Conservation of Li-ion matters
  #==============================================================================
  # Equilibrium potential and gradient w.r.t bulk concentration

  n_Li_s = sum1(c_avg_n) * p['L_n'] * p['delta_x_n'] * p['epsilon_s_n'] * p['Area'] \
          + sum1(c_avg_p) * p['L_p'] * p['delta_x_p'] * p['epsilon_s_p'] * p['Area']

  n_Li_e = sum1(c_e[0:Nn]) * p['L_n'] * p['delta_x_n'] * p['epsilon_e_n'] * p['Area'] \
          + sum1(c_e[Nn:Nn+Ns]) * p['L_s'] * p['delta_x_s'] * p['epsilon_e_s'] * p['Area'] \
          + sum1(c_e[Nn+Ns:]) * p['L_p'] * p['delta_x_p'] * p['epsilon_e_p'] * p['Area']

  c_e0n = c_ex[0]
  c_e0p = c_ex[-1]
  eta_s_Ln = phi_s_n_bcs[1] - phi_e[Nn+1]
  
  check1 = vertcat(phi_s_n-phi_e[0:Nn],eta_s_Ln)

  #==============================================================================
  # RETURN
  #==============================================================================

  f = vertcat(c_s_n_dot,c_s_p_dot,c_e_dot, T1_dot)
  g = vertcat(phi_sn_dot,phi_sp_dot,i_en_dot,i_ep_dot,phi_e_dot,jn_dot,jp_dot)
  L = Volt

  f_out = vertcat(c_s_n,c_s_p,T1)
  g_out = vertcat(phi_s_n,phi_s_p,i_en,i_ep,phi_e,jn,jp)
  alg_out = vertcat(c_ss_n_mat,c_ss_p_mat,c_ex, c_avg_n, c_avg_p, eta_n, eta_p, c_e0n, c_e0p, eta_s_Ln, Volt, n_Li_s, n_Li_e, i_0n, i_0p)
  param_out = vertcat(D_en, dD_en, D_es, dD_es, D_ep, dD_ep)

  debug_out = vertcat(check1)

  return f, g, L, f_out, g_out, alg_out, param_out, debug_out





def c_s_mats(p):
  # Pade approximations
  N = 3

  b_vecn = symsubsnum(p['R_s_n'],p['D_s_n'])
  a_vecn = symsubsden(p['R_s_n'],p['D_s_n'])

  b_vecp = symsubsnum(p['R_s_p'],p['D_s_p'])
  a_vecp = symsubsden(p['R_s_p'],p['D_s_p'])

  bp_vecn = b_vecn / a_vecn[-1]
  ap_vecn = a_vecn / a_vecn[-1]
  bp_vecp = b_vecp / a_vecp[-1]
  ap_vecp = a_vecp / a_vecp[-1]

  # Create state-space representation
  An_casadi = ca.diag(SX.zeros(N)) # triangular form for casadi
  Ap_casadi = ca.diag(SX.zeros(N)) # triangular form for casadi

  An = np.diag(np.ones(N-1),k=1)
  Ap = np.diag(np.ones(N-1),k=1)

  An_casadi = An_casadi + An # triangular form for casadi
  Ap_casadi = Ap_casadi + Ap # triangular form for casadi

  for idx in range(N):
    An_casadi[N-1,idx] = -ap_vecn[idx]
    Ap_casadi[N-1,idx] = -ap_vecp[idx]

  An = An_casadi # SX matrix
  Ap = Ap_casadi # SX matrix

  Bn = np.zeros((N,1))
  Bp = np.zeros((N,1))
  Bn[-1] = 1
  Bp[-1] = 1

  C1n = bp_vecn.transpose()
  C2n = b_vecn[0] * ap_vecn[1:].transpose()
  Cn = np.concatenate((C1n,C2n),axis=0)

  C1p = bp_vecp.transpose()
  C2p = b_vecp[0] * ap_vecp[1:].transpose()
  Cp = np.concatenate((C1p,C2p),axis=0)

  ## Convert to Jordan form
  Vn = np.array([[-((p['R_s_n']**(4))*(9*np.sqrt(2429)-457))/(381150*(p['D_s_n']**(2))), ((p['R_s_n']**4)*(9*np.sqrt(2429)+457))/(381150*p['D_s_n']**2), 1],\
                [(p['R_s_n']**(2)) * (np.sqrt(2429)-63) / (2310*p['D_s_n']), -(p['R_s_n']**(2)) * (np.sqrt(2429)+63)/(2310*p['D_s_n']), 0],\
                [1, 1, 0]])

  Vp = np.array([[-((p['R_s_p']**(4))*(9*np.sqrt(2429)-457)) / (381150*(p['D_s_p']**(2))), ((p['R_s_p']**(4))*(9*np.sqrt(2429)+457))/(381150*(p['D_s_p']**(2))),1],\
                [(p['R_s_p']**(2))*(np.sqrt(2429)-63) / (2310*p['D_s_p']), -(p['R_s_p']**(2)) * (np.sqrt(2429)+63)/(2310*p['D_s_p']), 0],\
                [1, 1, 0]])

  inv_Vn = inv(Vn)
  inv_Vp = inv(Vp)

  An1 = mtimes(mtimes(inv_Vn,An),Vn)
  Ap1 = mtimes(mtimes(inv_Vp,Ap),Vp)

  Bn1 = mtimes(inv_Vn,Bn)
  Bp1 = mtimes(inv_Vp,Bp)

  Cn1 = mtimes(Cn,Vn)
  Cp1 = mtimes(Cp,Vp)

  # Perform additional transformation such that it's exactly \bar{c}_{s}^{\pm}
  Vn2 = np.diag(np.array([1,1,1.0/Cn1[1,2]]))
  Vp2 = np.diag(np.array([1,1,1.0/Cp1[1,2]]))

  inv_Vn2 = inv(Vn2)
  inv_Vp2 = inv(Vp2)

  An2 = mtimes(mtimes(inv_Vn2,An1),Vn2)
  Ap2 = mtimes(mtimes(inv_Vp2,Ap1),Vp2)

  Bn2 = mtimes(inv_Vn2,Bn1)
  Bp2 = mtimes(inv_Vp2,Bp1)

  Cn2 = mtimes(Cn1,Vn2)
  Cp2 = mtimes(Cp1,Vp2)


  An = An2
  An[2,:]=np.array([0,0,0])
  An[0,1]=0
  An[1,0]=0
  An_normalized_D = An / p['D_s_n']
  An_normalized_R = An * (p['R_s_n']**(2))
  Ap = Ap2
  Ap[2,:] = np.array([0,0,0])
  Ap[0,1] = 0 # reduce numerical error
  Ap[1,0] = 0 # reduce numerical error
  Ap_normalized_D = Ap / p['D_s_p']
  Ap_normalized_R = Ap * (p['R_s_p']**(2))
  Bn = Bn2
  Bp = Bp2
  Cn = Cn2
  Cn[1,0] = 0 # reduce numerical error
  Cn[1,1] = 0 # reduce numerical error
  Cp = Cp2
  Cp[1,0] = 0 # reduce numerical error
  Cp[1,1] = 0 # reduce numerical error

  # RETURN
  return An, Bn, Ap, Bp, Cn, Cp


def symsubsnum(R,D):
  # n = np.concatenate((-3.0/R, -(4.0*R)/(11.0*D), -(R**(3))/(165*(D**(2)))),axis=0) # zero-dimensional arrays cannot be concatenated
  n = np.array([[-3.0/R],[-(4.0*R)/(11.0*D)],[-(R**(3))/(165*(D**(2)))]])
  return n

def symsubsden(R,D):
  # n = vertcat(0,1,(3*(R**(2)))/(55*D),(R**(4))/(3465*(D**(2))))
  n = np.array([[0],[1],[(3*(R**(2)))/(55*D)],[(R**(4))/(3465*(D**(2)))]])
  return n


def c_e_mats(p):
  # Lumped coefficients
  Del_xn = p['L_n'] * p['delta_x_n']
  Del_xs = p['L_s'] * p['delta_x_s']
  Del_xp = p['L_p'] * p['delta_x_p']

  ## Matrices in nonlinear dynamics
  # M1
  M1n = sps.dia_matrix(np.diag(np.ones(p['Nxn']-2),k=1) - np.diag(np.ones(p['Nxn']-2),k=-1)) / (2*Del_xn) #.toarray() for access the array
  M1s = sps.dia_matrix(np.diag(np.ones(p['Nxs']-2),k=1) - np.diag(np.ones(p['Nxs']-2),k=-1)) / (2*Del_xs) #.toarray() for access the array
  M1p = sps.dia_matrix(np.diag(np.ones(p['Nxp']-2),k=1) - np.diag(np.ones(p['Nxp']-2),k=-1)) / (2*Del_xp) #.toarray() for access the array

  M1n = M1n.toarray()
  M1s = M1s.toarray()
  M1p = M1p.toarray()

  # M2
  M2n = np.zeros((p['Nxn']-1,2))
  M2n[0,0] = -1 / (2*Del_xn)
  M2n[-1,-1] = 1 / (2*Del_xn)

  M2s = np.zeros((p['Nxs']-1,2))
  M2s[0,0] = -1 / (2*Del_xs)
  M2s[-1,-1] = 1 / (2*Del_xs)

  M2p = np.zeros((p['Nxp']-1,2))
  M2p[0,0] = -1 / (2*Del_xp)
  M2p[-1,-1] = 1 / (2*Del_xp)

  # M3
  M3n = sps.dia_matrix(-2 * np.diag(np.ones(p['Nxn']-1),k=0) + np.diag(np.ones(p['Nxn']-2),k=1) + np.diag(np.ones(p['Nxn']-2),k=-1)) / (Del_xn ** 2)
  M3s = sps.dia_matrix(-2 * np.diag(np.ones(p['Nxs']-1),k=0) + np.diag(np.ones(p['Nxs']-2),k=1) + np.diag(np.ones(p['Nxs']-2),k=-1)) / (Del_xs ** 2)
  M3p = sps.dia_matrix(-2 * np.diag(np.ones(p['Nxp']-1),k=0) + np.diag(np.ones(p['Nxp']-2),k=1) + np.diag(np.ones(p['Nxp']-2),k=-1)) / (Del_xp ** 2)

  M3n = M3n.toarray()
  M3s = M3s.toarray()
  M3p = M3p.toarray()


  # M4
  M4n = np.zeros((p['Nxn']-1,2))
  M4n[0,0] = 1 / (Del_xn ** 2)
  M4n[-1,-1] = 1 / (Del_xn ** 2)

  M4s = np.zeros((p['Nxs']-1,2))
  M4s[0,0] = 1 / (Del_xs ** 2)
  M4s[-1,-1] = 1 / (Del_xs ** 2)

  M4p = np.zeros((p['Nxp']-1,2))
  M4p[0,0] = 1 / (Del_xp ** 2)
  M4p[-1,-1] = 1 / (Del_xp ** 2)

  # M5
  M5n = (1-p['t_plus']) * p['a_s_n'] / p['epsilon_e_n'] * np.identity(p['Nxn']-1)
  M5p = (1-p['t_plus']) * p['a_s_p'] / p['epsilon_e_p'] * np.identity(p['Nxp']-1)

  ## Boundary conditions

  N1 = np.zeros((4,p['Nx']-3))
  N2 = np.zeros((4,4))

  # BC1
  N1[0,0] = 4
  N1[0,1] = -1
  N2[0,0] = -3

  # BC2
  N1[1,p['Nxn']-3] = (p['epsilon_e_n']**p['brug']) / (2*Del_xn)
  N1[1,p['Nxn']-2] = (-4*p['epsilon_e_n']**p['brug']) / (2*Del_xn)
  N2[1,1] = (3*p['epsilon_e_n']**p['brug']) / (2*Del_xn) + (3*p['epsilon_e_s']**p['brug'])/(2*Del_xs)
  N1[1,p['Nxn']-1] = (-4*p['epsilon_e_s']**p['brug']) / (2*Del_xs)
  N1[1,p['Nxn']]   = (p['epsilon_e_s']**p['brug']) / (2*Del_xs)

  # BC3
  N1[2,p['Nxn']+p['Nxs']-4] = (p['epsilon_e_s'] ** p['brug']) / (2*Del_xs)
  N1[2,p['Nxn']+p['Nxs']-3] = (-4 * p['epsilon_e_s'] ** p['brug']) / (2*Del_xs)
  N2[2,2] = (3*p['epsilon_e_s'] ** p['brug']) / (2*Del_xs) + (3*p['epsilon_e_p']**p['brug']) / (2*Del_xp)
  N1[2,p['Nxn']+p['Nxs']-2] = (-4*p['epsilon_e_p']**p['brug']) / (2*Del_xp)
  N1[2,p['Nxn']+p['Nxs']-1] = (p['epsilon_e_p']**p['brug']) / (2*Del_xp)

  # BC4
  N1[3,-2] = 1
  N1[3,-1] = -4
  N2[-1,-1] = 3

  # Boundary Mats
  C_ce = np.dot( - np.linalg.inv(N2), N1 )

  # Return
  return M1n, M2n, M3n, M4n, M5n, M1s, M2s, M3s, M4s, M1p, M2p, M3p, M4p, M5p, C_ce


def phi_s_mats(p):
  sigma_n_eff = p['sig_n'] * (p['epsilon_s_n']+p['epsilon_f_n'])**(p['brug'])
  sigma_p_eff = p['sig_p'] * (p['epsilon_s_p']+p['epsilon_f_p'])**(p['brug'])

  alpha_n = 1 / (2 * p['L_n'] * p['delta_x_n'])
  alpha_p = 1 / (2 * p['L_p'] * p['delta_x_p'])

  ## Block matrices
  # M1 : phi_s x
  M1n = alpha_n * (np.diag(np.ones(p['Nxn']-2), k=1) + np.diag(-1*np.ones(p['Nxn']-2), k=-1))
  M1p = alpha_p * (np.diag(np.ones(p['Nxp']-2), k=1) + np.diag(-1*np.ones(p['Nxp']-2), k=-1))

  # M2 : phi_s z
  M2n = np.zeros((p['Nxn']-1,2))
  M2n[0,0] = -alpha_n
  M2n[-1,-1] = alpha_n

  M2p = np.zeros((p['Nxp']-1,2))
  M2p[0,0] = -alpha_p
  M2p[-1,-1] = alpha_p

  # M3 : i_e
  M3n = horzcat(np.zeros((p['Nxn']-1,1)), np.diag(-1*np.ones(p['Nxn']-1)), np.zeros((p['Nxn']-1,1)) ) / sigma_n_eff
  M3p = horzcat(np.zeros((p['Nxp']-1,1)), np.diag(-1*np.ones(p['Nxp']-1)), np.zeros((p['Nxp']-1,1)) ) / sigma_p_eff

  # M4 : I
  M4n = 1/sigma_n_eff * np.ones((p['Nxn']-1,1))
  M4p = 1/sigma_p_eff * np.ones((p['Nxp']-1,1))

  # N1 : phi_s x
  N1n = np.zeros((2,p['Nxn']-1))
  N1n[0,0] = 4*alpha_n
  N1n[0,1] = -1*alpha_n
  N1n[1,-1] = -4*alpha_n
  N1n[1,-2] = 1*alpha_n

  N1p = np.zeros((2,p['Nxp']-1))
  N1p[0,0] = 4*alpha_p
  N1p[0,1] = -1*alpha_p
  N1p[1,-1] = -4*alpha_p
  N1p[1,-2] = 1*alpha_p

  # N2 : phi_s z
  N2n = np.array([[-3*alpha_n, 0],[0, 3*alpha_n]])
  N2p = np.array([[-3*alpha_p, 0],[0, 3*alpha_p]])

  # N3 : I
  N3n = np.array([[1.0/sigma_n_eff],[0]])
  N3p = np.array([[0],[1.0/sigma_p_eff]])

  # Form F, G matrices

  inv_N2n = np.linalg.inv(N2n)
  inv_N2p = np.linalg.inv(N2p)

  F1n = M1n - mtimes(M2n,mtimes(inv_N2n,N1n))
  F2n = M3n
  Gn = M4n - mtimes(M2n,mtimes(inv_N2n,N3n))

  F1p = M1p - mtimes(M2p,mtimes(inv_N2p,N1p))
  F2p = M3p
  Gp = M4p - mtimes(M2p,mtimes(inv_N2p,N3p))


  # Compute C,D matrices for boundary values
  Cn = - mtimes(inv_N2n,N1n)
  Dn = - mtimes(inv_N2n,N3n)

  Cp = - mtimes(inv_N2p,N1p)
  Dp = - mtimes(inv_N2p,N3p)


  # Return
  return F1n, F1p, F2n, F2p, Gn, Gp, Cn, Cp, Dn, Dp


def i_e_mats(p):

  # Conductivity and FD Length Coefficients
  alpha_n = 1 / (2 * p['L_n'] * p['delta_x_n'] * p['a_s_n'])
  alpha_p = 1 / (2 * p['L_p'] * p['delta_x_p'] * p['a_s_p'])

  beta_n = p['Faraday']
  beta_p = p['Faraday']


  ## Block matrices
  # M1 : i_e x
  M1n = alpha_n * (np.diag(np.ones(p['Nxn']-2),k=1) + np.diag(-1 * np.ones(p['Nxn']-2),k=-1))
  M1p = alpha_p * (np.diag(np.ones(p['Nxp']-2),k=1) + np.diag(-1 * np.ones(p['Nxp']-2),k=-1))

  # M2 : i_e z
  M2n = np.zeros((p['Nxn']-1,2))
  M2n[0,0] = -alpha_n
  M2n[-1,-1] = alpha_n

  M2p = np.zeros((p['Nxp']-1,2))
  M2p[0,0] = -alpha_p
  M2p[-1,-1] = alpha_p

  # M3 : J
  M3n = -beta_n * np.diag(np.ones(p['Nxn']-1))
  M3p = -beta_p * np.diag(np.ones(p['Nxp']-1))


  # N1 : i_e x
  N1n = np.zeros((2,p['Nxn']-1))
  N1p = np.zeros((2,p['Nxp']-1))


  # N2 : i_e z
  N2n = np.eye(2)
  N2p = np.eye(2)


  # N3 : jn
  N3n = np.zeros((2,p['Nxn']-1))
  N3p = np.zeros((2,p['Nxp']-1))


  # N4 : I
  N4n = np.array([[0],[1]])
  N4p = np.array([[1],[0]])


  # Form F matrices
  inv_N2n = np.linalg.inv(N2n)
  inv_N2p = np.linalg.inv(N2p)

  F1n = M1n - mtimes(M2n,mtimes(inv_N2n,N1n))
  F2n = M3n - mtimes(M2n,mtimes(inv_N2n,N3n))
  F3n = mtimes(M2n,mtimes(inv_N2n,N4n))

  F1p = M1p - mtimes(M2p,mtimes(inv_N2p,N1p))
  F2p = M3p - mtimes(M2p,mtimes(inv_N2p,N3p))
  F3p = mtimes(M2p,mtimes(inv_N2p,N4p))


  # Return
  return F1n, F1p, F2n, F2p, F3n, F3p


def phi_e_mats(p):

  M1_pen_skel = np.diag(np.ones(p['Nxn']-2),k=1) + np.diag(-np.ones(p['Nxn']-2),k=-1)
  M1_pes_skel = np.diag(np.ones(p['Nxs']-2),k=1) + np.diag(-np.ones(p['Nxs']-2),k=-1)
  M1_pep_skel = np.diag(np.ones(p['Nxp']-2),k=1) + np.diag(-np.ones(p['Nxp']-2),k=-1)


  # Conductivity and FD(Finite Diff.) length coefficients
  alpha_n = 1 / (2*p['L_n']*p['delta_x_n'])
  alpha_ns = 1 / (p['L_n'] * p['delta_x_n'] + p['L_s'] * p['delta_x_s'])
  alpha_s = 1 / (2 * p['L_s'] * p['delta_x_s'])
  alpha_sp = 1 / (p['L_s'] * p['delta_x_s'] + p['L_p'] * p['delta_x_p'])
  alpha_p = 1 / (2*p['L_p']*p['delta_x_p'])


  # eqns for Anode
  M1n_c = alpha_n * M1_pen_skel
  M1n_r = np.zeros(p['Nxn']-1)
  M1n_r[-1] = alpha_n

  M1_n = horzcat(M1n_c, M1n_r, np.zeros((p['Nxn']-1, p['Nxs']-1 +1 + p['Nxp'] - 1)))

  # eqns for ns interface
  M1ns_c = horzcat(-alpha_ns, 0, alpha_ns)

  M1_ns = horzcat(np.zeros((1,p['Nxn']-2)), M1ns_c, np.zeros((1,p['Nxs']-2 + 1 + p['Nxp'] -1)))

  # eqns for seperator, s
  M1s_c = alpha_s * M1_pes_skel
  M1s_l = np.zeros(p['Nxs']-1)
  M1s_l[0] = -alpha_s
  M1s_r = np.zeros(p['Nxs']-1)
  M1s_r[-1] = alpha_s

  M1_s = horzcat(np.zeros((p['Nxs']-1,p['Nxn']-1)), M1s_l, M1s_c, M1s_r, np.zeros((p['Nxs']-1,p['Nxp']-1)) )


  # eqns for sp interface
  M1sp_c = horzcat(-alpha_sp, 0, alpha_sp)
  M1_sp = horzcat(np.zeros((1,p['Nxn']-1 + 1 + p['Nxs'] - 2)), M1sp_c, np.zeros((1,p['Nxp']-2)))

  # eqns for cathode, p
  M1p_c = alpha_p * M1_pep_skel
  M1p_l = np.zeros(p['Nxp']-1)
  M1p_l[0] = -alpha_p

  M1_p = horzcat(np.zeros((p['Nxp']-1, p['Nxn']-1 +1 + p['Nxs']-1)), M1p_l, M1p_c)

  # assemble submatrices
  M1 = vertcat(M1_n, M1_ns, M1_s, M1_sp, M1_p)

  ## M2 : phi_e z
  M2 = SX.zeros((M1.shape[0],2))
  M2[0,0] = -alpha_n
  M2[-1,-1] = alpha_p

  ## M3 : i_e
  M3 = sps.eye(M1.shape[0],M1.shape[1])

  ## M4 : ln(c_ex)
  M4 = horzcat(M2[:,0],M1,M2[:,1])


  ## Boundary conditions
  N1 = np.zeros((2,M1.shape[0]))
  N1[0,0] = 0
  N1[0,1] = 0
  N1[1,-2] = 1
  N1[1,-1] = -4

  N2_inv = np.diag(np.array([1, 1.0/3]))


  ## Form F matrices

  C = - mtimes(N2_inv, N1)

  # return
  return M1, M2, M3, M4, C

def electrolyteDe(c_e, T, arg):

  if arg == 0:
    # Source: from DUALFOIL LiPF6 in EC:DMC, Capiglia et al. 1999

    D_e = 5.34e-10*np.exp(-0.65*c_e/1e3)
    dD_e = -0.65*D_e/1e3



  elif arg == 1:
    # Source: Transport Properties of LiPF6-Based Li-Ion Battery Electrolytes - Valoen and Reimers
    # T : temperature in Kalvin
    # c : mol/L, not mol/m^3

    c_e = c_e * 0.001 # unit conversion

    # param table
    D00 = -2.4887
    D01 = -822.3727
    D10 = 1.0369
    D11 = -407.9648
    Tg0 = 0.0094
    Tg1 = 2.8023

    Tg = Tg0 + c_e * Tg1
    D0 = D00 + D01 / (T-Tg)
    D1 = D10 + D11 / (T-Tg)

    D_e = 0.0001 * 10 ** (D0 + D1 * c_e)

    dD_e = 10**(-7) * 10**(D00 - D01/(Tg0 - T + Tg1*c_e) + c_e*(D10 - D11/(Tg0 - T + Tg1*c_e)))*log(10)*(D10 - D11/(Tg0 - T + Tg1*c_e) + (D01*Tg1)/(Tg0 - T + Tg1*c_e)**(2) + (D11*Tg1*c_e)/(Tg0 - T + Tg1*c_e)**(2))
    # print D_e, dD_e


  return D_e, dD_e


def electrolyteCond(c_e, arg):

  if arg == 0:

    # From DUALFOIL LiPF6 in EC:DMC, Capiaglia et al. 1999
    kappa = 0.0911+1.9101*c_e/1e3 - 1.052*(c_e/1e3)**2 + 0.1554*(c_e/1e3)**3

  elif arg == 1:

    # 'Transport and Electrochemical Properties and Spectral Features of Non-Aqueous Electrolytes Containing LiFSI in Linear Carbonate Solvents.

    P1 = 0.682
    P2 = -0.091
    c_max = 1 # [mol/l] ** c_e is [mol/m^3], so converted unit as [mol/l] see below (0.001 multiply term)
    k_max = 0.00918
    kappa = 100*k_max*(c_e*0.001 / c_max)**2 * np.exp(P2*(c_e*0.001 - c_max)**2 - P1*(c_e*0.001 - c_max)/c_max) # Unit match. [S/cm] -> [S/m]


  return kappa


def electrolyteAct(c_e, Temp, p):

  # From LiPF6, Valoen et al. 2005
  # Fig.6 in the paper

  # DataFitting Coefficients

  v00 = 0.601
  v01 = 0
  v10 = -0.24
  v11 = 0
  v20 = 0
  v21 = 0
  v30 = 0.982
  v31 = -0.0052

  c_e = c_e/1000 # UnitConversion: 1 mol/L -> 1000 mol/m^3
  dActivity = ((v00 + v10 * (c_e)**(0.5) + v30*(1+v31*(Temp - p['T_amb'])) * (c_e)**(1.5)) / (1-p['t_plus']))-1

  # if(nargout == 2)
  #     d_dactivity = (0.5 * v10 * (c_e).^(-0.5) + (1.5)*v30*(1+v31*(T - p.T_ref))*(c_e).^(0.5))/(1-p.t_plus);
  #     varargout{1} = d_dactivity;

  return dActivity


def init_cs_LCO(p,V0):
  # Input: Parameter sets, p
  #        Initial voltage, V0

  # Bi-section algorithm parameters
  max_iters = 500
  x = np.zeros(max_iters)
  f = np.nan * np.ones(max_iters)
  tol = 1e-5

  # Initial Guess
  x_low = 0.2 * p['c_s_p_max']
  x_high = 1.0 * p['c_s_p_max']
  x[0] = 0.6 * p['c_s_p_max']
  #print('target V0: {}'.format(V0))

  for idx in range(max_iters):

      theta_p = x[idx] / p['c_s_p_max']
      theta_n = (p['n_Li_s']-p['epsilon_s_p']*p['L_p']*p['Area']*x[idx])/(p['c_s_n_max']*p['epsilon_s_n']*p['L_n']*p['Area'])

      OCPn = refPotentialAnode_casadi(theta_n)
      OCPp = refPotentialCathode_casadi(theta_p)
      #print('search OCPp - OCPn : {}'.format(OCPp-OCPn))
      f[idx] = OCPp - OCPn - V0

      if np.abs(f[idx]) <= tol :
          break
      elif f[idx] <= 0 :
          x_high = x[idx]
      else:
          x_low = x[idx]

      # Bisection
      x[idx+1] = (x_high + x_low) / 2

      if idx == max_iters-1 :
          print ('PLEASE CHECK INITIAL VOLTAGE & CONDITION')

  csp0 = theta_p * p['c_s_p_max']
  csn0 = theta_n * p['c_s_n_max']

  # print(csp0, p['c_s_p_max'])
  # print(csn0, p['c_s_n_max'])

  return csn0, csp0


def exch_cur_dens(p, c_ss_n, c_ss_p, c_e):

  c_e_n = c_e[range(p['Nxn']-1)]
  c_e_p = c_e[range(p['Nxn']-1 + p['Nxs']-1 ,p['Nx']-3)]

  # Compute exchange current density
  scalar_term_n = p['k_n'] * ((p['c_s_n_max']-c_ss_n)**(p['alph'])) * ((c_ss_n)**(p['alph']))
  c_en = c_e_n ** (p['alph'])
  # i_0n = mtimes(scalar_term_n,c_en)
  i_0n = scalar_term_n * c_en

  scalar_term_p = p['k_p'] * ((p['c_s_p_max']-c_ss_p)**(p['alph'])) * ((c_ss_p)**(p['alph']))
  c_ep = c_e_p ** (p['alph'])
  # i_0p = mtimes(scalar_term_p,c_ep)
  i_0p = scalar_term_p * c_ep

  # return scalar_term1, scalar_term_p
  return i_0n, i_0p



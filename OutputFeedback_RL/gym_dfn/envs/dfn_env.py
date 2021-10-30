# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 2021

@author: Saehong Park
"""

import gym
import pdb
from gym import spaces
from gym.envs.toy_text import discrete
import numpy as np
from numpy import matlib as mb

import matplotlib.pyplot as plt
from control.matlab import *
from gym_dfn.envs.ParamFile_LCO2 import *
from gym_dfn.envs.DFN_Utils import *


DISCRETE = False

class DFN(discrete.DiscreteEnv):




	def __init__(self, sett, cont_sett, init_v=3.2, init_t=p['T_amb']):

		#==============================================================================
		# Environment Setting
		#==============================================================================
		self.sett = sett
		self.cont_sett = cont_sett

		# Time step
		self.dt = self.sett['sample_time'] 

		#==============================================================================
		# Battery specification
		#==============================================================================

		self.discrete = DISCRETE

		cn_low, cp_low = init_cs_LCO(p, p['volt_min'])
		cn_high, cp_high = init_cs_LCO(p, p['volt_max'])

		delta_cn = cn_high - cn_low
		delta_cp = cp_low - cp_high

		p['cn0'] = cn_low
		p['cn100'] = cn_high
		p['cp0'] = cp_low
		p['cp100'] = cp_high

		self.OneC = min(p['epsilon_s_n']*p['L_n']*p['Area']*delta_cn*p['Faraday']/3600, p['epsilon_s_p']*p['L_p']*p['Area']*delta_cp*p['Faraday']/3600)
		

		self.Temp = p['T_amb']
		self.cn0 = p['cn0']
		self.cn100 = p['cn100'] # x100, Cell SOC 100
		self.cp0 = p['cp0']  # y0, Cell SOC 0
		self.cp100 = p['cp100'] # y100, Cell SOC 100

		self.cn_low = p['c_s_n_max'] * self.cn0 # Minimum stochiometry of Anode (CELL_SOC=0)
		self.cn_high = p['c_s_n_max'] * self.cn100 # Maximum stoichiometry of Anode (CELL_SOC=1)

		self.cp_low  = p['c_s_p_max'] * self.cp100 # Minimum stochiometry of Cathode (CELL_SOC=1)
		self.cp_high = p['c_s_p_max'] * self.cp0 # Maximum stochiometry of Cathode (CELL_SOC=0)

		self.Delta_cn = self.cn_high - self.cn_low
		self.Delta_cp = self.cp_high - self.cp_low

		self.dtheta_n = self.cn100 - self.cn0
		self.dtheta_p = self.cp0 - self.cp100

		self.neg_theoretic_cap = (p['Faraday'] * p['epsilon_s_n'] * p['L_n'] * p['Area'] * self.dtheta_n * p['c_s_n_max']) / float(3600)
		self.pos_theoretic_cap = (p['Faraday'] * p['epsilon_s_p'] * p['L_p'] * p['Area'] * self.dtheta_p * p['c_s_p_max']) / float(3600)

		#print('Theoretical OneC is,  {} A'.format(self.OneC)) 

		self.exp_OneC = 58.397 # experiment OneC current [A]
	



		#==============================================================================
		# Vector length
		#==============================================================================

		self.Ncsn = p['PadeOrder'] * (p['Nxn']-1) # disretization in the anode times pade order
		self.Ncsp = p['PadeOrder'] * (p['Nxp']-1)# disretization in the cathode times pade order
		self.Nce = p['Nxn'] - 1 + p['Nxs'] -1 + p['Nxp'] - 1 # electrolyte discretization
		self.Nx = self.Nce
		self.Nc = self.Ncsn + self.Ncsp + self.Nce

		self.Nn = p['Nxn']-1
		self.Ns = p['Nxs']-1
		self.Np = p['Nxp']-1
		self.Nnp = self.Nn + self.Np

		#==============================================================================
		# Initial conditions
		#==============================================================================

		# Initial condition
		self.csn0, self.csp0 = init_cs_LCO(p, init_v) #init_cs_NMC(p, init_v) #init_cs_NMC(p, init_v) #init_cs_LCO(p, init_v)
		self.SOCn = self.csn0/p['c_s_n_max']
		#given initial voltage, every particle in the anode should have csn0, and vice versa

		self.c_s_n_pade0 = np.zeros((p['PadeOrder'],1))
		self.c_s_p_pade0 = np.zeros((p['PadeOrder'],1))

		self.c_s_n_pade0[2] = self.csn0
		self.c_s_p_pade0[2] = self.csp0

		self.c_s_n = mb.repmat(self.c_s_n_pade0,self.Nn,1)#vertical concatenation
		self.c_s_p = mb.repmat(self.c_s_p_pade0,self.Np,1)

		self.V = refPotentialCathode_casadi(self.csp0/p['c_s_p_max']) - refPotentialAnode_casadi(self.csn0/p['c_s_n_max'])
		#print('Init volt: {}'.format(self.V))
		# print('intial voltage ,  ', self.V) ### DEBUG CHECK

		# Temperature initial condition
		self.T0 = p['T_amb']

		# Electrolyte initial condition
		self.c_ex = p['c_e0'] * np.ones(p['Nxn']-1 + p['Nxs']-1 + p['Nxp']-1)

		# Solid potential
		self.Uref_n0 = refPotentialAnode_casadi(self.csn0 * np.ones(self.Nn) / p['c_s_n_max'])
		self.Uref_p0 = refPotentialCathode_casadi(self.csp0 * np.ones(self.Np) / p['c_s_p_max'])
		self.phi_s_n0 = self.Uref_n0
		self.phi_s_p0 = self.Uref_p0
		#used to calculate voltage

		# Electrolyte current
		self.i_en0 = np.zeros(self.Nn)
		self.i_ep0 = np.zeros(self.Np)

		# Electrolyte potential
		self.phi_e0 = np.zeros(self.Nx+2)

		# Molar ionic flux
		self.jn0 = np.zeros(self.Nn)
		self.jp0 = np.zeros(self.Np)


		## Integrate initial condition
		self.x_init = vertcat(self.c_s_n,self.c_s_p,self.c_ex, self.T0) # Pade
		self.x_init = self.x_init.full().reshape(self.x_init.shape[0],)

		self.state = self.x_init

		self.z_init = vertcat(self.phi_s_n0,self.phi_s_p0,self.i_en0,self.i_ep0,self.phi_e0,self.jn0,self.jp0)
		self.z_state = self.z_init.full()
		self.nLis = p['epsilon_s_n'] * p['L_n'] * p['Area'] * self.csn0 + p['epsilon_s_p'] * p['L_p'] * p['Area'] * self.csp0
		

		#==============================================================================
		# Simulation Setup
		#==============================================================================

		# States
		self.x0 = SX.sym("x0",self.c_s_n.size) # c_s_n
		self.x1 = SX.sym("x1",self.c_s_p.size) # c_s_p
		self.x2 = SX.sym("x2",self.c_ex.size) # c_e
		self.x3 = SX.sym("x3",1)   # T1

		self.x = vertcat(self.x0,self.x1,self.x2, self.x3)

		# Algebraic states
		self.z0 = SX.sym("z0", self.phi_s_n0.size)
		self.z1 = SX.sym("z1", self.phi_s_p0.size)
		self.z2 = SX.sym("z2", self.i_en0.size)
		self.z3 = SX.sym("z3", self.i_ep0.size)
		self.z4 = SX.sym("z4", self.phi_e0.size)
		self.z5 = SX.sym("z5", self.jn0.size)
		self.z6 = SX.sym("z6", self.jp0.size)

		self.z = vertcat(self.z0,self.z1,self.z2,self.z3,self.z4,self.z5,self.z6)

		# Input
		# self.u = vertcat(self.u0,self.u1)
		self.u = SX.sym("u")


		# Call DFN dynamics and outputs (casadi variables)
		self.x_dot, self.g_, self.L, self.x_outs, self.z_outs, self.info_outs, self.param_outs, self.debug_outs = dae_dfn_casadi_pade(self.x,self.z,self.u,p)


		# Function call.
		self.helper = Function('helper',[self.x,self.z,self.u],[self.info_outs, self.debug_outs],['x','z','p'],['info_outs', 'debug_out'])

		# Build integrator
		self.dae = { 'x' :  self.x , 'z' :  self.z, 'p' :  self.u ,  'ode' :  self.x_dot, 'alg' :  self.g_, 'quad' :  self.L }
		self.opts = {}
		self.opts["fsens_err_con"] = True # include the forward sensitivities in all error controls
		self.opts["quad_err_con"] = True  # Should the quadratures affect the step size control
		self.opts["abstol"] = 1e-4
		self.opts["reltol"] = 1e-4
		self.opts["t0"] = 0
		self.opts["tf"] = self.dt
		# print(" Delta t in environment ,  %f" %self.dt) ### DEBUG CHECK
		self.F = integrator( 'F' , 'idas' , self.dae, self.opts )

		# print("info_outs, ", self.info_outs)

		#==============================================================================
		# Indexing
		#==============================================================================

		#get indices for info_outs
		self.cssn_idx = range(0, p['Nxn']-1)
		self.cssp_idx = range(p['Nxn']-1, p['Nxn']-1 + p['Nxp']-1)
		self.cex_idx = range(p['Nxn']-1 + p['Nxp']-1, p['Nxn']-1 + p['Nxp']-1 + (p['Nxn']-1+p['Nxs']-1+p['Nxp']-1 +4))
		self.c_avgn_idx = range(p['Nxn']-1 + p['Nxp']-1 + (p['Nxn']-1+p['Nxs']-1+p['Nxp']-1 +4), p['Nxn']-1 + p['Nxp']-1 + (p['Nxn']-1+p['Nxs']-1+p['Nxp']-1 +4) + p['Nxn'] - 1)
		self.c_avgp_idx = range(-1 + p['Nxn']-1 + p['Nxp']-1 + (p['Nxn']-1+p['Nxs']-1+p['Nxp']-1 +4) +((p['Nxn']-1) +1) , (p['Nxn']-1 + p['Nxp']-1 + (p['Nxn']-1+p['Nxs']-1+p['Nxp']-1 +4) +(p['Nxn']-1) +(p['Nxp']-1)))
		self.etan_idx = range(-1 + p['Nxn']-1 + p['Nxp']-1 + (p['Nxn']-1+p['Nxs']-1+p['Nxp']-1 +4) +(p['Nxn']-1) +((p['Nxp']-1) +1) , (p['Nxn']-1 + p['Nxp']-1 + (p['Nxn']-1+p['Nxs']-1+p['Nxp']-1 +4) +(p['Nxn']-1) +(p['Nxp']-1) + (p['Nxn']-1)))
		self.etap_idx = range(-1 + p['Nxn']-1 + p['Nxp']-1 + (p['Nxn']-1+p['Nxs']-1+p['Nxp']-1 +4) +(p['Nxn']-1) +(p['Nxp']-1) + ((p['Nxn']-1) +1) , (p['Nxn']-1 + p['Nxp']-1 + (p['Nxn']-1+p['Nxs']-1+p['Nxp']-1 +4) + (p['Nxn']-1) +(p['Nxp']-1) + (p['Nxn']-1) + (p['Nxp']-1)))
		self.ce0n_idx = range(-1 + p['Nxn']-1 + p['Nxp']-1 + (p['Nxn']-1+p['Nxs']-1+p['Nxp']-1 +4) +(p['Nxn']-1) +(p['Nxp']-1) + (p['Nxn']-1) + ((p['Nxp']-1) +1) , (p['Nxn']-1 + p['Nxp']-1 + (p['Nxn']-1+p['Nxs']-1+p['Nxp']-1 +4) +(p['Nxn']-1) +(p['Nxp']-1) + (p['Nxn']-1) + (p['Nxp']-1) +1))
		self.ce0p_idx = range(-1 + p['Nxn']-1 + p['Nxp']-1 + (p['Nxn']-1+p['Nxs']-1+p['Nxp']-1 +4) +(p['Nxn']-1) +(p['Nxp']-1) + (p['Nxn']-1) + ((p['Nxp']-1) +2) , (p['Nxn']-1 + p['Nxp']-1 + (p['Nxn']-1+p['Nxs']-1+p['Nxp']-1 +4) +(p['Nxn']-1) +(p['Nxp']-1) + (p['Nxn']-1) + (p['Nxp']-1) +2))
		self.etasLn_idx = range(-1 + p['Nxn']-1 + p['Nxp']-1 + (p['Nxn']-1+p['Nxs']-1+p['Nxp']-1 +4) +(p['Nxn']-1) +(p['Nxp']-1) + (p['Nxn']-1) + ((p['Nxp']-1) +3) , (p['Nxn']-1 + p['Nxp']-1 + (p['Nxn']-1+p['Nxs']-1+p['Nxp']-1 +4) +(p['Nxn']-1) +(p['Nxp']-1) + (p['Nxn']-1) + (p['Nxp']-1) +3))
		self.Volt_idx = range(-1 + p['Nxn']-1 + p['Nxp']-1 + (p['Nxn']-1+p['Nxs']-1+p['Nxp']-1 +4) +(p['Nxn']-1) +(p['Nxp']-1) + (p['Nxn']-1) + ((p['Nxp']-1) +4) , (p['Nxn']-1 + p['Nxp']-1 + (p['Nxn']-1+p['Nxs']-1+p['Nxp']-1 +4) +(p['Nxn']-1) +(p['Nxp']-1) + (p['Nxn']-1) + (p['Nxp']-1) +4))
		self.nLis_idx = range(-1 + p['Nxn']-1 + p['Nxp']-1 + (p['Nxn']-1+p['Nxs']-1+p['Nxp']-1 +4) +(p['Nxn']-1) +(p['Nxp']-1) + (p['Nxn']-1) + ((p['Nxp']-1) +5) , (p['Nxn']-1 + p['Nxp']-1 + (p['Nxn']-1+p['Nxs']-1+p['Nxp']-1 +4) +(p['Nxn']-1) +(p['Nxp']-1) + (p['Nxn']-1) + (p['Nxp']-1) +5))
		self.nLie_idx = range(-1 + p['Nxn']-1 + p['Nxp']-1 + (p['Nxn']-1+p['Nxs']-1+p['Nxp']-1 +4) +(p['Nxn']-1) +(p['Nxp']-1) + (p['Nxn']-1) + ((p['Nxp']-1) +6) , (p['Nxn']-1 + p['Nxp']-1 + (p['Nxn']-1+p['Nxs']-1+p['Nxp']-1 +4) +(p['Nxn']-1) +(p['Nxp']-1) + (p['Nxn']-1) + (p['Nxp']-1) +6))
		self.i0n_idx = range(-1 + p['Nxn']-1 + p['Nxp']-1 + (p['Nxn']-1+p['Nxs']-1+p['Nxp']-1 +4) +(p['Nxn']-1) +(p['Nxp']-1) + (p['Nxn']-1) + ((p['Nxp']-1) +7) , (p['Nxn']-1 + p['Nxp']-1 + (p['Nxn']-1+p['Nxs']-1+p['Nxp']-1 +4) +(p['Nxn']-1) +(p['Nxp']-1) + (p['Nxn']-1) + (p['Nxp']-1) +6 + (p['Nxn']-1)))
		self.i0p_idx = range(-1 + p['Nxn']-1 + p['Nxp']-1 + (p['Nxn']-1+p['Nxs']-1+p['Nxp']-1 +4) +(p['Nxn']-1) +(p['Nxp']-1) + (p['Nxn']-1) + (p['Nxp']-1 +6 + (p['Nxn']-1)) + 1 , (p['Nxn']-1 + p['Nxp']-1 + (p['Nxn']-1+p['Nxs']-1+p['Nxp']-1 +4) +(p['Nxn']-1) +(p['Nxp']-1) + (p['Nxn']-1) + (p['Nxp']-1) +6 + (p['Nxn']-1)) + (p['Nxp']-1))

		#index for x
		self.out_csn_idx = range(0, (p['PadeOrder'] * (p['Nxn']-1)))
		self.out_csp_idx = range((p['PadeOrder'] * (p['Nxn']-1) + 1) -1,  (p['PadeOrder'] * (p['Nxn']-1) + p['PadeOrder'] * (p['Nxp']-1)))
		self.out_ce_idx = range((p['PadeOrder'] * (p['Nxn']-1) + p['PadeOrder'] * (p['Nxp']-1) + 1) -1 ,  (p['PadeOrder'] * (p['Nxn']-1) + p['PadeOrder'] * (p['Nxp']-1) + p['Nxn']-1+p['Nxs']-1+p['Nxp']-1))
		self.out_T_idx = -1


		#index for z
		self.out_phisn_idx = range(0, p['Nxn']-1)
		self.out_phisp_idx = range((p['Nxn']-1+1) -1 ,  (p['Nxn']-1 + p['Nxp']-1))
		self.out_ien_idx = range((p['Nxn']-1 + p['Nxp']-1 +1) -1 ,  (p['Nxn']-1 + p['Nxp']-1 + p['Nxn']-1))
		self.out_iep_idx = range((p['Nxn']-1 + p['Nxp']-1 + p['Nxn']-1 +1) -1 ,  (p['Nxn']-1 + p['Nxp']-1 + p['Nxn']-1 + p['Nxp']-1))
		self.out_phie_idx = range((p['Nxn']-1 + p['Nxp']-1 + p['Nxn']-1 + p['Nxp']-1 + 1) -1 ,  (p['Nxn']-1 + p['Nxp']-1 + p['Nxn']-1 + p['Nxp']-1 + (p['Nxn']-1 + p['Nxs']-1 + p['Nxp']-1)+2))  # Why 2? not 4?? used to plus 4 to consider boundary conditions.
		self.out_jn_idx = range((p['Nxn']-1 + p['Nxp']-1 + p['Nxn']-1 + p['Nxp']-1 + (p['Nxn']-1 + p['Nxs']-1 + p['Nxp']-1 + 2) + 1) -1 ,  (p['Nxn']-1 + p['Nxp']-1 + p['Nxn']-1 + p['Nxp']-1 + (p['Nxn']-1 + p['Nxs']-1 + p['Nxp']-1 + 2) + (p['Nxn']-1)))
		self.out_jp_idx = range((p['Nxn']-1 + p['Nxp']-1 + p['Nxn']-1 + p['Nxp']-1 + (p['Nxn']-1 + p['Nxs']-1 + p['Nxp']-1 + 2) + (p['Nxn']-1) +1) -1 ,  (p['Nxn']-1 + p['Nxp']-1 + p['Nxn']-1 + p['Nxp']-1 + (p['Nxn']-1 + p['Nxs']-1 + p['Nxp']-1+ 2) + (p['Nxn']-1) + (p['Nxp']-1)))
		
		#==============================================================================
		# GYM Output
		#==============================================================================
		'''
		Reward, States, Info

		'''

		if self.discrete:
			self.currents = np.linspace(-50, 0, 20)
		

		csn_high = np.ones(len(self.c_s_n))*p['c_s_n_max']
		csp_high = np.ones(len(self.c_s_p))*p['c_s_p_max']
		#
		cs_high = np.concatenate((csn_high, csp_high))
		# self.observation_space = spaces.Box(np.zeros(len(cs_high)), cs_high, dtype=np.float32)
		self.M = len(self.x_init)

		# ipdb.set_trace()
		self._max_episode_steps = int(3000 / self.sett['sample_time']) * 10
		self.episode_step = 0

		#==============================================================================
		# GYM Internal information
		#==============================================================================

		self.c_avgn = self.csn0 * np.ones([self.Nn,1])
		self.c_avgp = self.csp0 * np.ones([self.Np,1])
		self.T = self.T0

		self.info = dict()
		self.info['SOCn'] = self.SOCn
		self.info['c_s_n'] = self.c_s_n
		self.info['c_s_p'] = self.c_s_p
		self.info['c_ex'] = self.c_ex
		self.info['V'] = self.V
		self.info['T'] = self.Temp

		# Initial SOC is self.SOCn

		# Target SOC
		self.SOC_desired = cont_sett['references']['soc']


	@property
	def observation_space(self):
		return spaces.Box(low=0, high=49521, shape=(self.M,), dtype=np.float32)

	@property
	def action_space(self):
		if self.discrete:
			return spaces.Discrete(20)
		else: # continuous case.
			return spaces.Box(dtype=np.float32, low=-150, high=0, shape=(1,))


	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action):

		"""
		action,  some form of current input?
		reward,  based on the increase in SoC?
		next state,  output the new state based on the action
		"""

		is_done = False
		is_error = False

		if self.discrete:
			# if not self.action_space.contains(action):
			# 	print('invalid action!')
			# 	return None
			# else:
			action = self.currents[action]
		else:
			action = action
			action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)[0]

		self.Cur = action/p['Area']

		try:
			self.Fk = self.F(x0 = self.state, z0 = self.z_state, p = self.Cur)
		except:
			print('Numerical error occured. Check the code.')
			is_error = True
			pdb.set_trace()


		self.state = self.Fk['xf'].full()
		self.z_state = self.Fk['zf'].full()
		self.V = float(self.Fk['qf'].full() / self.dt)


		
		res = self.helper(x = self.state, z = self.z_state, p=[self.Cur])

		#==============================================================================
		# GYM Internal information
		#==============================================================================
		'''
		Reward, States, Info
		
		'''
		
		self.c_avgn = res['info_outs'][self.c_avgn_idx].full()
		self.c_avgp = res['info_outs'][self.c_avgp_idx].full()
		self.cssn = res['info_outs'][self.cssn_idx].full()
		self.cssp = res['info_outs'][self.cssp_idx ].full()
		self.c_exx = res['info_outs'][self.cex_idx].full()
		self.etan = res['info_outs'][self.etan_idx].full()
		self.etap = res['info_outs'][self.etap_idx].full()
		self.ce0n = res['info_outs'][self.ce0n_idx].full()
		self.ce0p = res['info_outs'][self.ce0p_idx].full()
		self.etasLn = res['info_outs'][self.etasLn_idx].full()[0]

		self.etas_all = res['debug_out'][:]


		self.Volt = res['info_outs'][self.Volt_idx].full()
		self.nLis = res['info_outs'][self.nLis_idx].full()
		self.nLie = res['info_outs'][self.nLie_idx].full()
		self.i0n = res['info_outs'][self.i0n_idx].full()
		self.i0p = res['info_outs'][self.i0p_idx].full()
		self.c_s_n = self.state[self.out_csn_idx]
		self.c_s_p = self.state[self.out_csp_idx]
		self.c_ex = self.state[self.out_ce_idx]
		self.Temp = self.state[self.out_T_idx]

		self.out_phisn = self.z_state[self.out_phisn_idx]
		self.out_phisp = self.z_state[self.out_phisp_idx]
		self.out_ien = self.z_state[self.out_ien_idx]
		self.out_iep = self.z_state[self.out_iep_idx]
		self.out_phie = self.z_state[self.out_phie_idx]
		self.out_jn = self.z_state[self.out_jn_idx]
		self.out_jp = self.z_state[self.out_jp_idx]

		self.theta_n0 = p['cn0']/p['c_s_n_max']
		self.theta_n100 = p['cn100']/p['c_s_n_max']# x100, Cell SOC 100
		self.theta_p0 = p['cp0']/p['c_s_p_max']  # y0, Cell SOC 0
		self.theta_p100 = p['cp100']/p['c_s_p_max'] # y100, Cell SOC 100

		self.SOCn = (float(sum(self.c_avgn)/len(self.c_avgn)) - self.theta_n0*p['c_s_n_max'])/(p['c_s_n_max']*(self.theta_n100 - self.theta_n0))

		self.c_ss_n = res['info_outs'][self.cssn_idx]

		# update the info dictionary

		self.info['SOCn'] = self.SOCn
		self.info['c_s_n'] = self.c_s_n
		self.info['c_s_p'] = self.c_s_p
		self.info['c_ex'] = self.c_ex
		self.info['V'] = self.V
		self.info['T'] = self.Temp[0]# output should be list format.

		# Reward function. (Soft constraint)
		r_temp = -5 * abs(self.Temp[0] - self.cont_sett['constraints']['temperature']['max']) if self.Temp[0] > self.cont_sett['constraints']['temperature']['max'] else 0
		
		
		#r_etas = -100 * abs(self.etasLn[0] - self.cont_sett['constraints']['etasLn']['min']) if self.etasLn[0] < self.cont_sett['constraints']['etasLn']['min'] else 0
		r_volt = -100 * abs(self.V - self.cont_sett['constraints']['voltage']['max']) if self.V > self.cont_sett['constraints']['voltage']['max'] else 0
		r_step = -0.1
		

		reward = r_step + r_temp + r_volt
		self.episode_step += 1


		if self.SOCn >= self.SOC_desired or self.episode_step >= self._max_episode_steps :
			is_done = True
		else:
			is_done = False

		if is_error:
			reward = -10*(2)
			is_done = True
		

		# return
		return np.concatenate((self.c_avgn, self.c_avgp, self.c_ex, self.Temp), axis = None), reward, is_done, self.info



	def render(self):
		pass


	def reset(self, init_v = 3.2, init_t=p['T_amb']):
		self.__init__(self.sett, self.cont_sett, init_v, init_t)
		
		# original
		# return self.state
		# print('Reset')
		#print('Initial set-Voltage: {}, Initial set-Temp: {}'.format(init_v, init_t))

		# modified
		return np.concatenate((self.c_avgn, self.c_avgp, self.c_ex, self.T), axis = None)




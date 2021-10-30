#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 17:08:57 2018

NMC-Pouch cell

@author: shpark
"""
import numpy as np

p={}


#==============================================================================
# Geometric params
#==============================================================================

# Thickness of each layer
p['L_n'] = 100.0e-6       # Thickness of negative electrode [m]
p['L_s'] = 25.4e-6       # Thickness of separator [m]
p['L_p'] = 100.5e-6     # Thickness of positive electrode [m]

L_ccn = 25e-6;    # Thickness of negative current collector [m]
L_ccp = 25e-6;    # Thickness of negative current collector [m]


# Particle Radii
p['R_s_n'] = 5.0e-06 # Radius of solid particles in negative electrode [m]
p['R_s_p'] = 7.5e-06 # Radius of solid particles in positive electrode [m]

# Volume fractions
p['epsilon_s_n'] = 0.70 # Volume fraction in solid for neg. electrode
p['epsilon_s_p'] = 0.67 # Volume fraction in solid for pos. electrode

p['epsilon_e_n'] = 0.3   # Volume fraction in electrolyte for neg. electrode
p['epsilon_e_s'] = 0.4	  # Volume fraction in electrolyte for separator
p['epsilon_e_p'] = 0.3   # Volume fraction in electrolyte for pos. electrode

p['epsilon_f_n'] = 1 - p['epsilon_s_n'] - p['epsilon_e_n']  # Volume fraction of filler in neg. electrode
p['epsilon_f_p'] = 1 - p['epsilon_s_p'] - p['epsilon_e_p']  # Volume fraction of filler in pos. electrode


# Specific interfacial surface area
p['a_s_n'] = 3*p['epsilon_s_n'] / p['R_s_n']  # Negative electrode [m^2/m^3]
p['a_s_p'] = 3*p['epsilon_s_p'] / p['R_s_p']  # Positive electrode [m^2/m^3]


#==============================================================================
# Transport params
#==============================================================================

p['D_s_n0'] = 3.50e-14 # Diffusion coeff for solid in neg. electrode, [m^2/s]
p['D_s_p0'] = 2.24e-14 # Diffusion coeff for solid in pos. electrode, [m^2/s]


# Conductivity of solid
p['sig_n'] = 100    # Conductivity of solid in neg. electrode, [1/Ohms*m]
p['sig_p'] = 0.1  # Conductivity of solid in pos. electrode, [1/Ohms*m]

#==============================================================================
# Kinetic params
#==============================================================================
p['R_f_n'] = 0 # [CCTA-Adaption case study: 1e-4]       # Resistivity of SEI layer, [Ohms*m^2]
p['R_f_p'] = 0 # [CCTA-Adaption case study: 1e-4]       # Resistivity of SEI layer, [Ohms*m^2]
#p.R_c = 2.5e-03;%5.1874e-05/p.Area; % Contact Resistance/Current Collector Resistance, [Ohms-m^2]

# Nominal Reaction rates
p['k_n0'] = 4.0e-05  # Reaction rate in neg. electrode, [(A/m^2)*(mol^3/mol)^(1+alpha)]
p['k_p0'] = 2.5e-06  # Reaction rate in pos. electrode, [(A/m^2)*(mol^3/mol)^(1+alpha)]


#==============================================================================
# Thermodynamic params
#==============================================================================

# Thermal dynamics
p['C_p'] = 1000    # Heat capacity, [J/kg-K]
p['R_th'] = 0.5   # Thermal resistance, [K/W]
p['mth'] = 0.8	   # Mass of cell [Kg]

# LGChem provided,
# Note that 'E_De' is in the electrolyteDe function
p['E_kn'] = 37480#77840
p['E_kp'] = 39570#32780
p['E_Dsn'] = 42770#110800
p['E_Dsp'] = 37040#364.2
p['E_kappa_e'] = 34700 #11357


# Ambient Temperature
p['T_amb'] = 298.15 # [K]
p['T_ref'] = 298.15 # [K] for ElectrolyteACT

#==============================================================================
# Miscellaneous
#==============================================================================
p['R'] = 8.314472;      # Gas constant, [J/mol-K]
p['Faraday'] = 96485.3329  # Faraday constant [Coulombs/mol]
p['Area'] = 1.425      # Electrode current collector area [m^2]
p['alph'] = 0.5         # Charge transfer coefficients
p['t_plus'] = 0.45		# Transference number
p['brug'] = 1.8		# Bruggeman porosity
#==============================================================================
# Concentrations
#==============================================================================

p['c_s_n_max'] = 3.0e+04 # Max concentration in anode, [mol/m^3]
p['c_s_p_max'] = 4.5e+04 # Max concentration in cathode, [mol/m^3]
p['n_Li_s'] = 3.0 # Total moles of lithium in solid phase [mol]
p['c_e0'] = 1.0e3    # Electrolyte concentration [mol/m^3]

# Stoichimetry points
p['cn0'] = 0.03
p['cn100'] = 0.9 # x100, Cell SOC 100
p['cp0'] = 0.8   # y0, Cell SOC 0
p['cp100'] = 0.2 # y100, Cell SOC 100
#==============================================================================
# Discretization params
#==============================================================================
p['PadeOrder'] = 3


p['Nr'] = 30
p['delta_r_n'] = 1/float(p['Nr'])
p['delta_r_p'] = 1/float(p['Nr'])

p['Nxn'] = 100;
p['Nxs'] = 100;
p['Nxp'] = 100;
p['Nx'] = p['Nxn']+p['Nxs']+p['Nxp']

p['delta_x_n'] = 1 / float(p['Nxn'])
p['delta_x_s'] = 1 / float(p['Nxs'])
p['delta_x_p'] = 1 / float(p['Nxp'])

#==============================================================================
# Safety Constraint
#==============================================================================
p['volt_min'] = 2.7
p['volt_max'] = 4.2


def refPotentialAnode_casadi(theta):

    Uref = 0.194+1.5*np.exp(-120.0*theta) +0.0351*np.tanh((theta-0.286)/0.083) - 0.0045*np.tanh((theta-0.849)/0.119) - 0.035*np.tanh((theta-0.9233)/0.05) - 0.0147*np.tanh((theta-0.5)/0.034) - 0.102*np.tanh((theta-0.194)/0.142) - 0.022*np.tanh((theta-0.9)/0.0164) - 0.011*np.tanh((theta-0.124)/0.0226) + 0.0155*np.tanh((theta-0.105)/0.029)

    return Uref

def refPotentialCathode_casadi(theta):


    Uref = 2.16216+0.07645*np.tanh(30.834-54.4806*theta) + 2.1581*np.tanh(52.294-50.294*theta) - 0.14169*np.tanh(11.0923-19.8543*theta) + 0.2051*np.tanh(1.4684-5.4888*theta) + 0.2531*np.tanh((-theta+0.56478)/0.1316) - 0.02167*np.tanh((theta-0.525)/0.006)
    return Uref

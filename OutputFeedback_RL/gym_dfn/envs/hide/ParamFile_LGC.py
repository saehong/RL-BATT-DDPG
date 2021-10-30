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
p['L_n'] = 84.0e-6       # Thickness of negative electrode [m]
p['L_s'] = 15.4e-6       # Thickness of separator [m]
p['L_p'] = 77.5e-6     # Thickness of positive electrode [m]

L_ccn = 8e-6;    # Thickness of negative current collector [m]
L_ccp = 12e-6;    # Thickness of negative current collector [m]


# Particle Radii
p['R_s_n'] = 4.75e-06 # Radius of solid particles in negative electrode [m]
p['R_s_p'] = 8.75e-06 # Radius of solid particles in positive electrode [m]

# Volume fractions
p['epsilon_s_n'] = 0.702576579530290 # Volume fraction in solid for neg. electrode
p['epsilon_s_p'] = 0.675243501279972 # Volume fraction in solid for pos. electrode

p['epsilon_e_n'] = 0.24   # Volume fraction in electrolyte for neg. electrode
p['epsilon_e_s'] = 0.41	  # Volume fraction in electrolyte for separator
p['epsilon_e_p'] = 0.23   # Volume fraction in electrolyte for pos. electrode

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
p['sig_n'] = 50    # Conductivity of solid in neg. electrode, [1/Ohms*m]
p['sig_p'] = 0.05  # Conductivity of solid in pos. electrode, [1/Ohms*m]

#==============================================================================
# Kinetic params
#==============================================================================
p['R_f_n'] = 0 # [CCTA-Adaption case study: 1e-4]       # Resistivity of SEI layer, [Ohms*m^2]
p['R_f_p'] = 0 # [CCTA-Adaption case study: 1e-4]       # Resistivity of SEI layer, [Ohms*m^2]
#p.R_c = 2.5e-03;%5.1874e-05/p.Area; % Contact Resistance/Current Collector Resistance, [Ohms-m^2]

# Nominal Reaction rates
p['k_n0'] = 3.85e-05  # Reaction rate in neg. electrode, [(A/m^2)*(mol^3/mol)^(1+alpha)]
p['k_p0'] = 2.43e-06  # Reaction rate in pos. electrode, [(A/m^2)*(mol^3/mol)^(1+alpha)]


#==============================================================================
# Thermodynamic params
#==============================================================================

# Thermal dynamics
p['C_p'] = 1015    # Heat capacity, [J/kg-K]
p['R_th'] = 0.5   # Thermal resistance, [K/W]
p['mth'] = 0.834	   # Mass of cell [Kg]

# LGChem provided,
# Note that 'E_De' is in the electrolyteDe function
p['E_kn'] = 77840
p['E_kp'] = 32780
p['E_Dsn'] = 110800
p['E_Dsp'] = 364.2
p['E_kappa_e'] = 11357


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

p['c_s_n_max'] = 3.0542e+04 # Max concentration in anode, [mol/m^3]
p['c_s_p_max'] = 4.9521e+04 # Max concentration in cathode, [mol/m^3]
p['n_Li_s'] = 3.268434502946604 # Total moles of lithium in solid phase [mol]
p['c_e0'] = 1.0e3    # Electrolyte concentration [mol/m^3]

# Stoichimetry points
p['cn0'] = 0.032
p['cn100'] = 0.932 # x100, Cell SOC 100
p['cp0'] = 0.863   # y0, Cell SOC 0
p['cp100'] = 0.237 # y100, Cell SOC 100
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


def refPotentialAnode_casadi(theta):
 #Coefficients
    a1 =   1.237e+12
    b1 =     -0.3552
    c1 =     0.07266
    a2 =       427.5
    b2 =     -0.3951
    c2 =      0.1663
    a3 =     0.03499
    b3 =       0.134
    c3 =     0.04924
    a4 =     0.05214
    b4 =      0.1475
    c4 =     0.07708
    a5 =     0.04712
    b5 =      0.2752
    c5 =      0.1939
    a6 =     0.08932
    b6 =      0.6794
    c6 =       1.713
    a7 =     0.02026
    b7 =      0.4694
    c7 =     0.07461


    Uref = a1*np.exp(-((theta-b1)/c1)**2) + a2*np.exp(-((theta-b2)/c2)**2) + a3*np.exp(-((theta-b3)/c3)**2) + a4*np.exp(-((theta-b4)/c4)**2)+ a5*np.exp(-((theta-b5)/c5)**2) + a6*np.exp(-((theta-b6)/c6)**2)+ a7*np.exp(-((theta-b7)/c7)**2)

    return Uref

def refPotentialCathode_casadi(theta):
 # Coefficients
    a0 =       1.972
    a1 =      -3.414
    b1 =      0.7653
    a2 =      -2.667
    b2 =      0.7529
    a3 =      -1.629
    b3 =      0.5915
    a4 =     -0.8025
    b4 =      0.3694
    a5 =     -0.3005
    b5 =      0.1636
    a6 =    -0.08076
    b6 =      0.0495
    a7 =    -0.01273
    b7 =    0.006789
    w =       5.312

    Uref = a0 + a1*np.cos(theta*w) + b1*np.sin(theta*w) + a2*np.cos(2*theta*w) + b2*np.sin(2*theta*w) + a3*np.cos(3*theta*w) + b3*np.sin(3*theta*w) + a4*np.cos(4*theta*w) + b4*np.sin(4*theta*w) + a5*np.cos(5*theta*w) + b5*np.sin(5*theta*w) + a6*np.cos(6*theta*w) + b6*np.sin(6*theta*w) + a7*np.cos(7*theta*w) + b7*np.sin(7*theta*w)

    return Uref

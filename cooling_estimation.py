#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:48:25 2024

@author: mam
"""

import numpy as np

def diff_temp(heat_load, Cp, rho, flow_rate):
    """
    Calculate the difference of temperature of the VLTI coolant after absorbing
    a heat load.
    The specification is the variation of temperature to be lower than 8 degC.

    Parameters
    ----------
    heat_load : float
        Heat load in W.
    Cp : float
        Thermal capacity in J/kg/m3.
    rho : float
        Volumic mass in kg/m3.
    flow_rate : float
        Flow rate in m3/s.

    Returns
    -------
    dT : float
        Difference of temperature in K.

    """
    dT = heat_load / (Cp * rho * flow_rate)
    return dT

def flow_speed(flow_rate, diam):
    """
    Calculate the flow speed from the flow rate and the diameter of the pipes.
    The specification is the flow speed to be lower than 1.2 m/s.

    Parameters
    ----------
    flow_rate : float
        Flow rate in m3/s.
    diam : float
        Diameter of the pipe in metre.

    Returns
    -------
    flow_speed : float
        Flow speed in m/s.

    """
    fl_speed = 4 * flow_rate / (np.pi * diam**2)
    return fl_speed

def calculate_diff_pressure(flow_rate, rho, diam1, diam2):
    """
    Calculate the differential pressure (in Pa).

    Parameters
    ----------
    flow_rate : float
        Flow rate in m3/s.
    rho : float
        Volumetric mass (in kg/m3).
    diam1 : float
        Diameter of the hose (in metre).
    diam2 : float
        Diameter of the pipe of the device (in metre).
        
    Returns
    -------
    float
        Differential pressure (in Pa).

    """
    beta = diam2 / diam1
    dp = 8 / (np.pi**2 * diam2**4) * rho * flow_rate**2 * (1 - beta**4) # in Pascal
    return dp * 1e-5 # in bar 

# =============================================================================
# VLTI specifications
# =============================================================================
"""
VLTI lab has a total flow rate of 6L/min with 2 vailable ports.
Thus 3L/min per port.

The storage room has a total flow rate of 12L/min with 3 available ports.
Thus 4L/min each
"""
max_dT = 8. # in deg C
max_speed = 1.2 # in m/s
lab_flow_rate = 0.003 # Flow rate in m3/min
lab_flow_rate /= 60
storage_flow_rate = 0.004 # Flow rate in m3/min
storage_flow_rate /= 60.
min_diff_pressure = 0.8 # in Pa
max_diff_pressure = 2. # in Pa
hose_diam = 0.75 # in inche
hose_diam *= 0.0254 # in metre

# =============================================================================
# Coolant properties
# =============================================================================
Cp = 3190 # J/kg/m3
rho = 1092 # kg/m3

# =============================================================================
# C-Red One
# =============================================================================
"""
The minimal flow rate to cool the C-Red One is 3L/min.
"""
d = 24e-3 # C-Red One connector diameter
heat_load = 300 # in W

dT = diff_temp(heat_load, Cp, rho, lab_flow_rate)
print('Temperature variation for C-Red One (degC):', dT)
print('Within specification?', dT <= max_dT)

fl_speed = flow_speed(lab_flow_rate, d)
print('Flow speed for C-Red One (m/s):', fl_speed)
print('Within specification?', fl_speed <= max_speed)

dp = calculate_diff_pressure(lab_flow_rate, rho, hose_diam, d)
print('Differential pressure (bar):', dp)
print('Within specification?', (dp <= max_diff_pressure) & (min_diff_pressure <= dp))

# =============================================================================
# VLTI lab above-table cooling cabinets
# =============================================================================
heat_load = 170 # in W
dT = diff_temp(heat_load, Cp, rho, lab_flow_rate)
print('\nTemperature variation for HDL above-table cooling cabinet (degC):', dT)
print('Within specification?', dT <= max_dT)

# =============================================================================
# VLTI lab DM under-table cooling cabinets
# =============================================================================
heat_load = 190 # in W
dT = diff_temp(heat_load, Cp, rho, lab_flow_rate)
print('\nTemperature variation for DM under-table cooling cabinet 1 (degC):', dT)
print('Within specification?', dT <= max_dT)

# =============================================================================
# VLTI lab NOTT under-table cooling cabinets
# =============================================================================
heat_load = 300 # in W
dT = diff_temp(heat_load, Cp, rho, lab_flow_rate)
print('\nTemperature variation for NOTT under-table cooling cabinet 1 (degC):', dT)
print('Within specification?', dT <= max_dT)

# =============================================================================
# Storage room NOTT cryo compressor
# =============================================================================
"""
Flow rate of the cryo compressor must be 9L/min minimum.
"""
heat_load = 30 # in W
dT = diff_temp(heat_load, Cp, rho, 0.009/60)
print('\nTemperature variation for NOTT Cryo compressor (degC):', dT)
print('Within specification?', dT <= max_dT)



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

# =============================================================================
# Coolant properties
# =============================================================================
Cp = 3190 # J/kg/m3
rho = 1082 # kg/m3

# =============================================================================
# C-Red One
# =============================================================================
"""
The minimal flow rate to cool the C-Red One is 3L/min.
"""
d = 24e-3 # C-Red One connector diameter
heat_load = 400 # in W

dT = diff_temp(heat_load, Cp, rho, lab_flow_rate)
print('Temperature variation for C-Red One:', dT)
print('Within specification?', dT <= max_dT)

fl_speed = flow_speed(lab_flow_rate, d)
print('Flow speed for C-Red One:', fl_speed)
print('Within specification?', fl_speed <= max_speed)

# =============================================================================
# VLTI lab above-table cooling cabinets
# =============================================================================
heat_load = 170 # in W
dT = diff_temp(heat_load, Cp, rho, lab_flow_rate)
print('\nTemperature variation for above-table cooling cabinet:', dT)
print('Within specification?', dT <= max_dT)

# =============================================================================
# VLTI lab under-table cooling cabinets
# =============================================================================
heat_load = 170 # in W
dT = diff_temp(heat_load, Cp, rho, lab_flow_rate)
print('\nTemperature variation for under-table cooling cabinet 1:', dT)
print('Within specification?', dT <= max_dT)

# =============================================================================
# Storage room NOTT cryo compressor
# =============================================================================
"""
Flow rate of the cryo compressor must be 9L/min minimum.
"""
heat_load = 40 # in W
dT = diff_temp(heat_load, Cp, rho, 0.009/60)
print('\nTemperature variation for NOTT Cryo compressor:', dT)
print('Within specification?', dT <= max_dT)



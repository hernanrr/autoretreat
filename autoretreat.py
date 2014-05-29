#!/usr/bin/env python

# Built in modules
from __future__ import division
import sys
import os
import pdb
import math

# Third party modules
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# Some Constants

g = 9.81 # Gravity,[ m / s ** 2 ]
nu =  1.0e-6 # Dynamic viscosity of water [ m **2 / s ]
rho = 1000 # Density of water [ kg / m ** 3 ]



# Input paramenters. Eventually this will read from a file.
# x_shore = 5000 # [ m ]
# dx = 200 # [ m ]

def nodes_in_domain(x_shore, dx):
    """Computes the number of nodes on the computational domain (fluvial reach).

    Args: 
        x_shore: x_coordinate of the shoreline. [ L ]
        dx : Computational spatial step [ L ]

    Returns: 
        N : Number of nodes required to fully describe the fluvial reach. [ 1 ]
        dx_shore : Computational spatial step of the shoreline [ L ]

    Comments: 
        The computational domain needs to have one more node than the number of 
        reaches, to be able to properly define both endpoints of every reach. 

        N is rounded, so that N * dx != x_shore. This is the expected
        behavior. The idea is that the shoreline does not need to be at a
        distance dx from the previous node. To account for this, an additional
        node is added to represent the shoreline only if its x-coordinate is not
        a multiple of dx.

    """
    N = np.array( math.ceil( x_shore / dx ) + 1)
    dx_shore = x_shore % dx
    return N, dx_shore


def init_domain(N, dx_shore, dx):
    """Specifies de number of computational nodes on the delta top.

    Args: 
        x_shore: x_coordinate of the shoreline. [ L ]
        dx : Computational spatial step [ L ]

    Returns:
        dx : Array with size equal to the number of computational 
             nodes in the domain and values equal to the distance from
             the previous computational node. 

    Comments: 
        The resize function is only applied if the number of computational nodes
        needed is different from the length of the dx-array. If they are the
        same, then dx is not resized. 

        If the shoreline node does not fall on a regular grid node, then the 
        last element of dx is set to dx_shore. Otherwise, nothing happens. This
        scenario is unlikely. 
    """
    dx = np.resize(dx, N)
    if  dx_shore != 0:
        dx[-1] = dx_shore
    return dx 

def initialize(dx, x_shore, eta_shore, eta_toe, S_d, S_b, S, Q_w, B0):
    """Initialize the variables for the simulation.

    Args:

    Returns: 
        eta_b : array of z-coordinate of the basement at every node. Has the
                same size as dx. [ L ] 

    Comments: 
    """
    # First thing we should do is specify our computational domain.
    N, dx_shore = nodes_in_domain(x_shore, dx)
    N_old = N.copy()
    dx = init_domain(N, dx_shore, dx)
    x_shore = x(dx)[-1]
    # Then we compute the basement elevation and the location of the delta toe.
    eta_b = np.resize(eta_toe, N)
    eta_b, x_toe = init_basement(dx, eta_shore, eta_toe, eta_b, S_d, S_b)
    # Next, we compute the initial fluvial profile
    eta, S = init_flumen(dx, eta_shore, S)
    # Instantiate a water-depth array for the domain.
    H = np.zeros_like(dx)
    # Compute unit flow rate
    qw = unit_flowrate(Q_w, B0) # [ L**2 / T ]
    # Compute critical depth for the section.
    Hc = critical_flow(qw)
    # Redefine B0 as a vector, to have that information on every node.    
    B0 = np.resize(B0, N)
    # Instantiate a sediment transport capacity array for the domain
    qt = np.zeros_like(dx)
    return (N, N_old, dx_shore, dx, x_shore, eta_b, x_toe, eta, S, H, Hc, qw,
            B0, qt)

def copy_old_values(N, dx, eta_b, eta, S):
    """Keeps memory of the values of the previous timestep"""
    N_old = N.copy()
    dx_old = dx.copy()
    eta_b_old = eta_b.copy()
    eta_old = eta.copy()
    S_old = S.copy()
    return N_old, dx_old, eta_b_old, eta_old, S_old
    

def x(dx):
    """Returns the distance of a computational node form the origin.

    Args: 
        dx : array of distances between computational nodes.

    Returns: 
        x : Distance of the computational node from the origin.

    Comments: 
        The first node is assigned a non-zero dx because the model assumes that
        sediment is feed from a ghost-node located upstream of the first node of 
        the computational domain at a distance dx. Despite this, the ghost-node
        is not part of the computational domain, so the cummulative incremental 
        distances need to be adjusted so that x = 0 actually corresponds to the 
        first computational node. 
    """
    return np.cumsum(dx) - dx[0]

def init_basement(dx, eta_shore, eta_toe, eta_b, S_d, S_b):
    """Initializes the basement of the delta.

    Args:
        dx : Array of distance between computational nodes [ L ]
        eta_shore : Z-coordinate of the delta toe [ L ]
        eta_toe : Z-coordinate of the delta toe [ L ]
        eta_b : array of z-coordinate of the basement at every node. 
        S_d : Slope of the delta front [ L / L ] = [ 1 ] 
        S_b : Slope of the basement. [ L / L ] = [ 1 ]

    Returns: 
        x_toe : x-coordinate of the delta toe. [ L ]

    Comments:
        The basement is computed on the computational domain; however, the way
        the computational domain is set up so far, we do not have the basement
        elevation at the shoreline. To obtain it, the x-coordinate of the toe
        needs to be computed. 
        
        Given the (x, z)-coordinates of the shoreline, the z-coordinate of the
        delta toe and the slope of the delta front, the x-coordinate of the
        delta toe can be determined. 

        With the (x, z)-coordinates of both the shoreline and delta toe, and
        the basement slope, the elevation of the basement at the shoreline is
        determined.

        Once the basement elevation at the shoreline is computed, the rest of
        the basement elevations can be propagated from that point upstream,
        given the basement slope. 

    """
    # Find the x-coordinate of the delta toe.
    x_toe = find_x(eta_toe, S_d, x(dx)[-1], eta_shore)
    # Find basement elevation at shoreline x-coordinate.
    eta_b[-1] = find_y(x(dx)[-1], S_b, x_toe, eta_toe)
    # Populate the basement elevation array.
    eta_b = find_y( x(dx), S_b, x(dx)[-1], eta_b[-1] )
    return eta_b, x_toe

def find_x(y, m, x0, y0):
    """Finds x in the point-slope line equation. Slope has changed sign."""
    return ( x0 - (y - y0) / m  )

def find_y(x, m, x0, y0):
    """Finds y in the point-slope line equation. Slope has changed sign."""
    return ( y0 - m * (x - x0) )
    
def init_flumen(dx, eta_shore, S):
    """Initializes the bed of the fluvial reach.
    
    Args: 
        dx : Array of distance between computational nodes. [ L ]
        eta_shore : z-coordinate of the delta shoreline. [ L ]
        S : Slope of the fluvial reach bed. [ L / L ] = [ 1 ]

    Returns: 
        eta : Array of z-coordinate of the fluvial reach at every node. Has the
              same size as dx. [ L ] 
        S : Array of slope of the fluvial reach bed. [ L / L ] = [ 1 ]

    """
    x_shore = x(dx)[-1]
    eta, eta[-1] = np.zeros_like(dx), eta_shore
    eta = find_y((x(dx)), S, x_shore, eta[-1])
    S = np.full_like(dx, S)
    return  eta, S

def unit_flowrate(Q_w, B0=1.0):
    """Returns the flow rate per unit width of the channel.
    
    Args:
        Q_w : Volumetric flowrate into the channel[ L ** 3 / T ]
        B0 : Cross-sectional bottom width of channel [ L ]

    Returns: 
        qw : Unit flowrate into the channel [ L ** 2 / T ]
    """
    return Q_w / B0

def critical_flow(qw):
    """Returns the critical flow depth at the section. """
    return (qw ** 2 / g) ** (1./3.)

def normal_flow(S, Cf, qw):
    """Returns the normal flow depth at the reach using a Chezy formulation
    
    Args:
        S : Slope of the fluvial reach bed. [ L / L ] = [ 1 ]

    Returns:
        H_n : Array of normal depth for the reach
    """
    return (Cf * qw**2 /( g * S) )**(1./3.)

def chezy_friction_coefficient(Cz):
    """Returns the chezy friction coefficient given the dimensionless chezy
       resistance coefficient.

    Args:
        Cz : Dimensionless Chezy resistance coefficient. [ 1 ]

    Returns: 
        Cf = Chezy friction factor [ 1 ] 
"""
    return 1. / Cz ** 2

def flow_area(H, B0=1.0):
    """returns the flow area of a rectangular channel. """
    return H * B0

def flow_velocity(Q_w, H, B0=1.0):
    """returns the depth-averaged flow velocity at the cross-section"""
    return Q_w / flow_area(H, B0)

def froude_number(Q_w, H, B0=1.0):
    """returns the froude number for the flow. """
    return flow_velocity(Q_w, H, B0 ) / np.sqrt(g * H)

def friction_slope(Cz, Fr):
    """Returns the friction slope of the cross-section."""
    return chezy_friction_coefficient(Cz)* Fr ** 2

def gradually_varied_flow_eq(H, S, Q_w, B0, Cz):
    """Returns the Gradually Varied Flow function value

    Args:
        eta : array of z-coordinate of the fluvial reach at every node. [ L ]
        dx : Array of distance between computational nodes. [ L ]
        H : Array containing the water surface elevation for every
            computational node in the grid.
        S : Slope of the fluvial reach bed. [ L / L ] = [ 1 ]

    Returns:
        dH : Increment of the water surface elevation. 

    """
    Fr = froude_number(Q_w, H, B0)
    numerator = ( S -  friction_slope(Cz, Fr) )
    denominator = ( 1. - Fr ** 2 )
    return numerator / denominator


def backwater(xi_d, dx, eta, H, S, Q_w, B0, Cz):
    """Solves the backwater equation using the Euler Method.

    Args: 
        xi_d : Water surface elevation at the downstream control. [ L ]
        dx : Array of distance between computational nodes. [ L ]
        eta : array of z-coordinate of the fluvial reach at every node.

    Returns: 
        H : Array containing the water surface elevation for every
            computational node in the grid.

    Comments: 
    
    """
    # Rebind the GVF equation to a more reasonable name
    F = gradually_varied_flow_eq
    # Generate a sequence of integers from 1 to N - 1
    for i in xrange(1, len(dx)):
        
        # Predictor step
        # 1. Compute the GVF equation for the predictor value from previous H.
        dhdx_p = F((H[::-1])[i-1], (S[::-1])[i], Q_w, (B0[::-1])[i], Cz) 
        # 2. Compute the predictor value
        predictor = dhdx_p * (dx[::-1])[i-1]
        # 3. Find the predicted H.
        (H[::-1])[i] = (H[::-1])[i-1] - predictor
        # Corrector step
        # 4. Compute the GVF from the predicted value.
        dhdx_c = F((H[::-1])[i], (S[::-1])[i], Q_w, (B0[::-1])[i], Cz) 
        # 5. Compute corrector term:
        corrector = 1./2. * ( dhdx_c + dhdx_p) * (dx[::-1])[i-1]
        # 6. Compute the corrected water surface elevation.
        (H[::-1])[i] = (H[::-1])[i-1] - corrector
    return H

def base_level(xi_d, t=0.0):
    """Returns the sea elevation for the downstream boundary condition.

    Args:
        xi_d : Initial downstream sea level elevation [ m ]

    Returns: 
        xi_d : Downstream sea level elevation at time t [ m ]

    Comments: 
        The downstream boundary condition can be specified as s function of
        time. When this function was writen, it assumed that the function was
        constant, thus returning the input value. 
    """
    return xi_d

def subsidence(sigma, t=0.0):
    """Returns the subsidence rate at time t.

    Args: 
        sigma : Initial subsidence rate. [ L / T ]c

    Returns:
        sigma : subsidence rate. [ L / T ]
    
    Comments:
        As the function is now, it returns a constant subsidence rate throughout
        the simulation. It can be modified to return a time-varying subsidence
        rate. Initial sigma is passed to the function to allow
    """
    return sigma

def mass_to_vol_feedrate_converter(Qt, R, rho):
    """Converts the input sediment mass feed rate into volumetric feed rate.
    
    Args:
        Qt : Mass sediment feed rate [ M / T ]

    Returns: 
        qt : Volumetric sediment feed rate [ L ** 3 / T ]

    Comments: 
        The mass sediment feed rate is input in cubic meters per second.  The
        conversion yields Tons per year.
    
    """
    return (Qt * 1000) / ( rho * (R + 1) ) * 1 / ( 365.25 * 24 * 60 * 60 )


def vol_to_mass_feedrate_converter(qt, R, rho):
    """Converts the input sediment volume feed rate into mass feed rate.
    
    Args:
        qt : Mass sediment feed rate [ M / T ]

    Returns: 
        Q_t : Volumetric sediment feed rate [ L ** 3 / T ]

    Comments: 
        The mass sediment feed rate is input in Tons per year. The conversion
        yields cubic meters per second. 
    
"""
    return (qt * rho * ( R + 1 )) / 1000  * ( 365.25 * 24 * 60 * 60 ) 

def engelund_hansen(Cf, tau_star, tau_c_star = 0.0):
    """Returns the Dimensionless Einstein Number based on the Engelund and
    Hansen formulation.

    Args: 
        tau_star: Dimensionless boundary shear stress at the bed. [ 1 ] 
        tau_c_star: Critical Dimensionless boundary shear stress at the bed. [1]
    Returns:
        qb_star : Dimensionless Einstein Number. [ 1 ]

    """
    qb_star = 0.05 / Cf * tau_star^(5.2.)
    return qb_star, tau_c_star
def ashida_michiue(tau_star, tau_c_star=0.05):
    """Returns the Dimensionless Einstein Number based on the Ashida & Michiue
       formulation. 

    Args:
        tau_star: Dimensionless boundary shear stress at the bed. [ 1 ] 
        tau_c_star: Critical Dimensionless boundary shear stress at the bed. [1]
        
    Returns:
        qb_star : Dimensionless Einstein Number. [ 1 ]
    """
    qb_star = 17 * ( tau_star - tau_c_star ).clip(0) * ( np.sqrt(tau_star) -
                                                 np.sqrt(tau_c_star) ).clip(0) 
    return qb_star, tau_c_star

def wong_parker(tau_star, tau_c_star=0.0495):
    """Returns the dimensionless Einstein Number based on the Wong & Parker 
       formulation.

    Args:
        tau_star: Array, Dimensionless boundary shear stress at the bed. [ 1 ] 
        tau_c_star: Critical Dimensionless boundary shear stress at the bed. [1]
        
    Returns:
        qb_star : Array with dimensionless Einstein Number. [ 1 ]

    """
    qb_star =  3.97 * ((tau_star -  tau_c_star).clip(0) ) ** (3./2.) 
    return  qb_star, tau_c_star

def shields_number(U, Cf, D, R):
    """Return the dimensionless Shields Number, shear stress at the channel bed.

    Args: 
        U : Depth-averaged flow velocity. [ L / T  ]
        Cf = Chezy friction factor [ 1 ] 
        D : Grain size of sediment [ L ]
        R : Submerged specific gravity of sediment [ 1 ]

    Returns:
        tau_star: shear stress at every node of the bed domain.
    """
    return ( Cf * U ** 2 ) / ( R * g * D )

def bedload_transport_capacity(tau_star, tau_c_star, qt, R, D, qb_star):
    """Computes the bedload transport capacity.

    Args:
        tau_star: Array, Dimensionless boundary shear stress at the bed. [ 1 ] 
        tau_c_star: Critical Dimensionless boundary shear stress at the bed. [1]
        R : Submerged specific gravity of sediment [ 1 ]
        D : Grain size of sediment [ L ]
        qb_star : Dimensionless Einstein Number. [ 1 ]
   
    Returns: 
        qt : array of bedload transport capacity. [ L ** 2 / T ]
    """
    qt = np.sqrt( R * g * D ) * D * qb_star
    return qt

def exner_equation(dt, dx, eta, qt, lambda_p, I_f=1.0, sigma=0.0):
    """Updates the bed elevation using the Exner equation.

    Args:
        dt : Computational time step. [ T ]
        dx : Array of distance between computational nodes. [ L ]
        eta : array of z-coordinate of the fluvial reach at every node.
        qt : array of bedload transport capacity. [ L ** 2 / T ]
        lambda_p : Sediment porosity [ 1 ]
        I_f : Flood intermitency factor. [ 1 ]. Defaults to I_f = 1.0
        sigma : Subsidence rate [ L / T ]. Defaults to sigma = 0.0

    Returns: 
        eta : array of z-coordinate of the fluvial reach at every node.
    """
    qt_u = qt[:-1]
    qt_d = qt[1:]
    # Set the sediment-feed boundary condition
    eta = eta + I_f * (dt /(1 - lambda_p)) * ( qt_u - qt_d ) /dx - sigma * dt
    return eta

def update_shoreline(dt, dx, eta, eta_old, qt,  lambda_p, I_f=1.0, sigma=0.0):
    """Function that calls the Exner equation on the shoreline.
    Args:
        dt : Computational time step. [ T ]
        dx : Array of distance between computational nodes. [ L ]
        eta : array of z-coordinate of the fluvial reach at every node.
        qt : array of bedload transport capacity. [ L ** 2 / T ]
        lambda_p : Sediment porosity [ 1 ]
        I_f : Flood intermitency factor. [ 1 ]. Defaults to I_f = 1.0
        sigma : Subsidence rate [ L / T ]. Defaults to sigma = 0.0

    Returns: 
        eta : array of z-coordinate of the fluvial reach at every node. [ L ]
        eta_shore : z-coordinate at the shoreline node. [ L ]
        d_eta_shore_dt : Time rate of change of shoreline elevation. [ L / T ]
    """
    # At last node of the computational domain...
    # but first we store the previous shoreline elevation
    eta_shore_old = eta_old[-1]
    eta[-1] = exner_equation(dt, dx[-2:].sum(), eta[-1], np.array([qt[-3],
                                                                   qt[-1]]),
                             lambda_p,I_f, sigma)
    eta_shore = eta[-1]
    d_eta_shore_dt = np.diff([eta_shore_old, eta[-1]]) / dt
    return eta, eta_shore, d_eta_shore_dt    

def update_eta(dt, dx, eta, eta_shore, qt, qt_f, lambda_p, N, N_old, I_f=1.0,
               sigma=0):
    """Auxiliary function that calls the Exner equation on select nodes.
   
    Args:
        dt : Computational time step. [ T ]
        dx : Array of distance between computational nodes. [ L ]
        eta : Array of z-coordinate of the fluvial reach atn every node.
        qt : Array of bedload transport capacity. [ L ** 2 / T ]
        qt_f : Sediment feed rate at upstream boundary. [ L ** 2 / T ]
        lambda_p : Sediment porosity [ 1 ]
        I_f : Flood intermitency factor. [ 1 ]. Defaults to I_f = 1.0
        sigma : Subsidence rate [ L / T ]. Defaults to sigma = 0.0

    Returns: 
        eta : array of z-coordinate of the fluvial reach at every node. [ L ]
        d_eta_shore_dt : Time rate of change of shoreline elevation. [ L / T ]
    """
    # At first node of the computational domain. 
    eta[0] = exner_equation(dt, dx[0], eta[0], np.array( [ qt_f, qt[0] ] ),
                            lambda_p, I_f, sigma) 
    # At interior nodes except second-to-last through end, before the remesh. 
    eta[1:N_old-2] = exner_equation(dt, dx[1:N_old - 2], eta[1:N_old - 2],
                                    qt[0:N_old-2], lambda_p, I_f, sigma)
    # Determine if the domain grew, shrank or stayed the same. 
    grows = N > N_old
    shrinks = N < N_old
    # If the domain grew, there are at least two nodes for which to compute eta.
    # It the domain shrank, nothing needs to be done other than resize. 
    # If the domain stayed the same, only one node needs to be computed.
    if grows or shrinks:
        eta = np.resize(eta, N)
        eta[-1] = eta_shore
    if not shrinks:
        slope =  ( eta[(N_old-1) - 2] - eta[-1] ) / ( dx[N_old-1 - 1: ].sum() )
        eta[N_old - 2:-1] = find_y(x(dx)[N_old - 2:-1], slope, x(dx)[-1],
                                   eta[-1])
    return eta


def shock_condition(dt, dx, x_shore, x_toe, d_eta_shore_dt, qt, lambda_p, S,
                    S_d, I_f=1.0, sigma=0.0):
    """Computes the shoreline migration rate and horizontal displacement.
    
    Args: 
        dx : Array of distance between computational nodes. [ L ]
        qt : array of bedload transport capacity. [ L ** 2 / T ]
        lambda_p : Sediment porosity [ 1 ]
        d_eta_shore_dt : Time rate of change of shoreline elevation. [ L / T ]

    Returns: 
        x_shore : New position of the shoreline. [ L ]
        c_shore_x : Migration rate of the shoreline in the x-direction. [ L ]
        
    """
    c_shore_x = ( ( ( ( I_f * qt[-1] ) / ( (1 - lambda_p) * (x_toe - x_shore )
                                       ) ) - sigma - d_eta_shore_dt ) * ( 1 / (
                                           S_d - S) ) ) 
    x_shore = x_shore + c_shore_x * dt 
    return c_shore_x, x_shore

def continuity_cond(dt, d_eta_shore_dt, x_toe, c_shore_x, S, S_d, S_b, sigma=0.):
    """Coordinate of delta toe base on a sediment continuity condition.

    Args: 
        dt : Computational time step. [ T ]
        d_eta_shore_dt : Time rate of change of shoreline elevation. [ L / T ]
        c_shore_x : Migration rate of the shoreline in the x-direction. [ L / T ]
        S : Slope of the fluvial reach bed. [ L / L ] = [ 1 ]
        S_d : Slope of the delta front. [ L / L ] = [ 1 ]
        S_b : Slope of the basement. [ L / L ] = [ 1 ]
        sigma : Subsidence rate [ L / T ]. Defaults to sigma = 0.0

    Returns: 
        c_toe_x : Migration rate of the delta toe in the x-direction. [ L ]
        x_toe : New x-position of the delta toe. [ L ]

    """
    numerator = d_eta_shore_dt - c_shore_x * (S - S_d ) + sigma 
    denominator = S_d - S_b 
    c_toe_x = numerator / denominator
    x_toe = x_toe + c_toe_x * dt
    return c_toe_x, x_toe

def remesh(dx, x_shore, N_old):
    """Remeshes the domain if needed.

    Once the the shoreline has moved, either seaward or landward, the position
    of the node representing the shore may fall outside of the domain of the
    previous node representing the shore. This means that nodes need to be added
    or removed from the computational domain to properly capture the new
    shoreline position and that a new shoreline distance from the previous node
    must be assigned to the new shoreline node. 

    Args: 
        dx : Array of distance between computational nodes. [ L ]
        x_shore : New position of the shoreline. [ L ]

    Intermediate args : 
        N_old, N_new: Number of computational nodes in current and required in 
        next time step. [ 1 ]
        grows, shrinks : Booleans specifying if the computational domain grows
        or shrinks respectively. 

    Returns: 
        N : Number of nodes in the domain.
        dx_shore : Computational spatial step of the shoreline [ L ]

    Comments: Grows and shrinks cannot be related by 'shrink = (not grows)' or
        'grows = (not shrink)' because these relations do not capture the
        possibility that the domain may stay the same (neither grow nor shrink). 

    """

    # Then we compute the number of nodes needed to fit the new domain
    N, dx_shore = nodes_in_domain(x_shore, dx[0])
    # Next, we determine if the domain grew, shrank or stayed the same. 
    grows = N > N_old
    shrinks = N < N_old
    # After, we redimension the array if necessary
    if grows or shrinks:
        dx = np.resize(dx, N)
        if grows:
            dx[N_old - 1:-1] = dx[0]
    # With the redimensioned array, we set the new shoreline spatial step.
    dx[-1] = dx_shore
    # Congratulations! You now have a remeshed domain. Time to remesh all the 
    # other arrays in the domain. 
    return dx, N

def update_basement(dx, x_toe, N, N_old, eta_b, S_b, dt, sigma=0):
    """Updates the basement. 

    Args: 
        dx : 
        N :
        N_old :
        eta_b : array of z-coordinate of the basement at every node.  [ L ]  
        sigma : Subsidence rate. [ L / T ]

    Returns:
        eta_b : array of z-coordinate of the basement at every node.  [ L ]  

    Comments:
        The way the find_y function works, it recomputes the value for the
        entire array. A x-coord array would be needed to slice the useful parts
        of the arrays for calculations. 
        By setting the known x and eta values for the find_y function to the
        first values of the array, we allow for the possibility to shrink the
        array to the minimum possible. 

    """
    grows = N > N_old
    shrinks = N < N_old
    # Resize the eta_b array to include the added nodes, if necessary
    if grows or shrinks:
        eta_b = np.resize(eta_b, N)
    # Populate the basement elevation array.
    eta_b, eta_toe = find_y( [x(dx), x_toe], S_b, x(dx)[0], eta_b[0] )
    # Apply the subsidence rate where appropriate.
    eta_b, eta_toe = eta_b - sigma * dt, eta_toe - sigma * dt
    return eta_b, eta_toe

def compute_water_surface(t, Q_w, qw, B0, S, Cf, Cz, dx, H, Hc, eta, N, N_old, xi_d0):
    """Computes the water surface elevation"""

    # At this point we can compute the water surface elevation.
    # First we verify that the channel slope is mild, comparing the depths of 
    # normal and critical flow at the computational nodes.
    Hn = normal_flow(S, Cf, qw)
    # Proceed if Hn > Hc, Otherwise halt and exit the simulation. 
    if t==0 and np.greater(Hc, Hn).any():
         sys.exit("The channel slope is not mild. The flow is supposed to " +
               " be Froude subcritical on the entire fluvial reach.")
    # Compute the downstream boundary condition
    xi_d = base_level(xi_d0, t)
    # Resize the arrays if needed
    grows = N > N_old
    shrinks = N < N_old
    # Resize the H array to include the added nodes, if necessary
    if grows or shrinks or t==0:
        H = np.resize(H, N)
    # Create the water depth array; set the downstream bounday condition.
    H[-1] = xi_d - eta[-1]
    # Compute the water surface profile using the backwater formulation.
    pdb.set_trace()
    H = backwater(xi_d, dx, eta, H, S, Q_w, B0, Cz)
    return H, Hn

def flow(t, Q_w, qw, B0, S, Cf, Cz, dx, H, Hc, eta, N, N_old, xi_d, D, R):
    """Auxiliary function that computes the flow parameters."""
    # Compute the water surface profile for the given bed.
    H, Hn = compute_water_surface(t, Q_w, qw, B0, S, Cf, Cz, dx, H, Hc, eta, N,
                                  N_old, xi_d)
    # Let's compute the flow velocity, to get stresses
    U = flow_velocity(Q_w, H, B0)
    # Compute the dimensionless shear stress at the bed for every node.
    tau_star = shields_number(U, Cf, D, R)
    return H, Hn, U, tau_star

def load(t, dx, N, N_old, qt, Cf, tau_star, R, D):
    """Auxiliary function to compute load parameters."""
    # Resize the qt array to include the added nodes, if necessary
    grows = N > N_old
    shrinks = N < N_old
    if grows or shrinks:
        qt = np.resize(qt, N)    
    # and an array for the dimensionless sediment transport rate:
#    qb_star, tau_c_star = wong_parker(tau_star)
    qb_star, tau_c_star = engelund_hansen(Cf, tau_star)
    # OK! Now we can compute the bedload transport capacity!
    qt = bedload_transport_capacity(tau_star, tau_c_star, qt, R, D, qb_star)
    return qt

def update_bed(dt, dx, x_shore, x_toe, eta, eta_old, S, S_d, S_b, qt, qt_f,
               lambda_p, N_old, I_f=1.0, sigma=0):
    """"""
    # We compute the Exner equation at the shoreline, skipping the second-to-
    # -last node when taking the difference, to prevent dividing by a number
    # possibly too close to zero.
    eta, eta_shore, d_eta_shore_dt = update_shoreline(dt, dx, eta, eta_old, qt,
                                                      lambda_p, I_f, sigma)
    # With the updated shoreline elevation, we implement the shock condition to
    # obtain the new horizontal position of the shoreline. 
    c_shore_x, x_shore = shock_condition(dt, dx, x_shore, x_toe, d_eta_shore_dt,
                                         qt, lambda_p, S[-1], S_d, I_f, sigma)
    # The position of the delta toe is then computed by means of continuity. 
    c_toe_x, x_toe = continuity_cond(dt, d_eta_shore_dt, x_toe, c_shore_x,
                                     S[-1], S_d, S_b, sigma)
    # The new x_shore might make the computational domain grow or shrink. This
    # means that we may need to add or remove nodes from our arrays.
    dx, N = remesh(dx, x_shore, N_old)
    # At this point, we compute the time-rate of change of the elevation of the
    # fluvial reach using the Exner equation.
    eta = update_eta(dt, dx, eta, eta_shore, qt, qt_f, lambda_p, N, N_old, I_f,
                     sigma)
    return dx, eta, x_toe, N

def update_channel_width(B0, N, N_old):
    """Resizes channel-width array to be the same size as the rest of the
    domain.

    Args:

    Returns:
    """
    grows, shrinks = N > N_old, N < N_old
    if grows or shrinks:
        B0 = np.resize(B0, N)
    return B0
    

def update_bedslope(dx, eta):
    """Computes the slope of the fluvial reach.

    Args: 
        dx : Array of distance between computational nodes. [ L ]
        eta : Array of z-coordinate of the fluvial reach at every node.
        S : Slope of the fluvial reach bed. [ L / L ] = [ 1 ]

    Returns: 
        S : Array of slope of the fluvial reach bed. [ L / L ] = [ 1 ]

    Comments: 
        The slope array is re-created every time. Ideally, it would just be
        filled at each pass and remeshed when the domain changed size. 

    """
    # Compute the slope between the computational nodes. 
    S = - np.diff(eta) / dx[1:]
    # Set slope of shoreline equal to slope of the previous computational node.
    S = np.append(S, S[-1])
    return S

def update_delta_slope(x_shore, x_toe, eta_shore, eta_toe):
    """Computes the slope of the delta front."""
    return (eta_shore - eta_toe) / (x_toe - x_shore)
    
def update_domain(dt, dx, x_toe, N, N_old, eta, eta_b, S_b, B0, sigma=0):
    """"""
    # Specify shoreline.
    x_shore, eta_shore = x(dx)[-1], eta[-1]
    # Update the basement. 
    eta_b, eta_toe = update_basement(dx, x_toe, N, N_old, eta_b, S_b, dt,
                                     sigma)
    # Update bedslope.
    S = update_bedslope(dx, eta)
    # Update slope of delta front
    S_d = update_delta_slope(x_shore, x_toe, eta_shore, eta_toe)
    B0 = update_channel_width(B0, N, N_old)
    return eta_b, eta_toe, S, S_d, B0, x_shore, x_toe

def output_times(m_prints, m_to_end):
    """Creates an array with the times at which the output will be saved.

    Args: 
        m_prints (type: int) : Number of outputs to save. 
        m_to_end (type: int) : Number of iterations in the simulation. 

    Returns:
        i_out (type: array) : Iterations at which output is saved.

    """
    return ( m_to_end - 1 ) / (m_prints - 1 ) * np.arange(m_prints)

def main():
    """This is the main routine of the program. Obviously."""
    # I am not a terribly clever person, so I'll talk my way through this.
    # Hopefully, as you read this, you will be able to make sense of 
    # my logic and my madness. 

    # First, some file handling: 
    home = os.path.expanduser("~")
    savepath = (home + '/Documents/SCarolina-classes/Spring-2014/delta-autoretreat/')
    dirname = datetime.now().strftime("%Y%m%d-%H%M%S")
    wd = os.path.join(savepath, dirname)

    # Input paramenters. Eventually this will read from a file.
    x_shore = 10000.0    # [ m ]
    eta_shore = 3.    # [ m ]
    eta_toe = np.float_(0.)    # [ m 
    S = np.float_(2.5e-4)    # [ 1 ] Initial Slope of the fluvial reach
    S_d = np.float_(2.e-1)    # [ 1 ] Initial Slope of the delta front
    S_b = np.float_(0.0) # 1e-4    # [ 1 ] Slope of the basement
    dx = np.float_(500)    # [ m ]
    dt = 86400 * 0.182625    # [ t ] Time step
    sim_time = 15 * 365.25 * 86400 # [ T ] Simulation time, in years to seconds.
    Cz = 15.0     # Chezy dimensionless resistance coefficient
    I_f = 1.    # Flood intermitency factor
    B0 = np.float_(1.)    # [ m ] Channel width
    Q_w = 6.    # [ m ** 3 / s ] Water flowrate into system
    xi_d = 8.5    # [ m ] Downstream water surface elevation.
    qt_f = 83627.64   # [ tons / y ] Sediment feed rate. (0.001 m2/s)
    D = 0.5 #2.0    # Grain size of sediment [ mm ]
    R = 1.65 # Submerged specific gravity of sediment [ 1 ]
    lambda_p = 0.4
    sigma = 0.0    # Redefine this as a function. [ L / T ]
    m_prints = 7    # Number of equally spaced printouts
    
    # Computed constants
    Cf = chezy_friction_coefficient(Cz) # [ 1 ]
    D = D / 1000.0    # Convert to meters
    qt_f_vol = mass_to_vol_feedrate_converter(qt_f, R, rho)
    qt_f = unit_flowrate(qt_f_vol, B0)
    m_to_end = np.int_(np.ceil(sim_time / dt) + 1)
    # ------------------------------------------------- End of input paramenters
    # Specify when 
    i_out = output_times(m_prints, m_to_end)
    # Initialize the model
    N, N_old, dx_shore, dx, x_shore, eta_b, x_toe, eta, S, H, Hc, qw, \
        B0, qt= initialize(dx, x_shore, eta_shore, eta_toe, S_d, S_b, S, Q_w,
                             B0)
    S_old = S.copy()
    dx_old = dx.copy()
    qt_old = qt.copy()
    U = np.zeros_like(H)
    # Begin time loop
    for i in xrange(m_to_end):
        pdb.set_trace()
        # What is the time?
        t = i * dt
        # We are currently at the beginning of the timestep. 
        # Specify subsidence rate
        sigma  = subsidence(sigma, t)
        # Compute flow parameters
        U_old = U.copy()
        H, Hn, U, tau_star = flow(t, Q_w, qw, B0, S, Cf, Cz, dx, H, Hc, eta, N,
                                  N_old, xi_d, D, R)
        # Compute load parameters
        qt_old = qt.copy()
        qt = load(t, dx, N, N_old, qt, Cf, tau_star, R, D)
        if i in i_out:
            if i == 0:
                os.mkdir(wd)
                os.chdir(wd)
            xi = eta + H
            x_out = np.append(x(dx), x_toe)
            eta_b_out = np.append(eta_b, eta_toe)
            eta_out = np.append(eta, eta_toe)
            xi_out = np.append(xi, xi[-1])
            S_out = np.append(S, S_d)
            qt_out = np.append(qt, 0)    # By definition, nothing leaves the toe
            output_data = np.transpose(np.vstack((x_out, eta_b_out, eta_out,
                                                  xi_out, S_out, qt_out )))
            print 'Output data stored at time {}.'.format(t)
            print output_data
            hdr = ('x-coord / m\t eta_b / m\t eta / m\t xi/ m\t S / m/m\t qt / m^3/s ')
            fmt = '%1.8f'
            f = str(int(t)) + '.txt'
            np.savetxt(f, output_data, fmt = fmt, delimiter='\t', header = hdr)
        # Make a copy of the domain, basement and bed profiles:
        N_old, dx_old, eta_b_old, eta_old, S_old = copy_old_values(N, dx,
                                                                   eta_b, eta,
                                                                   S)
        # We are currently at the end of the timestep. 
        # Move the bed; remesh if necessary. 
        dx, eta, x_toe, N = update_bed(dt, dx, x_shore, x_toe, eta, eta_old, S,
                                       S_d, S_b, qt, qt_f, lambda_p, N_old,
                                       I_f, sigma)
        # Update the domain. 
        eta_b, eta_toe, S, S_d, B0, x_shore, x_toe = update_domain(dt, dx,
                                                                   x_toe, N,
                                                                   N_old, eta,
                                                                   eta_b, S_b,
                                                                   B0, sigma)
    
    x_axis = x_out
    plt.plot(x_axis, eta_b_out, '-k')    # Plots the basement
    plt.plot(x_axis, eta_out, '-g')    # Plots the bed
    plt.plot(x_axis, xi_out, '-b')    # Plots water surface profile
    plt.show()
    plt.close()
    # Finalize the simulation
    pdb.set_trace()
    return

        

    

# Allow for this script to be imported as a module.
if __name__ == "__main__":
    main()

import probeinterface
from probeinterface import Probe

import numpy as np

"""
Poly2
"""

def create_poly2():

    n = 32
    positions = np.zeros((n, 2))

    height = 300
    for i in range(10):
        positions[i] = -21.65, height
        height += 50
    
    height = 250
    for i in range(10,16):
        positions[i] = -21.65, height
        height -= 50

    height = 25
    for i in range(16,22):
        positions[i] = 21.65, height
        height += 50
    
    height = 25 + 15*50
    for i in range(22,32):
        positions[i] = 21.65, height
        height -= 50

    probe = Probe(ndim=2, si_units='um')
    probe.set_contacts(positions=positions, shapes='circle', shape_params={'radius': 7.5})

    probeinterface.wiring.pathways['A32>RHD2132']=[30,26,21,17,27,22,20,25,28,23,19,24,29,18,31,16,0,15,2,13,8,9,7,1,6,14,10,11,5,12,4,3]
    probe.wiring_to_device('A32>RHD2132')

    return probe

def create_poly2_layout():
    """ 
    xy_layout is 2D numpy array where each row represents its
    corresonding channel number and each column gives the x, y coordinates
    of that channel in micrometers.
    """
    xy_layout = np.array([
    [21.65, 25], #0
    [21.65, 725], #1
    [21.65, 125], #2
    [21.65, 325], #3
    [21.65, 375], #4
    [21.65, 475], #5
    [21.65, 675], #6
    [21.65, 775], #7
    [21.65, 225], #8
    [21.65, 275], #9
    [21.65, 575], #10
    [21.65, 525], #11
    [21.65, 425], #12
    [21.65, 175], #13
    [21.65, 625], #14
    [21.65, 75], #15
    [-21.65, 0], #16
    [-21.65, 450], #17
    [-21.65, 100], #18
    [-21.65, 250], #19
    [-21.65, 600], #20
    [-21.65, 400], #21
    [-21.65, 550], #22
    [-21.65, 750], #23
    [-21.65, 200], #24
    [-21.65, 650], #25
    [-21.65, 350], #26
    [-21.65, 500], #27
    [-21.65, 700], #28
    [-21.65, 150], #29
    [-21.65, 300], #30
    [-21.65, 50] #31   
    ])
    return xy_layout

"""
Poly3
"""
def create_poly3():

    n = 32
    positions = np.zeros((n, 2))

    height = 12.5
    for i in range(10):
        positions[i] = -18, height
        height += 25

    height = 0
    for i in range(10,16):
        positions[i] = 0, height
        height += 50

    height -= 25
    for i in range(16,22):
        positions[i] = 0, height
        height -= 50

    height = 237.5
    for i in range(22,32):
        positions[i] = 18, height
        height -= 25

    probe = Probe(ndim=2, si_units='um')
    probe.set_contacts(positions=positions, shapes='circle', shape_params={'radius': 7.5})
    
    probeinterface.wiring.pathways['A32>RHD2132']=[30,26,21,17,27,22,20,25,28,23,19,24,29,18,31,16,0,15,2,13,8,9,7,1,6,14,10,11,5,12,4,3]
    probe.wiring_to_device('A32>RHD2132')

    return probe

def create_poly3_layout():
    """ 
    xy_layout is 2D numpy array where each row represents its
    corresonding channel number and each column gives the x, y coordinates
    of that channel in micrometers.
    """
    xy_layout = np.array([
    [0, 275], #0
    [18, 212.5], #1
    [0, 175], #2
    [18, 12.5], #3
    [18, 37.5], #4
    [18, 87.5], #5
    [18, 187.5], #6
    [18, 237.5], #7
    [0, 75], #8
    [0, 25], #9
    [18, 137.5], #10
    [18, 112.5], #11
    [18, 62.5], #12
    [0, 125], #13
    [18, 162.5], #14
    [0, 225], #15
    [0, 250], #16
    [-18, 87.5], #17
    [0, 150], #18
    [0, 0], #19
    [-18, 162.5], #20
    [-18, 62.5], #21
    [-18, 137.5], #22
    [-18, 237.5], #23
    [0, 50], #24
    [-18, 187.5], #25
    [-18, 37.5], #26
    [-18, 112.5], #27
    [-18, 212.5], #28
    [0, 100], #29
    [-18, 12.5], #30
    [0, 200] #31   
    ])
    return xy_layout



import numpy as np 

class converter:
    def __init__(self):
        pass

    def anglerange_deg(self,sumangle):
        if sumangle>180:
            sumangle-=360
        elif sumangle<=-180:
            sumangle+=360
        return sumangle

    def anglerange_rad(self,sumangle):
        if sumangle>np.pi:
            sumangle-=2*np.pi
        elif sumangle<=-np.pi:
            sumangle+=2*np.pi
        return sumangle

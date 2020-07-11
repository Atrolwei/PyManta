import numpy as np 
from math import sin,cos,tan


class waterdisturb:
    def __init__(self):
        pass

    def setdisturbonground(self,V_disturb,Psi,Theta):
        self.V_wg_x=V_disturb*cos(Psi)*cos(Theta)
        self.V_wg_y=V_disturb*sin(Theta)
        self.V_wg_z=-V_disturb*sin(Psi)*cos(Theta)


    def steadydisturb(self,psi,vartheta,gamma):
        v_wb_x=self.V_wg_x*cos(psi)*cos(vartheta) + self.V_wg_y*sin(vartheta) - self.V_wg_z*sin(psi)*cos(vartheta)
        v_wb_y=self.V_wg_x*(sin(gamma)*sin(psi) - sin(vartheta)*cos(gamma)*cos(psi)) + self.V_wg_y*cos(gamma)*cos(vartheta) + self.V_wg_z*(sin(gamma)*cos(psi) + sin(psi)*sin(vartheta)*cos(gamma))
        v_wb_z=self.V_wg_x*(sin(gamma)*sin(vartheta)*cos(psi) + sin(psi)*cos(gamma)) - self.V_wg_y*sin(gamma)*cos(vartheta) + self.V_wg_z*(-sin(gamma)*sin(psi)*sin(vartheta) + cos(gamma)*cos(psi))
        return v_wb_x,v_wb_y,v_wb_z

if __name__ == "__main__":
    steawaterdisturb=waterdisturb()
    steawaterdisturb.setdisturbonground(-0.25,-10/57.3,0)
    print(steawaterdisturb.steadydisturb(0,0,0))
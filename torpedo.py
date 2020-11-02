# -*- coding: utf-8 -*-
"""
Created on Wed May 20 09:24:09 2020

@author: weixufei
"""
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.interpolate import griddata
from math import pi


class torpedoforce:
    def __init__(self,S_ref,L_ref,rho):
        self.S_ref=S_ref
        self.L_ref=L_ref
        self.rho=rho
            
    def setderiv(self,cy_wz,cz_wy,mx_wx,my_wy,mz_wz,cx_wx=0):
        self.cx_wx=cx_wx
        self.cy_wz=cy_wz
        self.cz_wy=cz_wy
        self.mx_wx=mx_wx
        self.my_wy=my_wy
        self.mz_wz=mz_wz
    
    def waterforce(self,alpha,beta,wx,wy,wz,V):
        FM=np.array([0,0,0,0,0,0])
        FM=self.steadyforce(alpha,beta,V)+self.unsteadyforce(wx,wy,wz,V)
        return FM
    
    def steadyforce(self,alpha,beta,V):
        alphapoint,betapoint=self.alpha_beta_striction(alpha,beta)
        X_stea=1/2*self.rho*V**2*self.S_ref*self.Cx(alphapoint,betapoint)
        Y_stea=1/2*self.rho*V**2*self.S_ref*self.Cy(alphapoint,betapoint)
        Z_stea=1/2*self.rho*V**2*self.S_ref*self.Cz(alphapoint,betapoint)
        Mx_stea=1/2*self.rho*V**2*self.S_ref*self.L_ref*self.Cmx(alphapoint,betapoint)
        My_stea=1/2*self.rho*V**2*self.S_ref*self.L_ref*self.Cmy(alphapoint,betapoint)
        Mz_stea=1/2*self.rho*V**2*self.S_ref*self.L_ref*self.Cmz(alphapoint,betapoint)
        return np.array([X_stea,Y_stea,Z_stea,Mx_stea,My_stea,Mz_stea])
    
    def unsteadyforce(self,wx,wy,wz,V):
        X_unstea=1/2*self.rho*V*self.S_ref*self.L_ref*self.cx_wx*wx
        Y_unstea=1/2*self.rho*V*self.S_ref*self.L_ref*self.cy_wz*wz
        Z_unstea=1/2*self.rho*V*self.S_ref*self.L_ref*self.cz_wy*wy
        Mx_unstea=1/2*self.rho*V*self.S_ref*self.L_ref**2*self.mx_wx*wx
        My_unstea=1/2*self.rho*V*self.S_ref*self.L_ref**2*self.my_wy*wy
        Mz_unstea=1/2*self.rho*V*self.S_ref*self.L_ref**2*self.mz_wz*wz
        return np.array([X_unstea,Y_unstea,Z_unstea,Mx_unstea,My_unstea,Mz_unstea])
        
    
    def alpha_beta_striction(self,alpha,beta):
        # alpha(rad) beta(rad)
        if (alpha*57.3)>=18: alpha=18/57.3
        if (alpha*57.3)<=-18: alpha=-18/57.3
        if (beta*57.3)>=18: beta=18/57.3
        if (beta*57.3)<=-18: beta=-18/57.3
        return alpha,beta
    
    def Cx(self,alpha,beta):
        Cpoint=0.0052*2+0.0657*abs(alpha)
        return -Cpoint

    def Cy(self,alpha,beta):
        Cpoint=0.35*alpha
        return Cpoint

    def Cz(self,alpha,beta):
        Cpoint=-0.0134*beta
        return Cpoint
        
    def Cmx(self,alpha,beta):
        Cpoint=0
        return Cpoint

    def Cmy(self,alpha,beta):
        Cpoint=0.0124*beta
        return Cpoint

    def Cmz(self,alpha,beta):
        Cpoint=0.048*alpha
        return Cpoint
    
    def addtailrudder(self,dcx_dz,dcy_dz,dmz_dz):
        self.dcx_dz=dcx_dz
        self.dcy_dz=dcy_dz
        self.dmz_dz=dmz_dz
    
    def tailrudderforce(self,u,V):
        dzl,dzr=u
        assert abs(dzl)<pi and abs(dzr)<pi,"尾鳍/舵控制信号输入应为弧度制"
        dX_rudder=1/2*(1/2*self.rho*V**2*self.S_ref*self.dcx_dz*abs(dzl))+1/2*(1/2*self.rho*V**2*self.S_ref*self.dcx_dz*abs(dzr))
        dY_rudder=1/2*(1/2*self.rho*V**2*self.S_ref*self.dcy_dz*dzl)+1/2*(1/2*self.rho*V**2*self.S_ref*self.dcy_dz*dzr)
        dMz_rudder=1/2*(1/2*self.rho*V**2*self.S_ref*self.dmz_dz*dzl)+1/2*(1/2*self.rho*V**2*self.S_ref*self.dmz_dz*dzr)
        return dX_rudder,dY_rudder,dMz_rudder
    
    
if __name__=="__main__":
    ghostbody=torpedoforce(2,10.5,1024)
    ghostbody.setderiv(0.236,-0.236,0,-0.118,-0.118)
    print(ghostbody.waterforce(0/57.3,4/57.3,0,0,0,7))
        

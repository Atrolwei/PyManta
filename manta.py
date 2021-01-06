import numpy as np
from numpy import sin,cos,tan,sqrt,arcsin,arctan,pi,exp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from torpedo import torpedoforce
from thruster import Pectoralfin
from disturbance import waterdisturb
from utils import converter


class Manta:
    def __init__(self):
        # Induce the body
        self.mantabody=torpedoforce(0.6645,0.6,1024)
        self.mantabody.setderiv(cy_wz=-0.15,cz_wy=0,mx_wx=-0.115,my_wy=-0.003,mz_wz=-0.018)
        self.mantabody.addtailrudder(dcx_dz=-0.0174,dcy_dz=0.0431,dmz_dz=-0.0122)
        # Induce the 2 pectoral fins
        self.finchord_tip=finchord_tip=0.068
        self.finchord_root=finchord_root=0.402
        self.finspan_tip=finspan_tip=0.41
        self.finspan_root=finspan_root=0.07
        Lfront=276.5e-3
        Lmass=266.5e-3
        self.Pec_l=Pectoralfin([Lmass-Lfront-0.5*finspan_root,0,-90e-3],[1,1,-1],[finchord_root,finchord_root,0.1],[finspan_root,finspan_tip,0.44],120,(2.81,2.8,-0.019,0.0201),1)
        self.Pec_r=Pectoralfin([Lmass-Lfront-0.5*finspan_root,0,90e-3],[1,1,1],[finchord_root,finchord_root,0.1],[finspan_root,finspan_tip,0.44],120,(2.81,2.8,-0.019,0.0201),1)

        # Add Water Disturbance
        self.steawaterdisturb=waterdisturb()
        self.steawaterdisturb.setdisturbonground(0,-30/57.3,0)

        # Angle converter
        self.angleconverter=converter()

        # Define the dimension of obs space and action space
        self.dim_obs=18
        self.dim_action=12
    
    def reset(self,y0array,desire):
        # Setting initial states
        x0,y0,z0,vartheta0,psi0,gamma0,vx0,vy0,vz0,wx0,wy0,wz0=y0array
        self.vartheta_c,self.psi_c,self.gamma_c,self.vx_c,self.vy_c,self.vz_c=desire
        self.yn=np.array([x0,y0,z0,vartheta0,psi0,gamma0,vx0,vy0,vz0,wx0,wy0,wz0])
        self.state=np.append(y0array,np.array([
            self.vartheta_c-vartheta0,
            self.psi_c-psi0,
            self.gamma_c-gamma0,
            self.vx_c-vx0,
            self.vy_c-vy0,
            self.vz_c-vz0
        ]))

        # 重置失稳和出界的标记
        self.unstable=False
        self.outscale=False

        # 设置成功完成任务的标记
        self.success=False
        self.doneflag=False

        self.ynlist=[]
        self.alphalist=[]
        self.betalist=[]
        self.FMbodylist=[]
        self.TFMlist=[]
        self.tn=0
        return self.state
    
    def __Manta6dof(self,t,y,controlU,steptime):
        x, y, z, vartheta, psi, gamma, v_x, v_y, v_z, omega_x, omega_y, omega_z = y
        #重浮力抵消
        mass=14.4
        Buyoncy=14.4*9.81
        #附加质量
        lambda_11=2.9
        lambda_22=61
        lambda_33=5.3
        lambda_44=1.56
        lambda_55=0.071
        lambda_66=0.4
        lambda_26=lambda_62=-0.84
        lambda_35=lambda_53=0.175

        #set addmass to 0
        # lambda_11=lambda_22=lambda_33=lambda_44=lambda_55=lambda_66=lambda_26=lambda_62=lambda_35=lambda_53=0

        vwb_x,vwb_y,vwb_z= self.steawaterdisturb.steadydisturb(psi,vartheta,gamma)
        v_x_r=v_x-vwb_x
        v_y_r=v_y-vwb_y
        v_z_r=v_z-vwb_z

        y_G=-0.003*1
            
        alpha=-arctan(v_y/(v_x))
        beta=arctan(v_z/sqrt(v_x**2+v_y**2))
        alpha_r=-arctan(v_y_r/(v_x_r))
        beta_r=arctan(v_z_r/sqrt(v_x_r**2+v_y_r**2))
        self.alphalist.append(alpha_r)
        self.betalist.append(beta_r)

        V=sqrt(v_x**2+v_y**2+v_z**2)
        V_r=sqrt(v_x_r**2+v_y_r**2+v_z_r**2)

        theta=arcsin(cos(alpha)*cos(beta)*sin(vartheta)-sin(alpha)*cos(beta)*cos(vartheta)*cos(gamma)-sin(beta)*cos(vartheta)*sin(gamma))
        Psi=arcsin((cos(alpha)*cos(beta)*sin(psi)*cos(vartheta)+sin(alpha)*cos(beta)*sin(vartheta)*sin(psi)*cos(gamma)+sin(alpha)*cos(beta)*cos(psi)*sin(gamma)-sin(beta)*cos(psi)*cos(gamma)+sin(beta)*sin(vartheta)*sin(psi)*sin(gamma))/cos(theta))

        F_x,F_y,F_z,M_x,M_y,M_z=self.mantabody.waterforce(alpha_r,beta_r,omega_x,omega_y,omega_z,V_r)
        self.FMbodylist.append([F_x,F_y,F_z,M_x,M_y,M_z])

        # calculate the movement 
        frez=0.6
        dphi=pi/2

        Aflapl,Aflapr,Atwistl,Atwistr,Aflbiasl,Aflbiasr,Atwbiasl,Atwbiasr,dzl,dzr,dphil,dphir=controlU

        # generate the sine signal and calculate force
        Fxl,Fyl,Fzl,Mxl,Myl,Mzl=self.Pec_l.calcforce([v_x_r,v_y_r],self.Pec_l.sinemovegene2(t,frez,Aflapl,Atwistl,dphil,Aflbiasl,Atwbiasl),[omega_x,omega_y,omega_z],steptime)
        Fxr,Fyr,Fzr,Mxr,Myr,Mzr=self.Pec_r.calcforce([v_x_r,v_y_r],self.Pec_r.sinemovegene2(t,frez,Aflapr,Atwistr,dphir,Aflbiasr,Atwbiasr),[omega_x,omega_y,omega_z],steptime)
        Fxrudder,Fyrudder,Mzrudder=self.mantabody.tailrudderforce((dzl,dzr),V_r)
        T_x=Fxl+Fxr+Fxrudder
        T_y=Fyl+Fyr+Fyrudder
        T_z=Fzl+Fzr
        T_mx=Mxl+Mxr
        T_my=Myl+Myr
        T_mz=Mzl+Mzr+Mzrudder

        self.TFMlist.append([T_x,T_y,T_z,T_mx,T_my,T_mz])

        res=[V*cos(theta)*cos(Psi),
            V*sin(theta),
            -V*cos(theta)*sin(Psi),
            omega_z*cos(gamma)+omega_y*sin(gamma),
            1/cos(vartheta)*(omega_y*cos(gamma)-omega_z*sin(gamma)),
            omega_x-tan(vartheta)*(omega_y*cos(gamma)-omega_z*sin(gamma)),
            -lambda_26*mass*y_G*(Buyoncy*cos(gamma)*cos(vartheta) + F_y + T_y + lambda_35*omega_x*omega_y + mass*omega_x**2*y_G + mass*omega_z**2*y_G - 9.81*mass*cos(gamma)*cos(vartheta) + omega_x*v_z*(lambda_33 + mass) - omega_z*v_x*(lambda_11 + mass))/(-lambda_26**2*(lambda_11 + mass) + (lambda_22 + mass)*(-mass**2*y_G**2 + (lambda_11 + mass)*(lambda_66 + 0.1995))) + mass*y_G*(lambda_22 + mass)*(M_z + T_mz + 9.81*mass*y_G*sin(vartheta) + omega_x*omega_y*(lambda_44 + 0.0574) - omega_x*omega_y*(lambda_55 + 0.2177) - omega_z*(lambda_26*v_x + mass*v_y*y_G) + v_x*v_y*(lambda_11 + mass) - v_x*v_y*(lambda_22 + mass) - v_z*(lambda_35*omega_x - mass*omega_y*y_G))/(-lambda_26**2*(lambda_11 + mass) + (lambda_22 + mass)*(-mass**2*y_G**2 + (lambda_11 + mass)*(lambda_66 + 0.1995))) + (-lambda_26**2*(lambda_11 + mass) + mass**2*y_G**2*(lambda_22 + mass) + (lambda_22 + mass)*(-mass**2*y_G**2 + (lambda_11 + mass)*(lambda_66 + 0.1995)))*(Buyoncy*sin(vartheta) + F_x + T_x + lambda_26*omega_z**2 - lambda_35*omega_y**2 - mass*omega_x*omega_y*y_G - 9.81*mass*sin(vartheta) - omega_y*v_z*(lambda_33 + mass) + omega_z*v_y*(lambda_22 + mass))/((lambda_11 + mass)*(-lambda_26**2*(lambda_11 + mass) + (lambda_22 + mass)*(-mass**2*y_G**2 + (lambda_11 + mass)*(lambda_66 + 0.1995)))),
            -lambda_26*mass*y_G*(Buyoncy*sin(vartheta) + F_x + T_x + lambda_26*omega_z**2 - lambda_35*omega_y**2 - mass*omega_x*omega_y*y_G - 9.81*mass*sin(vartheta) - omega_y*v_z*(lambda_33 + mass) + omega_z*v_y*(lambda_22 + mass))/(-lambda_26**2*(lambda_11 + mass) + (lambda_22 + mass)*(-mass**2*y_G**2 + (lambda_11 + mass)*(lambda_66 + 0.1995))) - lambda_26*(lambda_11 + mass)*(M_z + T_mz + 9.81*mass*y_G*sin(vartheta) + omega_x*omega_y*(lambda_44 + 0.0574) - omega_x*omega_y*(lambda_55 + 0.2177) - omega_z*(lambda_26*v_x + mass*v_y*y_G) + v_x*v_y*(lambda_11 + mass) - v_x*v_y*(lambda_22 + mass) - v_z*(lambda_35*omega_x - mass*omega_y*y_G))/(-lambda_26**2*(lambda_11 + mass) + (lambda_22 + mass)*(-mass**2*y_G**2 + (lambda_11 + mass)*(lambda_66 + 0.1995))) + (-mass**2*y_G**2 + (lambda_11 + mass)*(lambda_66 + 0.1995))*(Buyoncy*cos(gamma)*cos(vartheta) + F_y + T_y + lambda_35*omega_x*omega_y + mass*omega_x**2*y_G + mass*omega_z**2*y_G - 9.81*mass*cos(gamma)*cos(vartheta) + omega_x*v_z*(lambda_33 + mass) - omega_z*v_x*(lambda_11 + mass))/(-lambda_26**2*(lambda_11 + mass) + (lambda_22 + mass)*(-mass**2*y_G**2 + (lambda_11 + mass)*(lambda_66 + 0.1995))),
            -(lambda_35*mass**2*y_G**2 + lambda_35*(-mass**2*y_G**2 + (lambda_33 + mass)*(lambda_44 + 0.0574)))*(M_y + T_my + lambda_26*omega_x*v_y + lambda_35*omega_y*v_x - omega_x*(-mass*v_x*y_G + omega_z*(lambda_44 + 0.0574)) - omega_z*(-mass*v_z*y_G - omega_x*(lambda_66 + 0.1995)) - v_x*(mass*omega_x*y_G + v_z*(lambda_11 + mass)) - v_z*(mass*omega_z*y_G - v_x*(lambda_33 + mass)))/(-lambda_35**2*mass**2*y_G**2 + (-lambda_35**2 + (lambda_33 + mass)*(lambda_55 + 0.2177))*(-mass**2*y_G**2 + (lambda_33 + mass)*(lambda_44 + 0.0574))) + ((lambda_33 + mass)*(lambda_44 + 0.0574)*(-lambda_35**2*mass**2*y_G**2 + (-lambda_35**2 + (lambda_33 + mass)*(lambda_55 + 0.2177))*(-mass**2*y_G**2 + (lambda_33 + mass)*(lambda_44 + 0.0574))) - (-lambda_35*mass**2*y_G**2 - lambda_35*(-mass**2*y_G**2 + (lambda_33 + mass)*(lambda_44 + 0.0574)))*(lambda_35*mass**2*y_G**2 + lambda_35*(-mass**2*y_G**2 + (lambda_33 + mass)*(lambda_44 + 0.0574))))*(-Buyoncy*sin(gamma)*cos(vartheta) + F_z + T_z + 9.81*mass*sin(gamma)*cos(vartheta) - omega_x*v_y*(lambda_22 + mass) + omega_y*v_x*(lambda_11 + mass) - omega_z*(lambda_26*omega_x + mass*omega_y*y_G))/((lambda_33 + mass)*(-mass**2*y_G**2 + (lambda_33 + mass)*(lambda_44 + 0.0574))*(-lambda_35**2*mass**2*y_G**2 + (-lambda_35**2 + (lambda_33 + mass)*(lambda_55 + 0.2177))*(-mass**2*y_G**2 + (lambda_33 + mass)*(lambda_44 + 0.0574)))) + (-lambda_35*mass*y_G*(lambda_33 + mass)*(lambda_35*mass**2*y_G**2 + lambda_35*(-mass**2*y_G**2 + (lambda_33 + mass)*(lambda_44 + 0.0574))) - mass*y_G*(lambda_33 + mass)*(-lambda_35**2*mass**2*y_G**2 + (-lambda_35**2 + (lambda_33 + mass)*(lambda_55 + 0.2177))*(-mass**2*y_G**2 + (lambda_33 + mass)*(lambda_44 + 0.0574))))*(M_x + T_mx - mass*omega_x*v_y*y_G + mass*omega_y*v_x*y_G + 9.81*mass*y_G*sin(gamma)*cos(vartheta) - omega_y*(lambda_35*v_y - omega_z*(lambda_55 + 0.2177)) - omega_z*(-lambda_26*v_z + omega_y*(lambda_66 + 0.1995)) - v_y*(lambda_26*omega_y - v_z*(lambda_22 + mass)) - v_z*(-lambda_35*omega_z + v_y*(lambda_33 + mass)))/((lambda_33 + mass)*(-mass**2*y_G**2 + (lambda_33 + mass)*(lambda_44 + 0.0574))*(-lambda_35**2*mass**2*y_G**2 + (-lambda_35**2 + (lambda_33 + mass)*(lambda_55 + 0.2177))*(-mass**2*y_G**2 + (lambda_33 + mass)*(lambda_44 + 0.0574)))),
            lambda_35*mass*y_G*(lambda_33 + mass)*(M_y + T_my + lambda_26*omega_x*v_y + lambda_35*omega_y*v_x - omega_x*(-mass*v_x*y_G + omega_z*(lambda_44 + 0.0574)) - omega_z*(-mass*v_z*y_G - omega_x*(lambda_66 + 0.1995)) - v_x*(mass*omega_x*y_G + v_z*(lambda_11 + mass)) - v_z*(mass*omega_z*y_G - v_x*(lambda_33 + mass)))/(-lambda_35**2*mass**2*y_G**2 + (-lambda_35**2 + (lambda_33 + mass)*(lambda_55 + 0.2177))*(-mass**2*y_G**2 + (lambda_33 + mass)*(lambda_44 + 0.0574))) + (lambda_35*mass*y_G*(-lambda_35*mass**2*y_G**2 - lambda_35*(-mass**2*y_G**2 + (lambda_33 + mass)*(lambda_44 + 0.0574))) - mass*y_G*(-lambda_35**2*mass**2*y_G**2 + (-lambda_35**2 + (lambda_33 + mass)*(lambda_55 + 0.2177))*(-mass**2*y_G**2 + (lambda_33 + mass)*(lambda_44 + 0.0574))))*(-Buyoncy*sin(gamma)*cos(vartheta) + F_z + T_z + 9.81*mass*sin(gamma)*cos(vartheta) - omega_x*v_y*(lambda_22 + mass) + omega_y*v_x*(lambda_11 + mass) - omega_z*(lambda_26*omega_x + mass*omega_y*y_G))/((-mass**2*y_G**2 + (lambda_33 + mass)*(lambda_44 + 0.0574))*(-lambda_35**2*mass**2*y_G**2 + (-lambda_35**2 + (lambda_33 + mass)*(lambda_55 + 0.2177))*(-mass**2*y_G**2 + (lambda_33 + mass)*(lambda_44 + 0.0574)))) + (lambda_35**2*mass**2*y_G**2*(lambda_33 + mass) + (lambda_33 + mass)*(-lambda_35**2*mass**2*y_G**2 + (-lambda_35**2 + (lambda_33 + mass)*(lambda_55 + 0.2177))*(-mass**2*y_G**2 + (lambda_33 + mass)*(lambda_44 + 0.0574))))*(M_x + T_mx - mass*omega_x*v_y*y_G + mass*omega_y*v_x*y_G + 9.81*mass*y_G*sin(gamma)*cos(vartheta) - omega_y*(lambda_35*v_y - omega_z*(lambda_55 + 0.2177)) - omega_z*(-lambda_26*v_z + omega_y*(lambda_66 + 0.1995)) - v_y*(lambda_26*omega_y - v_z*(lambda_22 + mass)) - v_z*(-lambda_35*omega_z + v_y*(lambda_33 + mass)))/((-mass**2*y_G**2 + (lambda_33 + mass)*(lambda_44 + 0.0574))*(-lambda_35**2*mass**2*y_G**2 + (-lambda_35**2 + (lambda_33 + mass)*(lambda_55 + 0.2177))*(-mass**2*y_G**2 + (lambda_33 + mass)*(lambda_44 + 0.0574)))),
            lambda_35*mass*y_G*(lambda_33 + mass)*(M_x + T_mx - mass*omega_x*v_y*y_G + mass*omega_y*v_x*y_G + 9.81*mass*y_G*sin(gamma)*cos(vartheta) - omega_y*(lambda_35*v_y - omega_z*(lambda_55 + 0.2177)) - omega_z*(-lambda_26*v_z + omega_y*(lambda_66 + 0.1995)) - v_y*(lambda_26*omega_y - v_z*(lambda_22 + mass)) - v_z*(-lambda_35*omega_z + v_y*(lambda_33 + mass)))/(-lambda_35**2*mass**2*y_G**2 + (-lambda_35**2 + (lambda_33 + mass)*(lambda_55 + 0.2177))*(-mass**2*y_G**2 + (lambda_33 + mass)*(lambda_44 + 0.0574))) + (lambda_33 + mass)*(-mass**2*y_G**2 + (lambda_33 + mass)*(lambda_44 + 0.0574))*(M_y + T_my + lambda_26*omega_x*v_y + lambda_35*omega_y*v_x - omega_x*(-mass*v_x*y_G + omega_z*(lambda_44 + 0.0574)) - omega_z*(-mass*v_z*y_G - omega_x*(lambda_66 + 0.1995)) - v_x*(mass*omega_x*y_G + v_z*(lambda_11 + mass)) - v_z*(mass*omega_z*y_G - v_x*(lambda_33 + mass)))/(-lambda_35**2*mass**2*y_G**2 + (-lambda_35**2 + (lambda_33 + mass)*(lambda_55 + 0.2177))*(-mass**2*y_G**2 + (lambda_33 + mass)*(lambda_44 + 0.0574))) + (-lambda_35*mass**2*y_G**2 - lambda_35*(-mass**2*y_G**2 + (lambda_33 + mass)*(lambda_44 + 0.0574)))*(-Buyoncy*sin(gamma)*cos(vartheta) + F_z + T_z + 9.81*mass*sin(gamma)*cos(vartheta) - omega_x*v_y*(lambda_22 + mass) + omega_y*v_x*(lambda_11 + mass) - omega_z*(lambda_26*omega_x + mass*omega_y*y_G))/(-lambda_35**2*mass**2*y_G**2 + (-lambda_35**2 + (lambda_33 + mass)*(lambda_55 + 0.2177))*(-mass**2*y_G**2 + (lambda_33 + mass)*(lambda_44 + 0.0574))),
            -lambda_26*(lambda_11 + mass)*(Buyoncy*cos(gamma)*cos(vartheta) + F_y + T_y + lambda_35*omega_x*omega_y + mass*omega_x**2*y_G + mass*omega_z**2*y_G - 9.81*mass*cos(gamma)*cos(vartheta) + omega_x*v_z*(lambda_33 + mass) - omega_z*v_x*(lambda_11 + mass))/(-lambda_26**2*(lambda_11 + mass) + (lambda_22 + mass)*(-mass**2*y_G**2 + (lambda_11 + mass)*(lambda_66 + 0.1995))) + mass*y_G*(lambda_22 + mass)*(Buyoncy*sin(vartheta) + F_x + T_x + lambda_26*omega_z**2 - lambda_35*omega_y**2 - mass*omega_x*omega_y*y_G - 9.81*mass*sin(vartheta) - omega_y*v_z*(lambda_33 + mass) + omega_z*v_y*(lambda_22 + mass))/(-lambda_26**2*(lambda_11 + mass) + (lambda_22 + mass)*(-mass**2*y_G**2 + (lambda_11 + mass)*(lambda_66 + 0.1995))) + (lambda_11 + mass)*(lambda_22 + mass)*(M_z + T_mz + 9.81*mass*y_G*sin(vartheta) + omega_x*omega_y*(lambda_44 + 0.0574) - omega_x*omega_y*(lambda_55 + 0.2177) - omega_z*(lambda_26*v_x + mass*v_y*y_G) + v_x*v_y*(lambda_11 + mass) - v_x*v_y*(lambda_22 + mass) - v_z*(lambda_35*omega_x - mass*omega_y*y_G))/(-lambda_26**2*(lambda_11 + mass) + (lambda_22 + mass)*(-mass**2*y_G**2 + (lambda_11 + mass)*(lambda_66 + 0.1995)))]
        return np.array(res)

    def calreward(self,state):
        # Reward计算时可以调用self.期望值以及当前航行状态yn计算
        # 此处针对跟踪期望速度任务给出了一个临时的简单线性Step Reward函数，未经验证是否可用
        a_vx,a_vy,a_vz=(5,0.1,0.5)
        stepreward=a_vx/(abs(state[-3]/(self.vx_c+0.01))+0.01)+a_vy/(abs(state[-2]/(self.vy_c+0.01))+0.01)+a_vz/(abs(state[-1]/(self.vz_c+0.01))+0.01)
        if stepreward>20000:
            stepreward=20000
        return stepreward

    def ifdone(self,yn):
        x, y, z, vartheta, psi, gamma, vx, vy, vz, wx, wy, wz= yn
        if np.abs(psi) > pi or np.abs(vartheta) > 100/180*pi or np.abs(gamma) > 0.4 or vx > 2 or vy>1:
            # Unstable done
            self.unstable=True
            self.doneflag=True
        if x < -10 or y > -0.01 or y < -20 or z < -5 or z > 5:
            # Outscale done
            self.outscale=True
            self.doneglag=True
        # 此处设计了按照实现期望三轴航行速度的结束条件
        if (vx-self.vx_c)**2+(vy-self.vy_c)**2+(vz-self.vz_c)**2<0.1:
            self.success=True
            self.doneflag=True
        return self.doneflag


    def step(self,controlU,steptime):
        #欧拉法解微分方程
        self.h=steptime
        K1=self.__Manta6dof(self.tn,self.yn,controlU,steptime)
        self.yn=self.yn+self.h*K1
        x, y, z, vartheta, psi, gamma, vx, vy, vz, wx, wy, wz= self.yn
        self.ynlist.append([x,y,z,self.angleconverter.anglerange_rad(vartheta),self.angleconverter.anglerange_rad(psi),self.angleconverter.anglerange_rad(gamma),vx,vy,vz,wx,wy,wz])
        self.tn+=self.h
        self.state=np.append(self.yn,np.array(
               [self.vartheta_c-vartheta,
                self.psi_c-psi,
                self.gamma_c-gamma,
                self.vx_c-vx,
                self.vy_c-vy,
                self.vz_c-vz
                ]))
        
        done=self.ifdone(self.yn)

        reward=self.calreward(self.state)
        return self.state,reward,done
    
    def render(self):
        #Plotting figures
        fig,ax=plt.subplots(3,3,figsize=(15,15))
        # fig.subplots_adjust(left=0,bottom=0,top=1,right=1)
        axes=ax.flatten()
        t_eval=np.linspace(0,self.tn,1/self.h*self.tn+1)
        [xlist,ylist,zlist,varthetalist,psilist,gammalist,vxlist,vylist,vzlist,wxlist,wylist,wzlist]=\
        np.array(self.ynlist)[:,0],np.array(self.ynlist)[:,1],np.array(self.ynlist)[:,2],np.array(self.ynlist)[:,3],np.array(self.ynlist)[:,4],np.array(self.ynlist)[:,5],np.array(self.ynlist)[:,6],np.array(self.ynlist)[:,7],np.array(self.ynlist)[:,8],np.array(self.ynlist)[:,9],np.array(self.ynlist)[:,10],np.array(self.ynlist)[:,11]
        Mxbody,Mybody,Mzbody=np.array(self.FMbodylist)[:,3],np.array(self.FMbodylist)[:,4],np.array(self.FMbodylist)[:,5]

        Tmxfin,Tmyfin,Tmzfin=np.array(self.TFMlist)[:,3],np.array(self.TFMlist)[:,4],np.array(self.TFMlist)[:,5]
        
        axes[0].plot(xlist,ylist)
        axes[0].set_title('X-Y Trace')
        axes[0].set_xlabel('X(m)')
        axes[0].set_ylabel('Y(m)')
        axes[1].plot(xlist,zlist)
        axes[1].set_title('X-Z Trace')
        axes[1].set_xlabel('X(m)')
        axes[1].set_ylabel('Z(m)')
        axes[2].plot(t_eval,np.sqrt(np.array(vxlist)**2+np.array(vylist)**2+np.array(vzlist)**2))
        axes[2].set_title("total velocity")
        axes[2].set_xlabel('t(s)')
        axes[2].set_ylabel('totalV(m/s)')
        axes[3].plot(t_eval,np.array(self.alphalist)*57.3,'g--',t_eval,np.array(varthetalist)*57.3,'r-')
        axes[3].set_title("alpha(deg)/pitch(deg)")
        axes[3].legend(['alpha','pitch'])
        axes[3].set_xlabel('t(s)')
        axes[3].set_ylabel('angle(deg)')
        axes[4].plot(t_eval,np.array(self.betalist)*57.3,'g--',t_eval,np.array(psilist)*57.3,'r-')
        axes[4].set_title("beta(deg)/psi(deg)")
        axes[4].legend(['beta','psi'])
        axes[4].set_xlabel('t(s)')
        axes[4].set_ylabel('angle(deg)')
        axes[5].plot(t_eval,np.array(gammalist)*57.3,'r-')
        axes[5].set_title("roll(deg)")
        axes[5].legend(['roll'])
        axes[5].set_xlabel('t(s)')
        axes[5].set_ylabel('angle(deg)')
        axes[6].plot(t_eval,Tmzfin,'g-',t_eval,Mzbody,'r-')
        axes[6].set_title("Mz/Tmz")
        axes[6].legend(['Tmz','Mz'])
        axes[6].set_xlabel('t(s)')
        axes[6].set_ylabel('M(N·m)')
        axes[7].plot(t_eval,Tmyfin,'g-',linewidth=0.2)
        axes[7].plot(t_eval,Mybody,'r-')
        axes[7].set_title("My/Tmy")
        axes[7].legend(['Tmy','My'])
        axes[7].set_xlabel('t(s)')
        axes[7].set_ylabel('M(N·m)')
        axes[8].plot(t_eval,Tmxfin,'g-',t_eval,Mxbody,'r-')
        axes[8].set_title("Mx/Tmx")
        axes[8].legend(['Tmx','Mx'])
        axes[8].set_xlabel('t(s)')
        axes[8].set_ylabel('M(N·m)')
        plt.show()

if __name__ == "__main__":
    env=Manta()
    ''' 
    状态量state设为 x,y,z,vartheta,psi,gamma,vx,vy,vz,wx,wy,wz,dvx,dvy,dvz,dvartheta,dpsi,dgamma
    其中，x,y,z为航行器在惯性系下的三轴位置，x沿航行器纵轴指向头部，y沿航行器中纵剖面指向上，z轴按右手定则指向右
    vartheta,psi,gamma分别为航行器欧拉角形式的姿态角，俯仰角、偏航角和滚动角
    vx,vy,vz分别为航行器惯性系下三轴速度
    wx,wy,wz分别为航行器体轴系下三轴角速度
    dvx,dvy,dvz,dvartheta,dpsi,dgamma分别为航行器期望速度、期望姿态角与当前姿态角的差，state_c-state_now
    '''
    # 初始化航行器状态x,y,z,vartheta,psi,gamma,vx,vy,vz,wx,wy,wz,当从外部调用时，初始状态向量可随机生成
    y0=[0,-5,0,0,0,0,0.1,0,0,0,0,0]
    # 初始化期望vartheta_c,psi_c,gamma_c,vx_c,vy_c,vz_c,当从外部调用时，初始状态向量可随机生成或者根据航路解算得到
    desire=[10/57.3,10/57.3,0,1,0,0]
    done=False
    env.reset(y0,desire)
    steptime=0.001
    tend=2
    if not done:
        for i in range(int(tend/steptime)):
            '''
            Action:
            Aflapl(左胸鳍摆幅),Aflapr(右胸鳍摆幅),[0/57.3，30/57.3]
            Atwistl(左胸鳍扭幅),Atwistr(右胸鳍扭幅),[0/57.3，30/57.3]
            Aflbiasl(左胸鳍摆动偏置，向上为正),Aflbiasr(右胸鳍摆动偏置，向上为正),[-30/57.3，30/57.3]
            Atwbiasl(左胸鳍扭转偏置，向上为正),Atwbiasr(右胸鳍扭转偏置，向上为正),[-30/57.3，30/57.3]
            dzl(左尾鳍摆幅),dzr(右尾鳍摆幅),[-30/57.3，30/57.3]
            dphil(左胸鳍扭摆相位差),dphir(右胸鳍扭摆相位差)，dphi=扭转初始相位-摆动初始相位[-180/57.3，180/57.3]
            '''
            # 下方action直接给出了，实际应由智能体计算得到
            action=[30/57.3,30/57.3,30/57.3,30/57.3,0,0,0,0,0,0,pi/2,pi/2]
            state,reward,done=env.step(action,steptime=steptime)
            print(reward)
    # env.render()
    

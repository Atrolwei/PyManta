# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:08:10 2020

@author: weixufei
"""
import numpy as np
from math import pi,sin,cos,tan,atan,sqrt
from matplotlib import pyplot as plt
from patterngenerator import sinegenerator

class Propeller:
    def __init__(self):
        self.thrusternum=0
        self.thrustermaxforce=[]
        self.thrusterposi=[]
        self.thrustertowards=[]
        
    def addthruster(self,maxforce,posi,towards):
        self.thrusterID=self.thrusternum
        self.thrusternum+=1
        self.thrustermaxforce.append([])
        self.thrustermaxforce[self.thrusterID]=maxforce*np.array(towards)
        self.thrusterposi.append([])
        self.thrusterposi[self.thrusterID]=posi
        self.thrustertowards.append([])
        self.thrustertowards[self.thrusterID]=towards
    
    def calcforce(self,u):
        assert len(u)==self.thrusternum,"请输入匹配推进器数量的控制参数"
        thrustforcecontrol=np.zeros([self.thrusternum,3])
        torqueforcecontrol=np.zeros([self.thrusternum,3])
        for i in range(self.thrusternum):
            thrustforcecontrol[i,:]=u[i]*self.thrustermaxforce[i]
            torqueforcecontrol[i,:]=np.cross(np.array(self.thrusterposi[i]),thrustforcecontrol[i,:])
        forceaddmatrix=np.ones([1,self.thrusternum])
        linearforce=forceaddmatrix@thrustforcecontrol
        torqueforce=forceaddmatrix@torqueforcecontrol
        return np.concatenate((np.squeeze(linearforce),np.squeeze(torqueforce)))
    
class Pectoralfin:
    '''
    Define:
    flap up and twist up are positive.
    '''

    def __init__(self,posi,towards,chordcharas,spancharas,Iter,Cs,d):
        self.posi=posi
        self.towards=towards
        # chord_fppercent means the force point position before the twist axis
        self.fin_chord_root,self.fin_chord_tip,self.chord_fppercent=chordcharas
        self.fin_span_root,self.fin_span_tip,self.span_fppercent=spancharas
        self.forceposi=self.getforceposi(posi,towards)
        self.Iter=Iter
        self.C1,self.C2,self.C3,self.C4=Cs
        self.d=d
        self.vy_piece_old=np.zeros(self.Iter)
        self.Aflrtlist=[]
        self.Atwrtlist=[]

    def getforceposi(self,posi,towards):
        # Note: the following calculation can only satisfies the horizontal instances now!
        # the fix position of fin
        posiX,posiY,posiZ=posi
        toX,toY,toZ=towards
        # get the mass point position
        relativeX=self.chord_fppercent*(self.span_fppercent*(self.fin_chord_tip-self.fin_chord_root)+self.fin_chord_root)
        relativeY=0
        relativeZ=self.span_fppercent*(self.fin_span_tip-self.fin_span_root)+self.fin_span_root

        finforceX=posiX+relativeX*toX
        finforceY=posiY+relativeY*toY
        finforceZ=posiZ+relativeZ*toZ
        return [finforceX,finforceY,finforceZ]

    
    def calcforce(self,Ulist,finmove,w1,T):
        '''
        d: excahge character
        '''
        Ux,Uy=Ulist
        finfrez,Aflap,dAflap,Atwist,dAtwist=finmove
        w1x,w1y,w1z=w1
        Vx=Ux
        Uxy=sqrt(Ux**2+Uy**2)
        Iter=self.Iter
        Fx=Fy=Fz=0
        for i in range(Iter):
            span_piece=(self.fin_span_tip-self.fin_span_root)/Iter*i+self.fin_span_root#切片展长
            chord_piece=(self.fin_chord_tip-self.fin_chord_root)/Iter*i+self.fin_chord_root#切片弦长
            Atwist_piece=Atwist*i/Iter  #胸鳍切片每片的扭转角
            dAtwist_piece=dAtwist*i/Iter  #胸鳍切片每片的扭转角速度
            Vy=Uy+self.posi[0]*w1z-(span_piece*self.towards[2]+self.posi[2])*w1x+dAflap*span_piece
            Vxy=sqrt(Vx**2+Vy**2)   #胸鳍剖面合速度大小
            Theta=atan(Vy/(Vx+0.00001))#弹道倾角
            alpha_local=Atwist_piece-Theta#胸鳍切片当地攻角
            fxPos,fyPos,fzPos=self.finposiforce(alpha_local,chord_piece,Vxy,Theta,Aflap)
            k_finfrez=finfrez*(pi*chord_piece)/(Uxy+0.00001)#折合频率f
            if k_finfrez>0.415:
                fxI,fyI,fzI=self.finIforce(Vx,Vy,Atwist_piece,dAtwist_piece,Aflap,chord_piece,T,i)
            else:
                fxI=fyI=fzI=0
            Fx+=fxPos+fxI
            Fy+=fyPos+fyI
            Fz+=fzPos+fzI
        Fx*=(self.fin_span_tip-self.fin_span_root)/Iter*self.d
        Fy*=(self.fin_span_tip-self.fin_span_root)/Iter*self.d
        Fz*=(self.fin_span_tip-self.fin_span_root)/Iter*self.d

        forceposiXrt=self.forceposi[0]
        forceposiYrt=self.forceposi[1]+sin(Aflap)*(self.fin_span_root+0.44*(self.fin_span_tip-self.fin_span_root))
        
        forceposiZrt=self.forceposi[2]
        Mx,My,Mz=np.cross(np.array([forceposiXrt,forceposiYrt,forceposiZrt]),np.array([Fx,Fy,Fz]))
        return Fx,Fy,Fz,Mx,My,Mz

    def finposiforce(self,alpha_local,chord_piece,Vxy,Theta,Aflap):
        CL=self.C1*alpha_local
        CD=self.C2*alpha_local**2+self.C3*abs(alpha_local)+self.C4
        water_density=1024
        fxPos=-water_density*chord_piece*Vxy**2/2*(CL*sin(Theta)+CD*cos(Theta))*self.towards[0]
        fyPos=water_density*chord_piece*Vxy**2/2*(CL*cos(Theta)-CD*sin(Theta))*cos(Aflap)*self.towards[1]
        fzPos=water_density*chord_piece*Vxy**2/2*(CL*cos(Theta)-CD*sin(Theta))*sin(Aflap)*self.towards[2]
        return fxPos,fyPos,fzPos

    def finIforce(self,Vx,Vy,Atwist_piece,dAtwist_piece,Aflap,chord_piece,T,pieceID):
        vx_piece=Vx*cos(Atwist_piece)+Vy*sin(Atwist_piece)
        vy_piece=-Vx*sin(Atwist_piece)+Vy*cos(Atwist_piece)
        dvy_piece=(vy_piece-self.vy_piece_old[pieceID])/T
        self.vy_piece_old[pieceID]=vy_piece
        if dvy_piece>=10:
            dvy_piece=0
        water_density=1024
        lamda22=3*pi/16*water_density*chord_piece**2
        fxIi=-lamda22*vy_piece*dAtwist_piece
        fyIi=-lamda22*dvy_piece
        # MzIi=-lamda22*vx_piece*vy_piece
        fxI=fxIi*cos(Atwist_piece)-fyIi*sin(Atwist_piece)*self.towards[0]
        fyI=(fyIi*cos(Atwist_piece)+fxIi*sin(Atwist_piece))*cos(Aflap)*self.towards[1]
        fzI=(fyIi*cos(Atwist_piece)+fxIi*sin(Atwist_piece))*sin(Aflap)*self.towards[2]
        return fxI,fyI,fzI
    
    def sinemovegene2(self,t,frez,Aflap,Atwist,dphi,Aflbias,Atwbias):
        flapsine=sinegenerator()
        twistsine=sinegenerator()
        Aflap_rt,dAflap_rt=flapsine.getsine(t,Aflap-abs(Aflbias),frez,0,Aflbias)
        Atwist_rt,dAtwist_rt=twistsine.getsine(t,Atwist-abs(Atwbias),frez,dphi,Atwbias)
        self.Aflrtlist.append(Aflap_rt)
        self.Atwrtlist.append(Atwist_rt)
        return frez,Aflap_rt,dAflap_rt,Atwist_rt,dAtwist_rt

    def selfcheck(self):
        plt.figure()
        plt.plot(np.array(self.Aflrtlist)*57.3,'r-',np.array(self.Atwrtlist)*57.3,'g--')
        plt.plot(np.ones_like(self.Aflrtlist)*np.array(self.Aflrtlist).mean(),'r.',np.ones_like(self.Atwrtlist)*np.array(self.Atwrtlist).mean(),'g.')


if __name__=='__main__':
    finchord_tip=0.3
    finchord_root=1.27
    finspan_tip=1.42
    finspan_root=0.27
    Lfront=4
    Lbehind=7
    Lmass=4.269
    Pec_lf=Pectoralfin([Lmass-Lfront-0.5*finspan_root,0,-0.77],[1,1,-1],[finchord_root,finchord_root,0.247],[finspan_root,finspan_tip,0.44],120,(2.477,2.6029,-0.0896,0.01),1.08)
    Pec_rf=Pectoralfin([Lmass-Lfront-0.5*finspan_root,0,0.77],[1,1,1],[finchord_root,finchord_root,0.247],[finspan_root,finspan_tip,0.44],120,(2.477,2.6029,-0.0896,0.01),1.08)
    Pec_lb=Pectoralfin([Lmass-Lbehind-0.5*finspan_root,0,-0.77],[1,1,-1],[finchord_root,finchord_root,0.247],[finspan_root,finspan_tip,0.44],120,(2.477,2.6029,-0.0896,0.01),1.08)
    Pec_rb=Pectoralfin([Lmass-Lbehind-0.5*finspan_root,0,0.77],[1,1,1],[finchord_root,finchord_root,0.247],[finspan_root,finspan_tip,0.44],120,(2.477,2.6029,-0.0896,0.01),1.08)

    frez=0.56
    dphi=pi/2

    Aflaplf=30/57.3
    Aflaprf=30/57.3
    Aflaplb=30/57.3
    Aflaprb=30/57.3

    Atwistlf=45/57.3      
    Atwistrf=45/57.3
    Atwistlb=45/57.3
    Atwistrb=45/57.3

    Aflbiaslf=0
    Aflbiasrf=0
    Aflbiaslb=0/57.3
    Aflbiasrb=0/57.3

    Atwbiaslf=0/57.3
    Atwbiasrf=0/57.3
    Atwbiaslb=0/57.3
    Atwbiasrb=0/57.3

    omega_x=0
    omega_y=0
    omega_z=0

    v_x_r=2.5
    v_y_r=0

    T_xlist=T_ylist=T_zlist=T_mxlist=T_mylist=T_mzlist=[]
    T_xmeanlist=T_ymeanlist=T_zmeanlist=T_mxmeanlist=T_mymeanlist=T_mzmeanlist=[]
    tend=1/frez*10
    for dflap in range(2):
        print(dflap)
        Aflaplf=dflap
        Aflaprf=30-dflap
        for t in np.linspace(0,tend,tend*100+1):
            # generate the sine signal and calculate force
            Fxlf,Fylf,Fzlf,Mxlf,Mylf,Mzlf=Pec_lf.calcforce([v_x_r,v_y_r],Pec_lf.sinemovegene2(t,frez,Aflaplf,Atwistlf,dphi,Aflbiaslf,Atwbiaslf),[omega_x,omega_y,omega_z],0.01)
            # Fxlb,Fylb,Fzlb,Mxlb,Mylb,Mzlb=Pec_lb.calcforce([v_x_r,v_y_r],Pec_lb.sinemovegene2(t,frez,Aflaplb,Atwistlb,dphi,Aflbiaslb,Atwbiaslb),[omega_x,omega_y,omega_z],0.01)
            Fxrf,Fyrf,Fzrf,Mxrf,Myrf,Mzrf=Pec_rf.calcforce([v_x_r,v_y_r],Pec_rf.sinemovegene2(t,frez,Aflaprf,Atwistrf,dphi,Aflbiasrf,Atwbiasrf),[omega_x,omega_y,omega_z],0.01)
            # Fxrb,Fyrb,Fzrb,Mxrb,Myrb,Mzrb=Pec_rb.calcforce([v_x_r,v_y_r],Pec_rb.sinemovegene2(t,frez,Aflaprb,Atwistrb,dphi,Aflbiasrb,Atwbiasrb),[omega_x,omega_y,omega_z],0.01)
            
            # T_x=Fxlf+Fxlb+Fxrf+Fxrb
            # T_y=Fylf+Fylb+Fyrf+Fyrb
            # T_z=Fzlf+Fzlb+Fzrf+Fzrb
            # T_mx=Mxlf+Mxlb+Mxrf+Mxrb
            # T_my=Mylf+Mylb+Myrf+Myrb
            # T_mz=Mzlf+Mzlb+Mzrf+Mzrb
            T_x=Fxlf+Fxrf
            T_y=Fylf+Fyrf
            T_z=Fzlf+Fzrf
            T_mx=Mxlf+Mxrf
            T_my=Mylf+Myrf
            T_mz=Mzlf+Mzrf

            T_xlist.append(T_x)
            T_ylist.append(T_y)
            T_zlist.append(T_z)
            T_mxlist.append(T_mx)
            T_mylist.append(T_my)
            T_mzlist.append(T_mz)
        T_xmeanlist.append(np.mean(T_xlist))
        T_ymeanlist.append(np.mean(T_ylist))
        T_zmeanlist.append(np.mean(T_zlist))
        T_mxmeanlist.append(np.mean(T_mxlist))
        T_mymeanlist.append(np.mean(T_mylist))
        T_mzmeanlist.append(np.mean(T_mzlist))
    fig,ax=plt.subplots(6,1,figsize=(15,7))
    ax[0].plot(T_xmeanlist)
    ax[1].plot(T_ymeanlist)
    ax[2].plot(T_zmeanlist)
    ax[3].plot(T_mxmeanlist)
    ax[4].plot(T_mymeanlist)
    ax[5].plot(T_mzmeanlist)
    plt.show()

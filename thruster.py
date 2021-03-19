import numpy as np
from math import pi, sin, cos, atan, sqrt

from numpy.lib.function_base import piecewise


class Pectoralfin:
    '''
    Define:
    flap up and twist up are positive.
    '''

    def __init__(self, posi, towards, chordcharas, spancharas, Iter, Cs, d):
        self.posi = posi
        self.towards = towards
        # chord_fppercent means the force point position before the twist axis
        self.fin_chord_root, self.fin_chord_tip, self.chord_fppercent = chordcharas
        self.fin_span_root, self.fin_span_tip, self.span_fppercent = spancharas
        self.forceposi = self.getforceposi(posi, towards)
        self.Iter = Iter
        self.C1, self.C2, self.C3, self.C4 = Cs
        self.d = d
        self.vy_piece_old = np.zeros(Iter)

    def getforceposi(self, posi, towards):
        # Note: the following calculation can only satisfies the horizontal instances now!
        # the fix position of fin
        posiX, posiY, posiZ = posi
        toX, toY, toZ = towards
        # get the mass point position
        relativeX = self.chord_fppercent * \
            (self.span_fppercent*(self.fin_chord_tip -
             self.fin_chord_root)+self.fin_chord_root)
        relativeY = 0
        relativeZ = self.span_fppercent * \
            (self.fin_span_tip-self.fin_span_root)+self.fin_span_root

        finforceX = posiX+relativeX*toX
        finforceY = posiY+relativeY*toY
        finforceZ = posiZ+relativeZ*toZ
        return [finforceX, finforceY, finforceZ]

    def calcforce(self, Ulist, finmove, w1, T):
        '''
        d: excahge character
        '''
        Ux, Uy = Ulist
        finfrez, Aflap, dAflap, Atwist, dAtwist = finmove
        w1x, _, w1z = w1
        Vx = Ux
        Uxy = sqrt(Ux**2+Uy**2)
        Iter = self.Iter
        Fx = Fy = Fz = 0
        for i in range(Iter):
            span_piece = (self.fin_span_tip-self.fin_span_root) / \
                Iter*i+self.fin_span_root  # 切片展长
            chord_piece = (self.fin_chord_tip-self.fin_chord_root) / \
                Iter*i+self.fin_chord_root  # 切片弦长
            Atwist_piece = Atwist*i/Iter  # 胸鳍切片每片的扭转角
            dAtwist_piece = dAtwist*i/Iter  # 胸鳍切片每片的扭转角速度
            Vy = Uy + \
                self.posi[0]*w1z-(span_piece*self.towards[2] +
                                  self.posi[2])*w1x+dAflap*span_piece
            Vxy = sqrt(Vx**2+Vy**2)  # 胸鳍剖面合速度大小
            Theta = atan(Vy/(Vx+0.00001))  # 弹道倾角
            alpha_local = Atwist_piece-Theta  # 胸鳍切片当地攻角
            fxPos, fyPos, fzPos = self.finposiforce(
                alpha_local, chord_piece, Vxy, Theta, Aflap)
            k_finfrez = finfrez*(pi*chord_piece)/(Uxy+0.00001)  # 折合频率f
            if k_finfrez > 0.415:
                fxI, fyI, fzI = self.finIforce(
                    Vx, Vy, Atwist_piece, dAtwist_piece, Aflap, chord_piece, T, i)
            else:
                fxI = fyI = fzI = 0
            Fx += fxPos+fxI
            Fy += fyPos+fyI
            Fz += fzPos+fzI
        piece_factor=(self.fin_span_tip-self.fin_span_root)/Iter*self.d
        Fx *= piece_factor
        Fy *= piece_factor
        Fz *= -piece_factor

        forceposiXrt = self.forceposi[0]
        forceposiYrt = self.forceposi[1]+sin(Aflap)*(
            self.fin_span_root+0.44*(self.fin_span_tip-self.fin_span_root))

        forceposiZrt = self.forceposi[2]
        Mx, My, Mz = np.cross(
            np.array([forceposiXrt, forceposiYrt, forceposiZrt]), np.array([Fx, Fy, Fz]))
        return Fx, Fy, Fz, Mx, My, Mz

    def finposiforce(self, alpha_local, chord_piece, Vxy, Theta, Aflap):
        CL = self.C1*alpha_local
        CD = self.C2*alpha_local**2+self.C3*abs(alpha_local)+self.C4
        water_density = 1024
        fxPos = -water_density*chord_piece*Vxy**2/2 * \
            (CL*sin(Theta)+CD*cos(Theta))*self.towards[0]
        fyPos = water_density*chord_piece*Vxy**2/2 * \
            (CL*cos(Theta)-CD*sin(Theta))*cos(Aflap)*self.towards[1]
        fzPos = water_density*chord_piece*Vxy**2/2 * \
            (CL*cos(Theta)-CD*sin(Theta))*sin(Aflap)*self.towards[2]
        return fxPos, fyPos, fzPos

    def finIforce(self, Vx, Vy, Atwist_piece, dAtwist_piece, Aflap, chord_piece, T, pieceID):
        # vx_piece=Vx*cos(Atwist_piece)+Vy*sin(Atwist_piece)
        vy_piece = -Vx*sin(Atwist_piece)+Vy*cos(Atwist_piece)
        dvy_piece = (vy_piece-self.vy_piece_old[pieceID])/T
        self.vy_piece_old[pieceID] = vy_piece
        if dvy_piece >= 10:
            dvy_piece = 0
        water_density = 1024
        lamda22 = 3*pi/16*water_density*chord_piece**2
        fxIi = -lamda22*vy_piece*dAtwist_piece
        fyIi = -lamda22*dvy_piece
        # MzIi=-lamda22*vx_piece*vy_piece
        fxI = (fxIi*cos(Atwist_piece)-fyIi*sin(Atwist_piece))*self.towards[0]
        fyI = (fyIi*cos(Atwist_piece)+fxIi*sin(Atwist_piece)) * \
            cos(Aflap)*self.towards[1]
        fzI = (fyIi*cos(Atwist_piece)+fxIi*sin(Atwist_piece)) * \
            sin(Aflap)*self.towards[2]
        return fxI, fyI, fzI

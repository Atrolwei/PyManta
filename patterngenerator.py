from math import sin,cos,pi

class sinegenerator:
    def __init__(self):
        pass

    def getsine(self,t,A,frez,phi0,bias):
        A_rt=A*sin(2*pi*frez*t+phi0)+bias
        dA_rt=2*pi*frez*A*cos(2*pi*frez*t+phi0)
        return A_rt,dA_rt

class CPGs:
    def __init__(self):
        pass

    def getCPG(self,t,A,frez,phi0,bias):
        A_rt=A*sin(2*pi*frez*t+phi0)+bias
        dA_rt=2*pi*frez*A*cos(2*pi*frez*t+phi0)
        return A_rt,dA_rt
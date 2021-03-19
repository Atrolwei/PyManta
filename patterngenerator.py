from math import sin, cos, pi


def getsine(t, A, frez, phi0, bias):
    A_rt = A*sin(2*pi*frez*t+phi0)+bias
    dA_rt = 2*pi*frez*A*cos(2*pi*frez*t+phi0)
    return A_rt, dA_rt

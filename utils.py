from math import pi


def anglerange_deg(angle):
    if angle > 180:
        angle -= 360
    elif angle <= -180:
        angle += 360
    return angle


def anglerange_rad(angle):
    if angle > pi:
        angle -= 2*pi
    elif angle <= -pi:
        angle += 2*pi
    return angle

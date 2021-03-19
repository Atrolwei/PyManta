from math import sin, cos


class waterdisturb:
    def __init__(self):
        self.V_wg_x = 0
        self.V_wg_y = 0
        self.V_wg_z = 0

    def setdisturbonground(self, VN_w, VY_w, VE_w):
        self.V_wg_x = VN_w
        self.V_wg_y = VY_w
        self.V_wg_z = VE_w

    def steadydisturb(self, psi, vartheta, gamma):
        v_wb_x = self.V_wg_x*cos(psi)*cos(vartheta) + self.V_wg_y * \
            sin(vartheta) - self.V_wg_z*sin(psi)*cos(vartheta)
        v_wb_y = self.V_wg_x*(sin(gamma)*sin(psi) - sin(vartheta)*cos(gamma)*cos(psi)) + self.V_wg_y*cos(
            gamma)*cos(vartheta) + self.V_wg_z*(sin(gamma)*cos(psi) + sin(psi)*sin(vartheta)*cos(gamma))
        v_wb_z = self.V_wg_x*(sin(gamma)*sin(vartheta)*cos(psi) + sin(psi)*cos(gamma)) - self.V_wg_y*sin(
            gamma)*cos(vartheta) + self.V_wg_z*(-sin(gamma)*sin(psi)*sin(vartheta) + cos(gamma)*cos(psi))
        return v_wb_x, v_wb_y, v_wb_z

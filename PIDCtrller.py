class PIDCtrller:
    def __init__(self, Kp, Ki, Kd, T) -> None:
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.T = T
        self.lasterror = 0
        self.sumerror = 0
        self.Ulimit_flag = False
        self.Plimit_flag = False
        self.Ilimit_flag = False

    def setUlimit(self, left_lim, right_lim):
        self.Ulimit_flag = True
        self.Ulimit = (left_lim, right_lim)

    def setPlimit(self, left_lim, right_lim):
        self.Plimit_flag = True
        self.Plimit = (left_lim, right_lim)

    def setIlimit(self, left_lim, right_lim):
        self.Ilimit_flag = True
        self.Ilimit = (left_lim, right_lim)

    def ctrlonce(self, desire, state):
        error = desire-state
        P_effect = self.Kp*error
        if self.Plimit_flag:
            if P_effect > self.Plimit[1]:
                P_effect = self.Plimit[1]
            elif P_effect < self.Plimit[0]:
                P_effect = self.Plimit[0]
        self.sumerror += error*self.T
        I_effect = self.Ki*self.sumerror
        if self.Ilimit_flag:
            if I_effect > self.Ilimit[1]:
                I_effect = self.Ilimit[1]
            elif I_effect < self.Ilimit[0]:
                I_effect = self.Ilimit[0]
        derror = (error-self.lasterror)/self.T
        D_effect = self.Kd*derror

        CtrlU = P_effect+I_effect+D_effect
        if self.Ulimit_flag:
            if CtrlU > self.Ulimit[1]:
                CtrlU = self.Ulimit[1]
            elif CtrlU < self.Ulimit[0]:
                CtrlU = self.Ulimit[0]
        self.lasterror = error
        return CtrlU

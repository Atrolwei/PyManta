import random
from math import atan, pi,sin,cos,sqrt
import numpy as np 


class Sea:
    def __init__(self):
        pass

    def _random_water_disturb(self,X_RANGE,Y_RANGE):
        X_w=random.random()*(X_RANGE[1]-X_RANGE[0])+X_RANGE[0]
        Y_w=random.random()*(Y_RANGE[1]-Y_RANGE[0])+Y_RANGE[0]
        R_w=random.uniform(5,40)
        V=random.uniform(-0.3,0.3)
        V_Psi=random.uniform(-pi,pi)
        VN_w=V*cos(V_Psi)
        VY_w=0
        VE_w=-V*sin(V_Psi)
        return X_w,Y_w,R_w,VN_w,VY_w,VE_w

    def _random_obstacle(self,X_RANGE,Y_RANGE):
        X=random.random()*(X_RANGE[1]-X_RANGE[0]-10)+(X_RANGE[0]+5) #在航行区域内与出发点、目标点不冲突的位置设置障碍物
        Y=random.random()*(Y_RANGE[1]-Y_RANGE[0]-10)+(Y_RANGE[0]+5)
        R=random.uniform(0.5,10)
        return X,Y,R


    def make_random_sea(self,X0,Y0):
        ''' 
        划分一个100m*100m的航行区域，航行起点在左下角，坐标（0，0）m，航行目标点右上角(100,100)m
        随机生成3个圆形洋流区，确定随机的圆心[航行区域内]、半径[5，40]m，洋流方向和洋流速度（形成V_disturb、Psi_disturb、Theta_disturb）。
        使用的时候，如果同时在一到多个洋流区，则等比例地将洋流累加。
        随机生成3个圆形障碍物区域，确定随机的圆心[航行区域内]、半径[0.5，10]m
        '''
        X_BOUND_m=[0,100]
        Y_BOUND_m=[0,100]

        self.X_RANGE=[X0,X_BOUND_m[1]]
        self.Y_RANGE=[Y0,Y_BOUND_m[1]]

        print(
            'The fish will swim in the BOUND of:\n\
            O1==========T\n\
            ||         ||\n\
            ||         ||\n\
            ||         ||\n\
            ||         ||\n\
            ||         ||\n\
            S===========O2\n\
            In which start point S({},{}), target point T({},{}), L_X={} and L_Y={}\n'.format(
                self.X_RANGE[0],
                self.Y_RANGE[0],
                self.X_RANGE[1],
                self.Y_RANGE[1],
                X_BOUND_m[1],
                Y_BOUND_m[1]))

        self.water_disturb_array=[]
        self.obstacle_array=[]
        for _ in range(3):
            w_dst=self._random_water_disturb(self.X_RANGE,self.Y_RANGE)
            self.water_disturb_array.append(w_dst)
            obs=self._random_obstacle(self.X_RANGE,self.Y_RANGE)
            self.obstacle_array.append(obs)

    def get_water_disturbance(self,X,Y):
        # Decide the water disturbance velocity
        VN_w_total,VY_w_total,VE_w_total=0,0,0
        disturbance_num=0
        for w_dst in self.water_disturb_array:
            X_w,Y_w,R_w,VN_w,VY_w,VE_w=w_dst
            L_X=X-X_w
            L_Y=Y-Y_w
            dist_to_w_dst=sqrt(L_X**2+L_Y**2)
            if dist_to_w_dst<R_w:
                VN_w_total+=VN_w
                # VY_w_total+=VY_w
                VE_w_total+=VE_w
                disturbance_num+=1
        if disturbance_num>0:
            VN_w_total/=disturbance_num
            # VY_w_total/=disturbance_num
            VE_w_total/=disturbance_num
        return VN_w_total, VY_w_total, VE_w_total

    def get_obstacle(self,X,Y,Psi_manta):
        local_obs=[]
        for obs in self.obstacle_array:
            X_obs,Y_obs,R_obs=obs
            L_X=-(X-X_obs)
            L_Y=-(Y-Y_obs)
            # 障碍物距离
            dist_to_obs=sqrt(L_X**2+L_Y**2)-R_obs
            if dist_to_obs<=10:  #设探测距离10m
                # 在大地惯性系下给出障碍物方向，北偏西[0,pi]为正，北偏东(-pi,0)为负，若要相对鱼体需再转换
                if L_X>0:
                    Psi_obs=-(pi/2-atan(L_Y/L_X))
                elif L_X<0:
                    Psi_obs=pi/2+atan(L_Y/L_X)
                elif L_Y>=0:
                    Psi_obs=0
                else:
                    Psi_obs=pi

                Psi_rel=Psi_obs-Psi_manta # 相对航行器体坐标系下的障碍物方向，左侧为正，右侧为负
                local_obs.append([Psi_obs,Psi_rel,dist_to_obs])
        return local_obs

    def iscollision(self,X,Y):
        iscollision=False
        for obs in self.obstacle_array:
            X_o,Y_o,R_o=obs
            L_X=X-X_o
            L_Y=Y-Y_o
            dist_to_obs=sqrt(L_X**2+L_Y**2)
            if dist_to_obs<=R_o:
                iscollision=True
        return iscollision

    def isunbalance(self,vel,pitch,roll):
        isunbalance=False
        if abs(pitch)>pi/2 or abs(roll)>pi/2 or abs(vel)>10:
            isunbalance=True
        return isunbalance

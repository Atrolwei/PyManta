import numpy as np
from math import pi
from PIDCtrller import PIDCtrller
from manta import Manta


CTRL_CYCLE_TAIL = 0.1  # 每0.1s进行一次尾鳍控制
CTRL_CYCLE_PEC = 1  # 每1s进行一次胸鳍控制
STEPTIME = 0.001  # 解算步长1ms
T_END = 10        # 每Episode总时长15s


def get_action(t, state, action_old):
    action = list(action_old)
    # 每0.1s更新一次尾鳍控制量，每1s更新一次胸鳍控制量
    if int(t*1000) % int(CTRL_CYCLE_TAIL*1000) == 0:
        dz = -pitchCtrller.ctrlonce(0, state[3]*57.3)
        action[11] = dz
        action[12] = dz

        if int(t*1000) % int(CTRL_CYCLE_PEC*1000) == 0:
            # get pectoral fin position
            frez = 0.5
            Aflap_l = 30/57.3
            Atwist_l = 30/57.3
            dphi_l = pi/2
            Aflbias_l = 0
            Atwbias_l = 0

            Aflap_r = 30/57.3
            Atwist_r = 30/57.3
            dphi_r = pi/2
            Aflbias_r = 0
            Atwbias_r = 0
            action[0:11] = [frez,
                            Aflap_l, Atwist_l, dphi_l, Aflbias_l, Atwbias_l,
                            Aflap_r, Atwist_r, dphi_r, Aflbias_r, Atwbias_r]
    return action


if __name__ == "__main__":
    # 定义仿真环境
    env = Manta()
    done = False      # 初始化训练状态
    # 状态量state介绍
    # 设为 x,y,z,vartheta,psi,gamma,vx,vy,vz,wx,wy,wz
    # 其中，x,y,z为航行器在惯性系下的三轴位置，x沿航行器纵轴指向头部，y沿航行器中纵剖面指向上，z轴按右手定则指向右
    # vartheta,psi,gamma分别为航行器欧拉角形式的姿态角，俯仰角、偏航角和滚动角
    # vx,vy,vz分别为航行器惯性系下三轴速度
    # wx,wy,wz分别为航行器体轴系下三轴角速度
    # 初始状态state0=[0,-5,0,0,0,0,0.1,0,0,0,0,0]

    # 环境初始化
    state = env.reset(tend=T_END)

    # 临时变量，用于周期性运动的执行机构的动作记录，action定义同
    action_old = (0.5,                        # 胸鳍运动频率，保持0.5Hz，设左右胸鳍频率相同且摆扭同频
                  30/57.3, 30/57.3, pi/2, 0, 0,   # 左胸鳍运动参数：摆幅、扭幅、扭摆相位差、摆动偏置、扭转偏置
                  30/57.3, 30/57.3, pi/2, 0, 0,   # 右胸鳍运动参数：摆幅、扭幅、扭摆相位差、摆动偏置、扭转偏置
                  0, 0)                        # 尾鳍摆幅，向下为正
    # action参数变量名及取值范围：
    # Aflap：胸鳍摆幅，[0/57.3，30/57.3]（rad）
    # Atwist：胸鳍扭幅，[0/57.3，30/57.3]（rad）
    # dphi：胸鳍扭摆相位差，dphi=扭转初始相位-摆动初始相位[-180/57.3，180/57.3]（rad）
    # Aflbias：胸鳍摆动偏置，向上为正，[-30/57.3，30/57.3]（rad）
    # Atwbias：胸鳍扭转偏置，向上为正，[-30/57.3，30/57.3]（rad）
    # dzl为左尾鳍摆幅，dzr为右尾鳍摆幅，[-30/57.3，30/57.3]（rad）

    pitchCtrller = PIDCtrller(0.13, 0, 0.1, 0.1)
    pitchCtrller.setUlimit(-30/57.3, 30/57.3)

    # 执行1Episode
    if not done:
        for t in np.linspace(0, T_END, 1000*T_END+1):
            action = get_action(t, state, action_old)
            state, reward, done = env.step(action, steptime=STEPTIME)
            action_old = action
            print('time is {} and reward is {}\n'.format(t, reward))
    # env.render()

class Delayer:
    def __init__(self,T) -> None:
        self.input=0
        self.output=0
        self.T=T 

    def run(self,input,h):
        output_dot=(input-self.output)/self.T
        self.output+=h*output_dot
        return self.output,output_dot


if __name__=='__main__':
    # For test
    import numpy as np
    import matplotlib.pyplot as plt
    t=np.linspace(0,1,1001)
    Y0=np.sin(2*np.pi*t)
    delayer=Delayer(0.01)
    Y1=np.zeros_like(t)
    for i in range(len(t)):
        input=Y0[i]
        Y1[i],_=delayer.run(input,0.001)
    
    plt.figure()
    plt.plot(t,Y0,'r',t,Y1,'b')
    plt.show()
    
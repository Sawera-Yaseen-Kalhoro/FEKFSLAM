from GFLocalization import *
from EKF import *
from DR_3DOFDifferentialDrive import *
from DifferentialDriveSimulatedRobot import *
from Feature import *

class EKF_3DOFDifferentialDriveCtVelocity(GFLocalization, DR_3DOFDifferentialDrive, EKF):

    def __init__(self, kSteps, robot, *args):

        self.x0 = np.zeros((6, 1))  # initial state x0=[x y z psi u v w r]^T
        self.P0 = np.zeros((6, 6))  # initial covariance

        # this is required for plotting
        self.index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1),
                 IndexStruct("u", 3, 2), IndexStruct("v", 4, 3), IndexStruct("yaw_dot", 5, None)]

        # TODO: To be completed by the student
        self.t_1 = 0
        self.t = 0
        self.Dt = self.t - self.t_1
        super().__init__(self.index, kSteps, robot, self.x0, self.P0, *args)

    def f(self, xk_1, uk):
        # TODO: To be completed by the student

        etak_1 = xk_1[0:3]
        nuk_1 = xk_1[3:6]

        etak = Pose3D(etak_1).oplus(nuk_1*self.robot.dt)
        nuk = nuk_1

        xk_bar = np.vstack((etak,nuk))

        return xk_bar

    def Jfx(self, xk_1, uk):
        # TODO: To be completed by the student
        etak_1 = xk_1[0:3]
        nuk_1 = xk_1[3:6]

        J = np.block([[Pose3D(etak_1).J_1oplus(nuk_1*self.robot.dt), Pose3D(etak_1).J_2oplus()*self.robot.dt],[np.zeros((3,3)), np.identity(3)]])

        return J

    def Jfw(self, xk_1):
        # TODO: To be completed by the student
        etak_1 = xk_1[0:3]
        vk_1 = xk_1[3:6]

        J = np.block([[(Pose3D(etak_1).J_2oplus()*(self.Dt**2/2))],[np.identity(3) * self.robot.dt]])

        return J

    def h(self, xk):  #:hm(self, xk):
        # TODO: To be completed by the student

        if self.yaw_measure: # if there is compass measurement
            h = np.array([xk[2],xk[3],xk[5]]) # [psi_k, uk, rk].T
        else:
            h = np.array([xk[3],xk[5]]) # [uk,rk].T

        return h  # return the expected observations

    def GetInput(self):
        """

        :return: uk,Qk:
        """
        # TODO: To be completed by the student

        uk = np.array([None])

        Qk = self.robot.Qsk

        return uk, Qk

    def GetMeasurements(self):  # override the observation model
        """

        :return: zk, Rk, Hk, Vk
        """
        # TODO: To be completed by the student

        # get yaw reading from compass
        (z_compass,R_compass) = self.robot.ReadCompass() 
        # returns yaw reading (already with noise) and std of yaw reading

        if z_compass: # if there is measurement
            # convert zk and Rk into np array
            z_compass = np.array([[z_compass]])
            R_compass = np.array([[R_compass]])
            # set Hk and Vk according to the corresponding line in the observation model
            H_compass = np.array([[0,0,1,0,0,0]])
            V_compass = np.array([[1]])
            self.yaw_measure = True
        else: # if there's no measurement
            z_compass = np.array([0])
            R_compass = np.array([0])
            H_compass = np.array([0,0,0,0,0,0])
            V_compass = np.array([0,0,0])
            self.yaw_measure = False

        # get pulse reading from encoders
        (z_encoder,R_encoder) = self.robot.ReadEncoders() 
        # returns np.array([[NL,NR]]).T pulses (already with noise) and covariance of encoders (diagonal matrix)

        # convert from pulses to robot velocities
        N_to_V = 2*np.pi*self.robot.wheelRadius/(self.robot.pulse_x_wheelTurns * self.robot.dt) # conversion factor
        v = z_encoder*N_to_V

        v_L = v[0,0] # left wheel velocity
        v_R = v[1,0] # right wheel velocity
        
        uk_encoder = (v_L+v_R)/2 # robot linear velocity from encoder
        rk_encoder = (v_R-v_L)/self.wheelBase # robot angular velocity from encoder
        z_vel = np.array([[uk_encoder,rk_encoder]]).T

        # convert covariance from pulses to velocities
        # A matrix in the magic table entry
        A = np.array([[0.5,0.5],[-1/self.robot.wheelBase, 1/self.robot.wheelBase]])
        A = A @ np.diag([N_to_V, N_to_V]) 

        R_vel = A @ R_encoder @ A.T # covariance of velocity from encoders
        
        # assign H and V according to the corresponding lines in the observation model
        H_vel = np.array([[0,0,0,1,0,0],[0,0,0,0,0,1]])
        V_vel = np.eye(2)

        # combine measurements from compass and encoders, if there is compass reading
        if self.yaw_measure:
            zk = np.vstack((z_compass,z_vel))
            Rk = scipy.linalg.block_diag(R_compass, R_vel)
            Hk = np.vstack((H_compass,H_vel))
            Vk = scipy.linalg.block_diag(V_compass,V_vel)
        else:
            zk = z_vel
            Rk = R_vel
            Hk = H_vel
            Vk = V_vel

        return zk, Rk, Hk, Vk
    
    # def GetMeasurements(self):  # override the observation model
    #     """

    #     :return:
    #     """
    #     zk = np.zeros((0, 1))  # empty vector
    #     Rk = np.zeros((0, 0))  # empty matrix

    #     self.yaw = False
    #     self.vel = False

    #     Hk, Vk = np.zeros((0,6)), np.zeros((0,0))
    #     H_yaw, V_yaw = np.array([[0,0,1,0,0,0]]), np.eye(1)

    #     z_yaw, sigma2_yaw = self.robot.ReadCompass()

    #     if z_yaw.size>0:  # if there is a measurement
    #         zk, Rk = np.block([[zk], [z_yaw]]), scipy.linalg.block_diag(Rk, sigma2_yaw)
    #         Hk, Vk = np.block([[Hk], [H_yaw]]), scipy.linalg.block_diag(Vk, V_yaw)
    #         self.yaw = True

    #     n, Rn = self.robot.ReadEncoders(); L=0; R=1  # read sensors

    #     if n.size>0:  # if there is a measurement
    #         self.vel=True

    #         H_n= np.array([[ 0,0,0,self.Kn_inv[0,0],0,self.Kn_inv[0,1]],
    #                         [ 0,0,0,self.Kn_inv[1,0],0,self.Kn_inv[1,1]]])

    #         zk, Rk = np.block([[zk], [n]]), scipy.linalg.block_diag(Rk, Rn)
    #         Hk, Vk = np.block([[Hk], [H_n]]), scipy.linalg.block_diag(Vk, np.eye(2))

    #     if zk.shape[0] == 0: # if there is no measurement
    #         return np.zeros((1,0)), np.zeros((0,0)),np.zeros((1,0)), np.zeros((0,0))
    #     else:
    #         return zk, Rk, Hk, Vk

if __name__ == '__main__':

    M = [CartesianFeature(np.array([[-40, 5]]).T),
           CartesianFeature(np.array([[-5, 40]]).T),
           CartesianFeature(np.array([[-5, 25]]).T),
           CartesianFeature(np.array([[-3, 50]]).T),
           CartesianFeature(np.array([[-20, 3]]).T),
           CartesianFeature(np.array([[40,-40]]).T)]  # feature map. Position of 2 point features in the world frame.

    xs0 = np.zeros((6,1))  # initial simulated robot pose

    robot = DifferentialDriveSimulatedRobot(xs0, M)  # instantiate the simulated robot object
    kSteps = 5000

    xs0 = np.zeros((6, 1))  # initial simulated robot pose
    index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1), 
             IndexStruct("u", 3, 2), IndexStruct("v", 4, None), IndexStruct("r", 5, 3)]

    x0 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
    # P0 = np.diag(np.array([0.0, 0.0, 0.0, 0.5 ** 2, 0 ** 2, 0.05 ** 2])) #default
    P0 = np.diag(np.array([0.0, 0.0, 0.0, 0.1 ** 2, 0 ** 2, 0.01 ** 2]))

    dd_robot = EKF_3DOFDifferentialDriveCtVelocity(kSteps, robot)  # initialize robot and KF
    dd_robot.LocalizationLoop(x0, P0, np.array([[0.5, 0.03]]).T)  # run localization loop

    exit(0)
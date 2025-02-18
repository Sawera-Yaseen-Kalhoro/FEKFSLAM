from GFLocalization import *
from EKF import *
from DR_3DOFDifferentialDrive import *
from DifferentialDriveSimulatedRobot import *
from Feature import *

class EKF_3DOFDifferentialDriveInputDisplacement(GFLocalization, DR_3DOFDifferentialDrive, EKF):
    """
    This class implements an EKF localization filter for a 3 DOF Diffenteial Drive using an input displacement motion model incorporating
    yaw measurements from the compass sensor.
    It inherits from :class:`GFLocalization.GFLocalization` to implement a localization filter, from the :class:`DR_3DOFDifferentialDrive.DR_3DOFDifferentialDrive` class and, finally, it inherits from
    :class:`EKF.EKF` to use the EKF Gaussian filter implementation for the localization.
    """
    def __init__(self, kSteps, robot, *args):
        """
        Constructor. Creates the list of  :class:`IndexStruct.IndexStruct` instances which is required for the automated plotting of the results.
        Then it defines the inital stawe vecto mean and covariance matrix and initializes the ancestor classes.

        :param kSteps: number of iterations of the localization loop
        :param robot: simulated robot object
        :param args: arguments to be passed to the base class constructor
        """

        self.dt = 0.1  # dt is the sampling time at which we iterate the KF
        x0 = np.zeros((3, 1))  # initial state x0=[x y z psi u v w r]^T
        P0 = np.zeros((3, 3))  # initial covariance

        # this is required for plotting
        index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("z", 2, 0), IndexStruct("yaw", 3, 1)]

        self.t_1 = 0
        self.t = 0
        self.Dt = self.t - self.t_1
        super().__init__(index, kSteps, robot, x0, P0, *args)

    def f(self, xk_1, uk):
        # TODO: To be completed by the student
        
        # assuming xk_1 is a Pose3D object

        xk_bar = Pose3D(xk_1).oplus(uk)

        return xk_bar

    def Jfx(self, xk_1, uk):
        # TODO: To be completed by the student

        J = Pose3D(xk_1).J_1oplus(uk)

        return J

    def Jfw(self, xk_1):
        # TODO: To be completed by the student

        J = Pose3D(xk_1).J_2oplus()

        return J

    def h(self, xk):  #:hm(self, xk):
        # TODO: To be completed by the student
        z = xk[2]

        return z  # return the expected observations

    def GetInput(self):
        """

        :return: uk,Qk
        """
        # TODO: To be completed by the student
        (pulses,cov_pulses) = self.robot.ReadEncoders() # returns pulses (already with noise) and covariance of pulse

        # convert to displacement
        N_to_d = 2*np.pi*self.robot.wheelRadius/self.robot.pulse_x_wheelTurns # conversion factor
        d = pulses*N_to_d

        dr = (d[0,0]+d[1,0])/2
        dthetar = (d[1,0]-d[0,0])/self.robot.wheelBase
        uk = Pose3D(np.array([[dr,0,dthetar]]).T)

        # converting covariance from pulses to displacement
        # A matrix in the magic table entry
        A = np.array([[0.5,0.5],[0,0],[-1/self.robot.wheelBase, 1/self.robot.wheelBase]])
        A = A @ np.diag([N_to_d, N_to_d]) 

        Qk = A @ cov_pulses @ A.T

        return uk, Qk

    def GetMeasurements(self):  # override the observation model
        """

        :return: zk, Rk, Hk, Vk
        """
        # TODO: To be completed by the student

        # zk = mean of the measurement
        # Rk = covariance of the measurement
        # Hk = coeff matrix of xk in the observation model
        # Vk = noise coeff

        (zk,Rk) = self.robot.ReadCompass() # returns yaw reading (already with noise) and std of yaw reading

        if zk: # if there is measurement
            # convert zk and Rk into np array
            zk = np.array([[zk]])
            Rk = np.array([[Rk]])
            # set Hk and Vk according to the observation model
            Hk = np.block([np.array([[0,0,1]]),np.zeros((1,self.nf*2))])
            Vk = np.array([[1]])
        else: # if there's no measurement
            # zk = np.array([None])
            # Rk = np.array([None])
            # Hk = np.array([None,None,None])
            # Vk = np.array([None])

            zk = np.zeros((1,0))
            Rk = np.zeros((1,0))
            Hk = np.zeros((1,3))
            Vk = np.zeros((1,0))

        return zk, Rk, Hk, Vk


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
    index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1)]

    x0 = np.zeros((3, 1))
    P0 = np.zeros((3, 3))

    dd_robot = EKF_3DOFDifferentialDriveInputDisplacement(kSteps,robot)  # initialize robot and KF
    dd_robot.LocalizationLoop(x0, P0, np.array([[0.5, 0.03]]).T)

    exit(0)
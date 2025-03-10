from FEKFSLAM import *
from FEKFMBL import *
from EKF_3DOFDifferentialDriveInputDisplacement import *
from Pose import *
from blockarray import *
from MapFeature import *
import numpy as np
from FEKFSLAMFeature import *

class FEKFSLAM_3DOFDD_InputVelocityMM_2DCartesianFeatureOM(FEKFSLAM2DCartesianFeature, FEKFSLAM, EKF_3DOFDifferentialDriveInputDisplacement):
    def __init__(self, *args):

        self.Feature = globals()["CartesianFeature"]
        self.Pose = globals()["Pose3D"]
       
        super().__init__(*args)


    # def GetFeatures(self):
    # Get features is inherited from EKF_3DOFDifferentialDriveInputDisplacement


if __name__ == '__main__':

    M = [CartesianFeature(np.array([[-40, 5]]).T),
           CartesianFeature(np.array([[-5, 40]]).T),
           CartesianFeature(np.array([[-5, 25]]).T),
           CartesianFeature(np.array([[-3, 50]]).T),
           CartesianFeature(np.array([[-20, 3]]).T),
           CartesianFeature(np.array([[40,-40]]).T)]  # feature map. Position of 2 point features in the world frame.

    xs0 = np.zeros((6, 1))
    kSteps = 5000
    alpha = 0.99

    index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1)]

    robot = DifferentialDriveSimulatedRobot(xs0, M)  # instantiate the simulated robot object

    x0 = Pose3D(np.zeros((3, 1)))

    # Enforce the map features in the initial state
    # Rf = np.eye(2)*5      # initial feature covariance
    # for feature in M:
    #     x0 = np.block([[x0],[feature]]) # without initial noise
    #     # x0 = np.block([[x0],[feature + np.random.multivariate_normal(np.zeros(2),Rf).reshape((2,1))]])

    # dr_robot = DR_3DOFDifferentialDrive(index, kSteps, robot, x0)
    robot.SetMap(M)

    auv = FEKFSLAM_3DOFDD_InputVelocityMM_2DCartesianFeatureOM([], alpha, kSteps, robot)

    P0 = np.zeros((3, 3))
    
    # Enforce the map features in the initial state
    # Initialize with all zeros
    # P0 = np.zeros((x0.shape[0],x0.shape[0]))
    # Initialize with initial uncertainty (Note: comment the line above before uncommenting the lines below)
    # for feature in M:
    #     P0 = scipy.linalg.block_diag(P0,Rf)

    # Add features observed at the beginning of the simulation
    zf0,Rf0 = auv.GetFeatures()
    auv.AddNewFeatures(x0,P0,zf0,Rf0)

    usk=np.array([[0.5, 0.03]]).T
    auv.LocalizationLoop(x0, P0, usk)

    exit(0)

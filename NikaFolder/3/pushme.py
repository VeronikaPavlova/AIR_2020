import sys
sys.path.append('../../build')
import libry as ry
import numpy as np
import time

def main():

    RealWorld = ry.Config()
    RealWorld.addFile('../../scenarios/pandasTable.g')
    RealWorld.addFrame("obj0", "table", "type:ssBox size:[.1 .1 .2 .02] color:[1. 0. 0.], contact, logical={ object }, joint:rigid, Q:<t(0 0 .15)>, mass:0.2" )


    S = RealWorld.simulation(ry.SimulatorEngine.physx, True)
    S.addSensor("camera")

    C = ry.Config()
    C.addFile('../../scenarios/pandasTable.g')
    V = ry.ConfigurationViewer()
    V.setConfiguration(C)
    cameraFrame = C.frame("camera")

    q0 = C.getJointState()
    R_gripper = C.frame("R_gripper")
    R_gripper.setContact(1)
    L_gripper = C.frame("L_gripper")
    L_gripper.setContact(1)

    input("Press Enter to continue...")

if __name__ == "__main__":
    # execute only if run as a script
    main()
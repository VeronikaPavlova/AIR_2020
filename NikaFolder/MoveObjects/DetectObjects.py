import sys

sys.path.append('../../build')
import libry as ry
import numpy as np
import time

import cv2 as cv

from centroidtracker import CentroidTracker


def detectObjects(rgb, depth):
    # canny edge detection
    edges = cv.Canny(rgb, 20, 60)
    edges = cv.dilate(edges, None, iterations=1)
    edges = cv.erode(edges, None, iterations=1)

    # find contours in edges
    contours, hierarchy = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    good_contour = []  # good contours which are identified
    hull_list = []  # convex hull - the end points of rectangle

    for cnt in contours:
        # if small contour area - ignore
        if cv.contourArea(cnt) < 200:
            continue

        # Ignore the objects which are too far away or in the background
        mask = np.zeros(rgb.shape[:2], np.uint8)
        cv.drawContours(mask, cnt, -1, 255, 1)
        mean_color = cv.mean(rgb, mask=mask)
        mean_depth = cv.mean(depth, mask=mask)

        # compute the approx shape - try to fit Poly in shape
        approx = cv.approxPolyDP(cnt, 0.05 * cv.arcLength(cnt, True), True)

        if mean_color[2] > 165 and mean_depth[0] < 2 and len(approx) < 5:

            good_contour.append(cnt)
            # get convex hull
            hull = cv.convexHull(approx)
            hull_list.append(hull)

            # compute the center of the contour
            M = cv.moments(approx)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # draw the contour and center of the shape on the image
                cv.circle(rgb, (cX, cY), 7, (255, 255, 255), -1)
                cv.putText(rgb, "center", (cX - 20, cY - 20),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv.drawContours(rgb, hull, -1, (255, 0, 255), 3)

    return good_contour, edges, hull_list


def main():
    # Initialization
    np.random.seed(25)

    RealWorld = ry.Config()

    RealWorld.addFile("../../scenarios/challenge.g")
    # Change color of objects
    for o in range(1, 30):
        color = list(np.random.choice(np.arange(0, 1, 0.05), size=2)) + [1]
        name = "obj%i" % o
        RealWorld.frame(name).setColor(color)

    S = RealWorld.simulation(ry.SimulatorEngine.physx, True)
    S.addSensor("camera")

    C = ry.Config()
    C.addFile('../../scenarios/pandasTable.g')
    V = ry.ConfigurationViewer()
    V.setConfiguration(C)
    cameraFrame = C.frame("camera")

    # q0 = C.getJointState()
    R_gripper = C.frame("R_gripper")
    R_gripper.setContact(1)
    L_gripper = C.frame("L_gripper")
    L_gripper.setContact(1)

    # Initialize centroid tracker
    ct = CentroidTracker()

    # the focal length
    f = 0.895
    f = f * 360.
    # the relative pose of the camera
    # pcl.setRelativePose('d(-90 0 0 1) t(-.08 .205 .115) d(26 1 0 0) d(-1 0 1 0) d(6 0 0 1) ')
    fxfypxpy = [f, f, 320., 180.]

    # points = []
    tau = .01
    t = 0

    while True:
        time.sleep(0.01)
        t += 1
        # grab sensor readings from the simulation
        # q = S.get_q()
        if t % 10 == 0:
            [rgb, depth] = S.getImageAndDepth()  # we don't need images with 100Hz, rendering is slow
            points = S.depthData2pointCloud(depth, fxfypxpy)
            cameraFrame.setPointCloud(points, rgb)
            V.recopyMeshes(C)
            V.setConfiguration(C)

            good_contours, edges, hull = detectObjects(rgb=rgb, depth=depth)

            # find hough lines
            # lines = cv.HoughLines(edges, 0.6, np.pi/120, 50)
            objects = ct.update(hull)
            good = np.zeros(rgb.shape, np.uint8)
            cv.drawContours(good, good_contours, -1, (0, 255, 0), 1)

            if len(rgb) > 0: cv.imshow('OPENCV - rgb', rgb)
            if len(edges) > 0: cv.imshow('OPENCV - gray_bgd', edges)
            if len(good) > 0: cv.imshow('OPENCV - depth', good)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        S.step([], tau, ry.ControlMode.none)


if __name__ == "__main__":
    main()

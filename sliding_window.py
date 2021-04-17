#!/usr/bin/env python3

import cv2
import depthai as dai
import matplotlib.pyplot as plt
from collections import deque
import numpy as np



class DataPlot:
    def __init__(self, max_entries=20):
        self.axis_x = deque(maxlen=max_entries)
        self.axis_y = deque(maxlen=max_entries)

        self.max_entries = max_entries

        self.buf1 = deque(maxlen=5)
        self.buf2 = deque(maxlen=5)

    def add(self, x, y):
        self.axis_x.append(x)
        self.axis_y.append(y)


class RealtimePlot:
    def __init__(self, axes):
        self.axes = axes

        self.lineplot, = axes.plot([], [], "ro-")

    def plot(self, dataPlot):
        self.lineplot.set_data(dataPlot.axis_x, dataPlot.axis_y)

        self.axes.set_xlim(min(dataPlot.axis_x), max(dataPlot.axis_x))
        ymin = 0
        ymax = max(dataPlot.axis_y) + 10
        self.axes.set_ylim(ymin, ymax)
        self.axes.relim();


stepSize = 0.01

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - two mono (grayscale) cameras
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()
spatialLocationCalculator = pipeline.createSpatialLocationCalculator()

xoutDepth = pipeline.createXLinkOut()
xoutSpatialData = pipeline.createXLinkOut()
xinSpatialCalcConfig = pipeline.createXLinkIn()

xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

# MonoCamera
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

outputDepth = True
outputRectified = False
lrcheck = False
subpixel = False

# StereoDepth
stereo.setOutputDepth(outputDepth)
stereo.setOutputRectified(outputRectified)
stereo.setConfidenceThreshold(0)

stereo.setLeftRightCheck(lrcheck)
stereo.setSubpixel(subpixel)

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)
spatialLocationCalculator.setWaitForConfigInput(False)

topLeft = None
bottomRight = None
config = None

x = 0
configX = None

for i in range(15):
    j = 2
    topLeft1 = dai.Point2f(i/15 + 0.025, j/5 + 0.05)
    bottomRight1 = dai.Point2f(i/15 + 0.065, j/5 + 0.1)
    config1 = dai.SpatialLocationCalculatorConfigData()
    config1.depthThresholds.lowerThreshold = 50
    config1.depthThresholds.upperThreshold = 5000
    config1.roi = dai.Rect(topLeft1, bottomRight1)
    spatialLocationCalculator.initialConfig.addROI(config1)
    if i == 4 and j == 2:
        topLeft = topLeft1
        bottomRight = bottomRight1
        config = config1
        configX = x
    x += 1

spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

# Pipeline is defined, now we can connect to the device
with dai.Device(pipeline) as device:
    device.startPipeline()

    # Output queue will be used to get the depth frames from the outputs defined above
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")

    color = (0, 255, 0)

    fig, axes = plt.subplots()
    plt.title('Plotting Data')

    data = DataPlot();
    dataPlotting = RealtimePlot(axes)

    count = 0
    w = 600
    h = 600
    # Make empty black image
    projectionImage = np.zeros((h, w, 3), np.uint8)

    while True:
        count += 1
        inDepth = depthQueue.get() # Blocking call, will wait until a new data has arrived
        inDepthAvg = spatialCalcQueue.get() # Blocking call, will wait until a new data has arrived

        depthFrame = inDepth.getFrame()
        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

        spatialData = inDepthAvg.getSpatialLocations()

        i = 0
        projectionImage[:] = [0, 0, 0]
        xy = ""
        xysplit = ""
        leftCoords = None
        midCoords = None
        rightCoords = None

        for depthData in spatialData:
            roi = depthData.config.roi
            roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
            xmin = int(roi.topLeft().x)
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)

            x = int(xmin/1.5)
            y = 599 - int(depthData.spatialCoordinates.z/10)
            xy = xy + str(x) + ":" + str(y) + " "
            xysplit = xy.split(" ")
            cv2.rectangle(projectionImage, (x, y), (x+5, y+5), color, 2)

            fontType = cv2.FONT_HERSHEY_TRIPLEX
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, 2)

            if i == configX:
                data.add(count, int(depthData.spatialCoordinates.z))
                dataPlotting.plot(data)
                plt.pause(0.001)
            i += 1
        leftCoords = int(xysplit[6][4:])
        midCoords = int(xysplit[7][4:])
        rightCoords = int(xysplit[8][4:])
        print(f"leftCoords: {leftCoords}, midCoords: {midCoords}, rightCoords: {rightCoords}")
        if leftCoords < rightCoords and leftCoords < midCoords:
            print("Turn Left")
        elif rightCoords < leftCoords and rightCoords < midCoords:
            print("Turn Right")
        elif midCoords < leftCoords and midCoords < rightCoords:
            print("Forward")
        cv2.imshow("projection", projectionImage)
        cv2.imshow("depth", depthFrameColor)

        newConfig = False
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('w'):
            if topLeft.y - stepSize >= 0:
                topLeft.y -= stepSize
                bottomRight.y -= stepSize
                newConfig = True
        elif key == ord('a'):
            if topLeft.x - stepSize >= 0:
                topLeft.x -= stepSize
                bottomRight.x -= stepSize
                newConfig = True
        elif key == ord('s'):
            if bottomRight.y + stepSize <= 1:
                topLeft.y += stepSize
                bottomRight.y += stepSize
                newConfig = True
        elif key == ord('d'):
            if bottomRight.x + stepSize <= 1:
                topLeft.x += stepSize
                bottomRight.x += stepSize
                newConfig = True
        elif key == ord('e'):
            topLeft.x += 0.01
            topLeft.y += 0.01
            bottomRight.x -= 0.01
            bottomRight.y -= 0.01
            newConfig = True
        elif key == ord('r'):
            topLeft.x -= 0.01
            topLeft.y -= 0.01
            bottomRight.x += 0.01
            bottomRight.y += 0.01
            newConfig = True
        if newConfig:
            config.roi = dai.Rect(topLeft, bottomRight)
            cfg = dai.SpatialLocationCalculatorConfig()
            cfg.addROI(config)
            spatialCalcConfigInQueue.send(cfg)
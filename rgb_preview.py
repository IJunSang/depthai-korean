import cv2
import depthai as dai

# Create pipeline
# represent the pipeline, set of nodes and connections
pipeline = dai.Pipeline()

# Define source and output
# ColorCamera node. For use with color sensors
camRgb = pipeline.createColorCamera()
# XLinkOut node. Send messages over XLink
xoutRgb = pipeline.createXLinkOut()
# specify xlink name to use
xoutRgb.setStreamName("rgb")

# Properties
# set preview output size
# camRgb.setPreviewSize(300, 300)
# set planar or interleaved data of preview output frames (RGB or BGR planar/interleave configure)
# Image Post Processing converts YUV420 planar frames fro mthe ISP into frames
# interleaved(packed) -> yuyv|yuyv|yuyv| ...
# planar -> yyyy... | uuuu... | vvv...
camRgb.setInterleaved(False)
# set color order of preview output images / RGB or BGR
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)

#Linking
# link current output to input
camRgb.preview.link(xoutRgb.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    print('Connected cameras: ', device.getConnectedCameras())
    print('Usb speed: ', device.getUsbSpeed().name)

    # gets output queue corresponding to stream name. if name exists, throaw and sets queue option
    # Output queue will be used to get the rgb frames from the output defined above
    qRgb = device.getOutputQueue(name = "rgb", maxSize=4, blocking=False)

    while True:
        inRgb = qRgb.get() # blocking call, will wait until a new data has arrived
        
        # Retrieve 'bgr' frame
        cv2.imshow("rgb", inRgb.getCvFrame()) # get BGR frame compatible with use in opencv function

        if cv2.waitKey(1) == ord('q'):
            break
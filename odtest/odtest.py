
# -*- coding: utf-8 -*-
#
#
# env-version:
# macos 10.14.2
# python 3.5.6 :: anconda 4.5.11
# opencv 3.4.2
# imutils 0.5.2
# numpy 1.15.2

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2


ap=argparse.ArgumentParser()
ap.add_argument("-p","--prototxt",required=True,help="path to prototxt file")
ap.add_argument("-m","--model",required=True,help="path to model")

args=vars(ap.parse_args())

CLASSES=["background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair",
        "cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa",
        "train","tvmonitor"]

COLORS=np.random.uniform(0,255,size=(len(CLASSES),3))

print("INFO:loading model...")
net=cv2.dnn.readNetFromCaffe(args["prototxt"],args["model"])

vs=VideoStream(src=0).start()
time.sleep(3.0)
fps=FPS().start()

while True:
    frame=vs.read()
    frame=imutils.resize(frame,width=400)
    (h,w)=frame.shape[:2]
    blob=cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),0.007842,(300,300),127.5)
    net.setInput(blob)
    detections=net.forward()
    for i in np.arange(0,detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence > 0.2:
            idx=int(detections[0,0,i,1])
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype("int")
            label="JMTest "+"{}:{:.2f}%".format(CLASSES[idx],confidence*100)
            cv2.rectangle(frame,(startX,startY),(endX,endY),COLORS[idx],2)
            y=startY-15 if startY-15>15 else startY+15
            cv2.putText(frame,label,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,COLORS[idx],2)
        cv2.imshow("JingMing Test",frame)
        key=cv2.waitKey(1) & 0xFF
        if key==ord("q") or key==ord("Q"):
            break
        fps.update()
fps.stop()

print("INFO elapsed time {:.2f}".format(fps.elapsed()))
print("iNFO approx, FPS:{:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()

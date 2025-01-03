import time
import numpy as np
import cv2
import HandTrackModule_MediaPipe as htm
import math
from cvzone.HandTrackingModule import HandDetector

wcam, hcam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)
ptime = 0

detector = htm.handDetector(detectionCon=0.7)
stop = HandDetector(detectionCon=0.7)


from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volrange = volume.GetVolumeRange()
# volume.SetMasterVolumeLevel(-20.0, None)
minVol = volrange[0]
maxVol = volrange[1]
vol = 0
volbar = 350
while True:
    
    success, img = cap.read()
    img = detector.findhands(img)
    lmlist = detector.findposition(img, draw=False)
    if len(lmlist) != 0:
        # print(lmlist[16],lmlist[20])
        
        x1 , y1 = lmlist[4][1], lmlist[4][2]
        x2 , y2 = lmlist[8][1], lmlist[8][2]
        cx , cy = (x1 + x2) // 2 , (y1 + y2) // 2
        
        
        cv2.circle(img, (x1, y1), 10, (255,0,255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255,0,255), cv2.FILLED)
        cv2.line(img, (x1,y1),(x2,y2), (255,0,255), 2)
        
        cv2.circle(img, (cx, cy), 10, (255,0,255), cv2.FILLED)
        
        length = math.hypot(x2 - x1, y2 - y1)
        # print(length)
        
        vol = np.interp(length, [20,100], [minVol, maxVol])
        volbar = np.interp(length, [20,100], [350,150])
        print(int(length),vol)
        volume.SetMasterVolumeLevel(vol, None)
        
        if length<20:
            cv2.circle(img, (cx,cy), 15, (255,255,255), cv2.FILLED)
    
    cv2.rectangle(img, (50,150), (85,350), (0,255,0), 3)
    cv2.rectangle(img, (50, int(volbar)), (85,350), (0,255,0), cv2.FILLED)    
        
    ctime = time.time()
    fps = 1/ (ctime - ptime)
    ptime = ctime
    # print(int(fps))
    
    cv2.putText(img, f'FPS : {int(fps)}', (40,50) ,cv2.FONT_HERSHEY_COMPLEX, 1, (255 ,0 ,0), 2)
    cv2.imshow("Img",img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

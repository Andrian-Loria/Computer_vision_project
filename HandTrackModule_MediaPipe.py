import time
import cv2
import mediapipe as mp


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpdraw = mp.solutions.drawing_utils         

    def findhands(self, img, draw = True):
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgrgb)
        #    print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handl in self.results.multi_hand_landmarks:
                if draw:                
                    self.mpdraw.draw_landmarks(img, handl, self.mphands.HAND_CONNECTIONS)
        
        return img
   
    def findposition(self, img, handNo=0, draw= True):
      
        lmlist = []
        
        if self.results.multi_hand_landmarks:
            
            myhand = self.results.multi_hand_landmarks[handNo]
            
            for id, lm in enumerate(myhand.landmark):
                # print(id,lm)
                h, w, c=img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id,cx,cy)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 10, (255,0,255), cv2.FILLED)

        return lmlist

def main():
    pTime = 0
    cTime = 0
    
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    
    while True:
    
        success, img = cap.read()
        img = detector.findhands(img)
        lmlist = detector.findposition(img)
        if len(lmlist) != 0:
            print(lmlist[4])
        
        cTime = time.time()
        fps =  1 / (cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, 
                    (255,0,255),3)
                
        cv2.imshow('Image',img)
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()
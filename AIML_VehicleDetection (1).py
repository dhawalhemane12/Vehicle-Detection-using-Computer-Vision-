import cv2
cap = cv2.VideoCapture("carvideo.mp4")                                        #Input as Video
#object Detection
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=30)         #Background Subtraction
while True:
    success, frame = cap.read()
    height, width, _ = frame.shape

    #Height and Width of the Frame
    print(height, width)
    #Extracting the Region of Interest
    Region = frame[10:360, 10:550]                                                          #Region of Interest

    mask = object_detector.apply(Region)                                                   #Creating mask on the Region of Interest
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        #Calculate the area and remove the small elements
        area = cv2.contourArea(cnt)
        if area > 50:
            #cv2.drawContours(Region, [cnt], -1,(0,255,0), 2)
            x, y, w ,h =cv2.boundingRect(cnt)
            cv2.rectangle(Region, (x,y), (x+w, y+h), (0,255,0), 3)
            detections.append([x, y, w ,h])

    print(detections)
    cv2.imshow("Region", Region)                                               #Displaying the Detection on Region of Interest

    cv2.imshow("Mask", mask)
    key = cv2.waitKey(20)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
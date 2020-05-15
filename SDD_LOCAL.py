import numpy as np
import time
import cv2
import math

labelsPath = "./coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

weightsPath = "yolov3.weights"
configPath = "yolov3.cfg"
cap = cv2.VideoCapture('test_video.mp4')
hasFrame, frame = cap.read()
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
#vid_writer = cv2.VideoWriter('town_out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))
vid_writer = cv2.VideoWriter('salida.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (640,360))


while cv2.waitKey(1) < 0:
    
    ret,image=cap.read()
    if( not ret ):
        break
    image=cv2.resize(image,(640,360))
    (H, W) = image.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 1 / 300.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    print("Frame Prediction Time : {:.6f} seconds".format(end - start))
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.1 and classID == 0:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2 ))
                y = int(centerY - (height / 2 ))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
    #idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)
    ind = []
    for i in range(0,len(classIDs)):
        if(classIDs[i]==0):
            ind.append(i)
    a = []
    b = []
    idxs_list = idxs.flatten()
    if len(idxs) > 0:
        for i in idxs_list:
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            a.append(x)
            b.append(y)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255,255,255), 2)
    print(idxs_list,len(idxs))   
    print("a.len:%d"%len(a))            
    distance= []
    nsd = []
    for i in range(0,len(a)-1):
        for k in range(i+1,len(a)):
            if(k==i):
                continue
            else:
                x_dist = (a[k] - a[i])
                y_dist = (b[k] - b[i])
                d = math.sqrt(x_dist * x_dist + y_dist * y_dist)
                print("dist %d"%d)
                distance.append(d)
                if(d <=100):
                    #nsd no social distancing
                    nsd.append(i)
                    nsd.append(k)
                    cv2.line(image, (a[i],b[i]), (a[k],b[k]), (255,255,255), 1)
                else:
                    cv2.line(image, (a[i],b[i]), (a[k],b[k]), (255,0,0), 1)
                nsd = list(dict.fromkeys(nsd))
                print(nsd)

    print("len.dist %d "%len(distance))
    color = (0, 0, 255) 
    cc = 0
    for i in nsd:
        cc += 1
        j = idxs_list[i]
        (x, y) = (boxes[j][0], boxes[j][1])
        (w, h) = (boxes[j][2], boxes[j][3])
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "Cuidado!"
        cv2.putText(image, text, (x, y + 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 1)
    print(cc)

    color = (0, 255, 0) 
    cc = 0

    if len(idxs) > 0:
        for i in range(len(a)):
            if (i in nsd):
                continue
            else:
                cc += 1
                j = idxs_list[i]
                (x, y) = (boxes[j][0], boxes[j][1])
                (w, h) = (boxes[j][2], boxes[j][3])
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = 'OK'
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 1)   
    print(cc)
    cv2.imshow("Social Distancing Detector", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    vid_writer.write(image)

vid_writer.release()
cap.release()
cv2.destroyAllWindows()

import numpy as np
import time
import cv2
import math

#PROC_WIDTH = 960
#PROC_HEIGTH = 540
#PROC_WIDTH_EXP = 360

PROC_WIDTH = 640
PROC_HEIGTH = 360
PROC_WIDTH_EXP = 320

DEST_W = PROC_HEIGTH // 3
DEST_H = PROC_HEIGTH // 3

VIDEO_INPUT = "TownCentreXVID.avi"
VIDEO_OUTPUT = "TownCentre_out.mp4"

RE = (0,0,255)
GR = (0,255,0)
BL = (255,0,0)
BLK = (0,0,0)
WHT = (255,255,255)
YL = (0, 255, 255)

mouse_pts = []

def get_mouse_points(event, x, y, flags, param):
    # usado para marcar 4 puntos que representan un rectangulo
    # mas dos puntos que están a 1.80mts de distancia
    global mouseX, mouseY, mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        if "mouse_pts" not in globals():
            mouse_pts = []
        if len(mouse_pts)>=4:
            cv2.circle(image, (x, y), 7, BL, 3)
        else:
            cv2.circle(image, (x, y), 7, YL, 3)
        mouse_pts.append((x, y))
        if len(mouse_pts)==4:            
            cv2.putText(image, txt2 , (5, 35), cv2.FONT_HERSHEY_SIMPLEX,0.5, BLK, 3) 
            cv2.putText(image, txt2 , (5, 35), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,144,155), 1)  
        print("Point detected")
        print(mouse_pts)

def get_camera_perspective(dsize, src_points):
    #IMAGE_H = 100
    #IMAGE_W = 100
    DX,DY = 100,250
    IMAGE_H,IMAGE_W=dsize
    src = np.float32(np.array(src_points))
    dst = np.float32([[DX, DY+IMAGE_H], [DX+IMAGE_W,DY+ IMAGE_H], [DX, DY], [DX+IMAGE_W, DY]])

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    return M, M_inv

labelsPath = "./coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

weightsPath = "yolov3.weights"
configPath = "yolov3.cfg"
cap = cv2.VideoCapture(VIDEO_INPUT)
hasFrame, frame = cap.read()
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
#vid_writer = cv2.VideoWriter('town_out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))
vid_writer = cv2.VideoWriter(VIDEO_OUTPUT,cv2.VideoWriter_fourcc('M','J','P','G'), 25, (PROC_WIDTH+PROC_WIDTH_EXP,PROC_HEIGTH))

cv2.namedWindow("image")
cv2.setMouseCallback("image", get_mouse_points)
num_mouse_points = 0
image=cv2.resize(frame,(PROC_WIDTH,PROC_HEIGTH))
txt1 = "ESCOGE 4 PUNTOS QUE REPRESENTEN UN CUADRADO EN EL SUELO"
txt2 = "ESCOGE 2 PUNTOS QUE REPRESENTEN UNA SEPARACION DE 2 mts"
cv2.putText(image, txt1, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,0.5, BLK, 3)
cv2.putText(image, txt1 , (5, 15), cv2.FONT_HERSHEY_SIMPLEX,0.5, YL, 1) 
 
while True:
    #image = frame
    cv2.imshow("image", image)
    cv2.waitKey(1)
    if len(mouse_pts) == 7:
        cv2.destroyWindow("image")
        break
    first_frame_display = False
four_points = mouse_pts

M, Minv = get_camera_perspective((DEST_W,DEST_H), four_points[0:4])

#calcular distancia minima en el espacio transformado
p4arr = np.array(four_points,np.float)
warp_dist= cv2.perspectiveTransform(np.array([p4arr[4:6,:]]),M)
dist_x = warp_dist[0,1,0]-warp_dist[0,0,0]
dist_y = warp_dist[0,1,1]-warp_dist[0,0,1]

DIST_MIN = np.sqrt(dist_x*dist_x+dist_y*dist_y)

image_exp = cv2.warpPerspective(image, M, (PROC_WIDTH_EXP,PROC_HEIGTH) ) 
print(image_exp.shape)
#CREAR ARRAY EXPANSION
#image_exp = np.zeros((PROC_HEIGTH,PROC_WIDTH_EXP,3), dtype=np.uint8)

while cv2.waitKey(1) < 0:
    
    ret,image=cap.read()
    if( not ret ):
        break
    image=cv2.resize(image,(PROC_WIDTH,PROC_HEIGTH))
    
    image_exp = cv2.warpPerspective(image, M, (PROC_WIDTH_EXP,PROC_HEIGTH) )
    cv2.putText(image_exp, "Vista aerea" , (5, 15), cv2.FONT_HERSHEY_SIMPLEX,0.5, BLK, 3) 
    cv2.putText(image_exp, "Vista aerea" , (5, 15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,144,155), 1) 

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
                boxes.append([x, y, int(width), int(height), centerX, centerY])
                confidences.append(float(confidence))
                classIDs.append(classID)

    #crear un array con las coordenadas de cada persona
    bmat = np.array(boxes,np.float)
    center_mat = np.array([bmat[:,4:6]])
    warp_center = np.array(cv2.perspectiveTransform(center_mat,M) , dtype=np.int)
#    for a in range(1,len(boxes)):
#        cv2.circle(image_exp, (warp_center[0,a,0],warp_center[0,a,1]), 3, (255,255,255), 2)    


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
            (wx, wy) = (warp_center[0,i,0], warp_center[0,i,1])
            a.append(wx)
            b.append(wy)
            #cv2.rectangle(image, (x, y), (x + w, y + h), (255,255,255), 2)
            cv2.circle(image, (boxes[i][4],boxes[i][5]), 4, (255,255,255), 2)
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
                if(d <= DIST_MIN):
                    #nsd no social distancing
                    nsd.append(i)
                    nsd.append(k)
                    cv2.line(image_exp, (a[i],b[i]), (a[k],b[k]), (0,0,255), 1)
                    cv2.circle(image_exp, (a[i],b[i]), DIST_MIN//2, (0,0,255), 1)                    
                    cv2.circle(image_exp,  (a[k],b[k]),  DIST_MIN//2, (0,0,255), 1)
                #else:
                    #cv2.line(image, (a[i],b[i]), (a[k],b[k]), (200,0,0), 1)
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

        #marcarlos en la imagen warpeada
        cv2.circle(image_exp, (warp_center[0,j,0],warp_center[0,j,1]), 3, (0,0,255), 2)
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

                #marcarlos en la imagen warpeada
                cv2.circle(image_exp, (warp_center[0,j,0],warp_center[0,j,1]), 3, (0,255,0), 2)   
    print(cc)
    #expande la imagen
    
    


    image_out = np.concatenate((image,image_exp), axis = 1)
    print(image_out.shape)
    print(image.dtype)
    print(image_exp.dtype)
    print(image_out.dtype)
    cv2.imshow("Distanciamiento Social", image_out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    vid_writer.write(image_out)

vid_writer.release()
cap.release()
cv2.destroyAllWindows()

# proyecto de vision computacional

import numpy as np
import time
import cv2
import math


lines_flag = False

#definimos el tamaño de salida del video
#el video original es de 1920x1080 por lo que es necesario un resize
PROC_WIDTH = 860
PROC_HEIGTH = 480

#el ancho de la pantalla extra para el bird-eye
PROC_WIDTH_EXP = 440

#dimensiones del cuadrado destino
DEST_W = 120
DEST_H = 120

#video de prueba , el cual es una version reducida del original TownCentreXVID.avi
VIDEO_INPUT = "test_video.mp4"
VIDEO_OUTPUT = "test_video_out2.avi"

#definimos constantes que describen los colores en el BGR de openCV
RE = (0,0,255)
GR = (0,255,0)
BL = (255,0,0)
BLK = (0,0,0)
WHT = (255,255,255)
YL = (0, 255, 255)

#offset de transformación
DX,DY = 150,300

#esta funcion se lo asignamos a un callback asociado a la primera ventana
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
        print(mouse_pts)

#AQUI CALCULAMOS LA MATRIX  M  DE LA TRANSFORMACION DE PERSPECTIVA
#PARA ELLO USAMOS LOS CUATRO PUNTOS QUE ESCOGIO EL USUARIO COMO source
#Y LUEGO DEFINIMOS UN CUADRADO EN LA IMAGEN DESTINO COMO destination
def get_camera_perspective(dsize, src_points):

    IMAGE_H,IMAGE_W=dsize
    src = np.float32(np.array(src_points))
    dst = np.float32([[DX, DY+IMAGE_H], [DX+IMAGE_W,DY+ IMAGE_H], [DX, DY], [DX+IMAGE_W, DY]])

    M = cv2.getPerspectiveTransform(src, dst)
    return M

#constantes que apuntan a los archivos que definen la "darknet" implementada en openCV
labelsPath = "./coco.names"
weightsPath = "yolov3.weights"
configPath = "yolov3.cfg"
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#leemos el primer frame del video para que el usuario pueda
#escoger los cuatro puntos que definin la transformacion de perspectiva
cap = cv2.VideoCapture(VIDEO_INPUT)
hasFrame, frame = cap.read()

#vid_writer escribe frame por frame el video de salida,
#las dimensiones del video de salida las definimos en las constantes del comienzo
vid_writer = cv2.VideoWriter(VIDEO_OUTPUT,cv2.VideoWriter_fourcc(*'XVID'), 25, (PROC_WIDTH+PROC_WIDTH_EXP,PROC_HEIGTH))


cv2.namedWindow("image")
cv2.setMouseCallback("image", get_mouse_points)
num_mouse_points = 0
image=cv2.resize(frame,(PROC_WIDTH,PROC_HEIGTH))
txt1 = "ESCOGE 4 PUNTOS QUE REPRESENTEN UN CUADRADO EN EL SUELO"
txt2 = "ESCOGE 2 PUNTOS QUE REPRESENTEN UNA SEPARACION DE 2 mts"
cv2.putText(image, txt1, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,0.5, BLK, 3)
cv2.putText(image, txt1 , (5, 15), cv2.FONT_HERSHEY_SIMPLEX,0.5, YL, 1) 

#esperamos a tener los 6 puntos mas un 7mo de confirmacion 
while True:
    #image = frame
    cv2.imshow("image", image)
    cv2.waitKey(1)
    if len(mouse_pts) == 7:
        cv2.destroyWindow("image")
        break
    first_frame_display = False
four_points = mouse_pts

#calcular la matrix M
M = get_camera_perspective((DEST_W,DEST_H), four_points[0:4])

#calcular distancia minima en el espacio transformado
#la distancia entre el 5to y 6to punto representan 2mts en la imagen original
#se transforma esa distancia y se obtiene la DIST_MINIMA que se usa para verificar violaciones
#del distanciamiento social
p4arr = np.array(four_points,np.float)
warp_dist= cv2.perspectiveTransform(np.array([p4arr[4:6,:]]),M)
dist_x = warp_dist[0,1,0]-warp_dist[0,0,0]
dist_y = warp_dist[0,1,1]-warp_dist[0,0,1]
DIST_MIN = int(np.sqrt(dist_x*dist_x+dist_y*dist_y))

#warpPerspective de la imagen original
image_exp = cv2.warpPerspective(image, M, (PROC_WIDTH_EXP,PROC_HEIGTH) ) 

#bucle que se repite hasta que se lea el ultimo frame
#o se presione "q"
while cv2.waitKey(1) < 0:   

    #leer un frame
    ret,image=cap.read()
    if( not ret ):
        break

    image=cv2.resize(image,(PROC_WIDTH,PROC_HEIGTH))

    #dibujamos un cuadrado amarillo en la imagen original
    pts_ord = [0,1,3,2,0]
    for i in range(0,4):
        j =pts_ord[i]
        k =pts_ord[i+1]
        cv2.circle(image, four_points[i],6, YL, 3)
        cv2.line(image,four_points[j],four_points[k],YL,1 )
        print(i)
    
    image_exp = cv2.warpPerspective(image, M, (PROC_WIDTH_EXP,PROC_HEIGTH) )
    cv2.putText(image_exp, "VISTA AEREA" , (5, 15), cv2.FONT_HERSHEY_SIMPLEX,0.5, BLK, 3) 
    cv2.putText(image_exp, "VISTA AEREA" , (5, 15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,144,155), 1) 

    #dibujamos un cuadrado amarillo en la imagen destino
    cv2.circle(image_exp, (DX, DY+DEST_H), 3, YL, 3)
    cv2.circle(image_exp, (DX+DEST_W,DY+ DEST_H), 3, YL, 3)
    cv2.circle(image_exp, (DX, DY), 3, YL, 3)
    cv2.circle(image_exp, (DX+DEST_W, DY), 3, YL, 3)
    cv2.line(image_exp,(DX, DY+DEST_H),(DX+DEST_W,DY+ DEST_H),YL,1 )
    cv2.line(image_exp,(DX+DEST_W,DY+ DEST_H),(DX+DEST_W, DY),YL,1 )
    cv2.line(image_exp,(DX+DEST_W, DY),(DX, DY),YL,1 )
    cv2.line(image_exp,(DX, DY),(DX, DY+DEST_H),YL,1 )

    #adaptamos el frame leido para alimentar a YOLOv3
    (H, W) = image.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 1 / 300.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()

    #aqui obtenemos las detecciones en las 3 capas de salida
    layerOutputs = net.forward(ln)
    end = time.time()
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


    #los bounding boxes pueden estar repetidos porque hay 3 capas de salida trabajando a diferente escala
    #para eso se escoge los mejores candidatos con la funcion
    #Non Maximum Suppression (NMSBoxes())
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)
    ind = []
    for i in range(0,len(classIDs)):
        if(classIDs[i]==0):
            ind.append(i)

    #las listas a y b las llenamos con las coordenadas de los centroides
    #transformados (warp_center[])
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
            cv2.circle(image, (boxes[i][4],boxes[i][5]), 4, (255,255,255), 2)

    #nsd tiene la lista de indices de personas que no respetan el distanciamiento
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
                distance.append(d)
                if(d <= DIST_MIN):
                    #nsd no social distancing
                    nsd.append(i)
                    nsd.append(k)
                    if lines_flag:
                        cv2.line(image_exp, (a[i],b[i]), (a[k],b[k]), RE, 3)
                    else:
                        cv2.line(image_exp, (a[i],b[i]), (a[k],b[k]), RE, 1)
                        cv2.circle(image_exp, (a[i],b[i]), DIST_MIN//2, RE, 1)                    
                        cv2.circle(image_exp,  (a[k],b[k]),  DIST_MIN//2, RE, 1)
                else:
                    if lines_flag:
                        cv2.line(image_exp, (a[i],b[i]), (a[k],b[k]), WHT, 1)
                nsd = list(dict.fromkeys(nsd))
                print(nsd)


    for i in nsd:
        j = idxs_list[i]
        (x, y) = (boxes[j][0], boxes[j][1])
        (w, h) = (boxes[j][2], boxes[j][3])
        cv2.rectangle(image, (x, y), (x + w, y + h), RE, 2)
        text = "Cuidado!"
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, RE, 1)

        #marcarlos en la imagen warpeada
        cv2.circle(image_exp, (warp_center[0,j,0],warp_center[0,j,1]), 3, RE, 2)

    if len(idxs) > 0:
        for i in range(len(a)):
            if (i in nsd):
                continue
            else:
                j = idxs_list[i]
                (x, y) = (boxes[j][0], boxes[j][1])
                (w, h) = (boxes[j][2], boxes[j][3])
                cv2.rectangle(image, (x, y), (x + w, y + h), GR, 2)
                text = 'OK'
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, GR, 1)

                #marcarlos en la imagen warpeada
                cv2.circle(image_exp, (warp_center[0,j,0],warp_center[0,j,1]), 3, GR, 2)   
      
    #concatenamos la imagen original y transformada en una sola imagen
    image_out = np.concatenate((image,image_exp), axis = 1)

    cv2.imshow("Distanciamiento Social", image_out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('l'):
        lines_flag = True
    if cv2.waitKey(1) & 0xFF == ord('k'):
        lines_flag = False   
    vid_writer.write(image_out)

vid_writer.release()
cap.release()
cv2.destroyAllWindows()

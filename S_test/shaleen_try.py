import cv2
import time
import math
import numpy as np
import pandas as pd
#from car import cars


#thres = 0.45 # Threshold to detect object

classNames = []
count = 0
center_points_prev_frame = []

center_points_cur_frame = []

tracking_objects = {}


classFile = "/home/omicron/Videos/S_test/coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/omicron/Videos/S_test/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/omicron/Videos/S_test/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)

    if len(objects) == 0: objects = classNames
    objectInfo =[]

    
    track_id = 0
    if len(classIds) != 0:

        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box,className])

                if (draw):
                    
                    (x, y, w, h) = box
                    cx = int((x+x+w)/2)
                    cy = int((y+y+h)/2)
                    center_points_cur_frame.append((cx, cy))
                    cv2.circle(img, (cx, cy), 5, (0,0,255), -1)
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    
                    fps= cap.get(cv2.CAP_PROP_FPS)
                    cv2.putText(img, str(fps),(5,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                    text = "(" + str(cx) + ", " + str(cy) + ")"
                    cv2.putText(img,text,(cx+10,cy+30),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),2)
                                        
                    #cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    #cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    
                    
                #    center_points_cur_frame = []

                    
            # Only at the beginning we compare previous and current frame
                if count <= 2:
                    #print("hello")
                    for pt in center_points_cur_frame:
                        for pt2 in center_points_prev_frame:
                            distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                            

                            if distance < 10:
                                tracking_objects[track_id] = pt                           
                                track_id += 1
                                print(track_id)
                else:

                    tracking_objects_copy = tracking_objects.copy()
                    center_points_cur_frame_copy = center_points_cur_frame.copy()
                    

                    for object_id, pt2 in tracking_objects_copy.items():
                        object_exists = False
                        for pt in center_points_cur_frame_copy:
                            distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                            # Update IDs position
                            if distance < 10:
                                tracking_objects[object_id] = pt
                                object_exists = True
                                if pt in center_points_cur_frame:
                                    center_points_cur_frame.remove(pt)
                                continue

                        # Remove IDs lost
                        if not object_exists:
                            tracking_objects.pop(object_id)

                    # Add new IDs found
                    for pt in center_points_cur_frame:
                        tracking_objects[track_id] = pt
                        
                        track_id += 1      

                for object_id, pt in tracking_objects.items():
                    cv2.circle(img, pt, 5, (0, 0, 255), -1)
                    cv2.putText(img, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)  
                print("Tracking objects")
                print(tracking_objects)
                distancer(tracking_objects)


                print("CUR FRAME LEFT PTS")
                print(center_points_cur_frame)
    return img,objectInfo

def distancer(trackdic):
    cor_arr = list(trackdic.values())
    df = pd.DataFrame(cor_arr, columns= ['x', 'y'])

    df['dis'] = df[['x','y']].apply(lambda row : np.linalg.norm((row.x, row.y)), axis=1)
    df.sort_values('dis', ignore_index=True, inplace= True)
    #print(df[['x','y']].values[:1])
    #print(list(df.loc[0]))
    #print(df.iloc[:,0].to_list())
    coordd = df[['x','y']].values[:1]
    #print(coordd)
    x, y = cor_arr[0]
    print(x)
    print(y)
    cmd = direction(x,y)
    print(cmd)
    #car(cmd)
    return


def direction(x1, y1):
    y2 = 415
    x2 = 310
    angle = math.atan2(y2 - y1, x2 - x1)
    angle_degrees = math.degrees(angle)
    print(angle_degrees)
    if 75 <= angle_degrees <= 95:
        return "straight"
    elif 15 < angle_degrees <= 74:
        return "left"
    elif 96 <= angle_degrees < 155:
        return "right"
    #35 85 145



if __name__ == "__main__":
        	
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    cv2.waitKey(100)
    #cap.set(cv2.CAP_PROP_FPS, 5)


    while True:
        success, img = cap.read()
        # Initialize count

        
        count += 1
        if not success:
            break
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        #clahe_output = clahe.apply(gray)
        result, objectInfo = getObjects(img,0.5,0.2,objects=["bottle","cup"])
        print(objectInfo)
        cv2.imshow("Output",img)

        if cv2.waitKey(100) & 0xFF == ord('q'):
        	break
        
        #cv2.waitKey(1)

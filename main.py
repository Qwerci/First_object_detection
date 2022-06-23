import cv2
from gui_buttons import Buttons
from numpy import size


#initialize buttons
button = Buttons()
button.add_button('person',20,20)
button.add_button("cell phone", 20, 100)
button.add_button("keyboard", 20, 180)
button.add_button("remote", 20, 260)
button.add_button("scissors", 20, 340)

colors = button.colors


#Opencv DNN
net = cv2.dnn.readNet( 'dnn_model/yolov4-tiny.weights',"dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale = 1/255)


#load class lists
classes = []
with open("dnn_model/classes.txt", 'r') as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)



#Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)

def click_button(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        button.button_click(x,y)

#Create Window
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', click_button)

while True:
    #get frames and resize it
    ret, frame = cap.read()
    
    # Active button List
    active_buttons = button.active_buttons_list()
    print("Active buttons", active_buttons)
    #object Detection 
    class_ids, scores, bboxes = model.detect(frame,confThreshold =0.3, nmsThreshold=.4)

    for class_id,scores,bboxes in zip(class_ids, scores,bboxes):
        x,y,w,h = bboxes
        class_name = classes[class_id]
        color = colors[class_id]
        if class_name in active_buttons:
            cv2.putText(frame,class_name, (x,y-5), cv2.FONT_HERSHEY_DUPLEX,2, (150,0,60), 2)
            cv2.rectangle(frame, (x,y),(x+w, y+h),(200,0,60),3)

    #Display Buttons
    button.display_buttons(frame)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        cap.release()
cv2.destroyAllWindows()
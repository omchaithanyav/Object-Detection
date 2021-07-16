import os
import speech_recognition as sr 
import playsound 
from gtts import gTTS 
import cv2


num = 1
def voice(output):
    global num
  
    num += 1
    print("VoiceBot: ", output)
  
    toSpeak = gTTS(text = output, lang ='en', slow = False)
    
    file = str(num)+".mp3" 
    toSpeak.save(file)
      
    playsound.playsound(file, True) 
    os.remove(file)
    
    
    
def getting_audio():
  
    rObject = sr.Recognizer()
    audio = ''
  
    with sr.Microphone() as source:
        print("Say something")
          
        audio = rObject.listen(source, phrase_time_limit = 3) 
  
    try:
        text = rObject.recognize_google(audio, language ='en-US')
        print("user: ", text)
        return text
  
    except:
  
        voice("Sorry, I Could not understand ,PLease do try again!")
        return getting_audio()


def detect():
    
    Conf_threshold = 0.7
    NMS_threshold = 0.4
    COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    class_name = []
    with open('coco.names', 'r') as f:
        class_name = [cname.strip() for cname in f.readlines()]

    net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg') 
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size = (416,416), scale = 1/255, swapRB = True)
    
    cap = cv2.VideoCapture('video.mp4')
    
    while(True):
        ret, frame = cap.read()
        
        if ret == False:
            break
        
        classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
        for (classid, score, box) in zip(classes, scores, boxes):
            color = COLORS[int(classid) % len(COLORS)] 
            label = "%s : %f" % (class_name[classid[0]], score)
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color, 1)
            
        cv2.imshow('frame',frame) 
        
        key = cv2.waitKey(1)
        if key == ord('s'):
            text = getting_audio()
            if 'stop' in str(text):
                break
            
        
    cap.release()
    cv2.destroyAllWindows() 
    
    


class speak():
    
    if __name__ == "__main__":
        
        
        voice("Hello")
        

        while(True):
          
                voice("just say, 'detect' to start object detection")
                text = getting_audio()
                
                if text == 0:
                        break
                    
                elif "detect" in str(text):
                        detect()
                    
      
                elif "exit" in str(text) or "bye" in str(text) or "stop" in str(text) or "terminate" in str(text):
                        voice("bye buddy, "+ "have a great time"+'.')
                        break
                    
                else:
                    voice('Sorry cant help you with that.')
                    break   
                    
               
              
        
            
               
       
    
    



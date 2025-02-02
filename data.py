import cv2
from mss import mss
import numpy as np
import os
import time
import keyboard
import pyautogui
import torch
from torchvision import transforms
from model import create_model
from datetime import datetime
from pathlib import Path
import shutil

class AutoStateDetector:
   def __init__(self):
       self.cap = mss()
       self.region = {'left': 0, 'top': 0, 'width': 1280, 'height': 800}
       self.states = ['menu', 'gameplay', 'loading']
       
       self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       print(f"Using device: {self.device}")
       self.model = self.load_model('l4d2_model.pth')
       
       self.transform = transforms.Compose([
           transforms.ToPILImage(),
           transforms.Resize((224, 224)),
           transforms.ToTensor(),
           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       ])

       self.save_dir = "training_data"
       self.uncertain_dir = os.path.join(self.save_dir, "uncertain")
       os.makedirs(self.uncertain_dir, exist_ok=True)

   def load_model(self, model_path):
       model = create_model(num_classes=3)
       model.load_state_dict(torch.load(model_path, map_location=self.device))
       model.to(self.device)
       model.eval()
       return model

   def find_window_position(self):
       print("move your cursor to the top left corner of l4d2 and press t")
       while not keyboard.is_pressed('t'):
           time.sleep(0.1)
       x1, y1 = pyautogui.position()
       time.sleep(0.5)
       
       print("move to the bottom right corner and press t")
       while not keyboard.is_pressed('t'):
           time.sleep(0.1)
       x2, y2 = pyautogui.position()
       
       self.region = {
           'left': x1,
           'top': y1,
           'width': x2 - x1,
           'height': y2 - y1
       }
       print(f"capture region set to: {self.region}")

   def predict_state(self, frame):
       frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       frame = self.transform(frame).unsqueeze(0).to(self.device)
       
       with torch.no_grad():
           outputs = self.model(frame)
           probabilities = torch.nn.functional.softmax(outputs, dim=1)
           confidence, predicted = torch.max(probabilities, 1)
       
       return self.states[predicted.item()], confidence.item()

   def save_uncertain_frame(self, frame, state, confidence):
       timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
       filename = f"uncertain_{state}_{confidence*100:.2f}_{timestamp}.png"
       filepath = os.path.join(self.uncertain_dir, filename)
       cv2.imwrite(filepath, frame)
       print(f"saved uncertain frame: {filename}")

   def run(self):
       self.find_window_position()
       last_state = None
       print("\nstarted state detection")
       print("press q to quit\n")
       print("uncertain predictions (<= 95% confidence) will be saved to 'training_data/uncertain/'")
       
       while True:
           screenshot = self.cap.grab(self.region)
           frame = np.array(screenshot)
           frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
           
           state, confidence = self.predict_state(frame)
           
           if confidence <= 0.95:
               self.save_uncertain_frame(frame, state, confidence)
           
           if state != last_state:
               print(f"detected state: {state} (confidence: {confidence*100:.2f}%)")
               last_state = state
           
           preview = frame.copy()
           color = (0, 0, 255) if confidence <= 0.70 else (0, 255, 0)
           cv2.putText(preview, f"state: {state}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
           cv2.putText(preview, f"confidence: {confidence*100:.2f}%", (10, 70),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
           
           cv2.imshow('L4D2 Data', preview)
           cv2.waitKey(1)
           
           if keyboard.is_pressed('q'):
               break
       
       cv2.destroyAllWindows()
       print("\nstopped state detection")

if __name__ == "__main__":
   print("\nL4D2 Model Data Getter | Made by Rastastick")
   print("============================")
   print("use t to set l4d2 window (recommended to change resolution to 1280x720 for 1080p or 1920x1080 for 144op)")
   print("l4d2 model will identify the game state")
   print("uncertain predictions (<= 95% confidence) will be saved at 'training_data/uncertain/' for manual sorting")
   print("press q to quit")
   
   detector = AutoStateDetector()
   detector.run()
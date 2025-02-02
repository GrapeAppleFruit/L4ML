import torch
import cv2
import numpy as np
from torchvision import transforms
from model import create_model 
import mss
import keyboard
import time
import pyautogui

def find_window_position():
   print("move your cursor to the top left corner of l4d2 and press t")
   while not keyboard.is_pressed('t'):
       time.sleep(0.1)
   x1, y1 = pyautogui.position()
   time.sleep(0.5)
   
   print("then move your cursor to the bottom right corner and press t") 
   while not keyboard.is_pressed('t'):
       time.sleep(0.1)
   x2, y2 = pyautogui.position()
   
   region = {
       'left': x1,
       'top': y1,
       'width': x2 - x1,
       'height': y2 - y1
   }
   print(f"capture region set to: {region}")
   return region

def load_model(model_path, device='cuda'):
   model = create_model(num_classes=3)
   model.load_state_dict(torch.load(model_path, map_location=device))
   model.to(device)
   model.eval()
   return model

def predict_state(model, frame, device='cuda'):
   transform = transforms.Compose([
       transforms.ToPILImage(),
       transforms.Resize((224, 224)),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   ])
   
   frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   frame = transform(frame).unsqueeze(0).to(device)
   
   with torch.no_grad():
       outputs = model(frame)
       probabilities = torch.nn.functional.softmax(outputs, dim=1)
       confidence, predicted = torch.max(probabilities, 1)
   
   states = ['menu', 'gameplay', 'loading']
   return states[predicted.item()], confidence.item()

def main():
   sct = mss.mss()
   region = find_window_position()
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   print(f"Using device: {device}")
   
   try:
       model = load_model('l4d2_model.pth', device)
       print("model loaded")
   except Exception as e:
       print(f"error loading: {e}")
       return
   
   print("\nstate Detection")
   print("press q to quit")
   
   last_state = None
   while True:
       screenshot = sct.grab(region)
       frame = np.array(screenshot)
       frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
       
       state, confidence = predict_state(model, frame, device)
       
       if state != last_state:
           print(f"detected State: {state} (confidence: {confidence*100:.2f}%)")
           last_state = state
       
       preview = frame.copy()
       cv2.putText(preview, f"state: {state}", (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
       cv2.putText(preview, f"confidence: {confidence*100:.2f}%", (10, 70),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
       
       cv2.imshow('state Detection', preview)
       cv2.waitKey(1)
       
       if keyboard.is_pressed('q'):
           break
   
   cv2.destroyAllWindows()

if __name__ == '__main__':
   main()
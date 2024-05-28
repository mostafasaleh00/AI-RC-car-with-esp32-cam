
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import websocket



model = YOLO('yolov8s.pt') 
camera_url = "ws://192.168.4.1/Camera"  
ws_camera = None

def on_message(ws, message):
    np_arr = np.frombuffer(message, dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is not None:
    
        results = model(image)

        annotated_image = results[0].plot()

        cv2.imshow('ESP32 Camera with YOLOv8 Object Detection', annotated_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            ws.close()

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")

def on_open(ws):
    print("WebSocket connection opened")

def run_websocket():
    global ws_camera
    ws_camera = websocket.WebSocketApp(camera_url,
                                       on_message=on_message,
                                       on_error=on_error,
                                       on_close=on_close)
    ws_camera.on_open = on_open
    ws_camera.run_forever()

if __name__ == "__main__":
    websocket_thread = threading.Thread(target=run_websocket)
    websocket_thread.start()
    websocket_thread.join()
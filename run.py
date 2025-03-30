import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import time
from fastapi import FastAPI, Response, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse,JSONResponse,FileResponse
import asyncio
import uvicorn
import threading
import queue
app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500", 
        "https://your-frontend-domain.com",
        "*"  # Chỉ dùng trong dev
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=[
        "Content-Type", 
        "X-Requested-With", 
        "Authorization", 
        "X-CSRF-Token",
        "Accept"
    ],
)

AUDIO_DIR = "posture_data/audio_alerts"
POSTURE_AUDIO_MAPPING = {
    "hunched_back": "hunched_back.mp3",
    "leaning_forward": "leaning_forward.mp3", 
    "leaning_backward": "leaning_backward.mp3",
    "slouching": "slouching.mp3",
    "crossed_legs": "crossed_legs.mp3",
    "vai_nho": "vai_nho.mp3",
    "nghieng_sang_trai": "nghieng_sang_trai.mp3", 
    "nghieng_sang_phai": "nghieng_sang_phai.mp3"
}

# Biến theo dõi trạng thái tư thế
bad_posture_tracking = {
    "current_bad_posture": None,
    "start_time": None
}

posture_names_vi = {
    "straight_back": "Lưng thẳng",
    "hunched_back": "Lưng gù quá cong, cổ gập xuống",
    "leaning_forward": "Nghiêng trước",
    "leaning_backward": "Nghiêng sau",
    "slouching": "Lưng bị cong",
    "crossed_legs": "Bắt chéo chân",
    "vai_nho":"Vai bị lệch",
    "vai_thang": "Vai thẳng",
    "nghieng_sang_trai": "Nghiêng người sang trái",
    "nghieng_sang_phai": "Nghiêng người sang phải",
}

def translate_posture_to_vi(posture_class):
    """
    Chuyển đổi tên posture class sang tiếng Việt
    
    Args:
        posture_class (str): Tên class gốc
    
    Returns:
        str: Tên tiếng Việt tương ứng
    """
    return posture_names_vi.get(posture_class, "Chưa xác định")
# Khởi tạo MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Đường dẫn đến thư mục dữ liệu và mô hình
data_dir = "posture_data"
models_dir = os.path.join(data_dir, "models")

# Biến để theo dõi trạng thái camera
camera_active = True  # Mặc định camera ở trạng thái ON
last_prediction = {"class": None, "confidence": 0.0}
current_frame = None
frame_lock = threading.Lock()
frame_queue = queue.Queue(maxsize=1)
# Hàm tính góc giữa ba điểm
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    angle_deg = np.degrees(angle)
    return angle_deg

# Hàm trích xuất đặc trưng từ landmarks
def extract_features_from_landmarks(landmarks):
    features = []
    for landmark in landmarks:
        features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    
    try:
        left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                   landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
        right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
        
        left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])
        right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
        
        left_ear = np.array([landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y])
        right_ear = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y])
        
        left_knee = np.array([landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y])
        right_knee = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y])
        
        left_ankle = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y])
        right_ankle = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y])
        
        mid_shoulder = (left_shoulder + right_shoulder) / 2
        mid_hip = (left_hip + right_hip) / 2
        mid_ear = (left_ear + right_ear) / 2
        
        vertical_ref = np.array([mid_shoulder[0], 0])
        
        back_angle = calculate_angle(vertical_ref, mid_shoulder, mid_hip)
        neck_angle = calculate_angle(mid_ear, mid_shoulder, vertical_ref)
        left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
        
        knee_distance = np.linalg.norm(left_knee - right_knee)
        
        additional_features = [back_angle, neck_angle, left_leg_angle, right_leg_angle, knee_distance]
        features.extend(additional_features)
    except Exception as e:
        print(f"Lỗi khi trích xuất đặc trưng: {e}")
        features.extend([0, 0, 0, 0, 0])
    
    return features

# Hàm tải mô hình
def load_models():
    models = {}
    
    # Tải Random Forest
    rf_path = os.path.join(models_dir, 'random_forest_model.pkl')
    if os.path.exists(rf_path):
        with open(rf_path, 'rb') as f:
            models['rf'] = pickle.load(f)
    
    # Tải Gradient Boosting
    gb_path = os.path.join(models_dir, 'gradient_boosting_model.pkl')
    if os.path.exists(gb_path):
        with open(gb_path, 'rb') as f:
            models['gb'] = pickle.load(f)
    
    # Tải SVM
    svm_path = os.path.join(models_dir, 'svm_model.pkl')
    if os.path.exists(svm_path):
        with open(svm_path, 'rb') as f:
            models['svm'] = pickle.load(f)
    
    # Tải Neural Network
    nn_path = os.path.join(models_dir, 'neural_network_model.keras')
    if os.path.exists(nn_path):
        models['nn'] = tf.keras.models.load_model(nn_path)
    
    # Tải scaler
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            models['scaler'] = pickle.load(f)
    else:
        print("Cảnh báo: Không tìm thấy scaler.")

    # Tải label encoder
    label_encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
    if os.path.exists(label_encoder_path):
        with open(label_encoder_path, 'rb') as f:
            models['label_encoder'] = pickle.load(f)
    tf.config.run_functions_eagerly(False)
    return models

# Khởi tạo biến models
models = load_models()

def predict_posture(features, models):
    """Dự đoán tư thế sử dụng mô hình tổng hợp"""
    with tf.device('/CPU:0'):
        if 'scaler' not in models or 'label_encoder' not in models:
            return "Model components missing", 0.0
        
        try:
            features_scaled = models['scaler'].transform([features])
        except KeyError as e:
            print(f"Lỗi thiếu thành phần mô hình: {e}")
            return "Model error", 0.0
        # Dự đoán với từng mô hình
        predictions = {}
        
        if 'rf' in models:
            predictions['rf'] = models['rf'].predict_proba(features_scaled)
        
        if 'gb' in models:
            predictions['gb'] = models['gb'].predict_proba(features_scaled)
        
        if 'svm' in models:
            predictions['svm'] = models['svm'].predict_proba(features_scaled)
        
        if 'nn' in models:
            predictions['nn'] = models['nn'].predict(features_scaled)
        
        # Kết hợp các dự đoán
        if len(predictions) > 0:
            # Tính trung bình có trọng số các dự đoán
            ensemble_pred = np.zeros_like(list(predictions.values())[0])
            
            # Nếu có thông tin về ensemble
            if 'ensemble_meta' in models:
                weights = dict(zip(models['ensemble_meta']['models'], models['ensemble_meta']['weights']))
                
                for model_name, pred in predictions.items():
                    if model_name in weights:
                        ensemble_pred += pred * weights[model_name]
            else:
                # Nếu không có thông tin về ensemble, tính trung bình đơn giản
                for pred in predictions.values():
                    ensemble_pred += pred
                ensemble_pred /= len(predictions)
            
            # Lấy lớp có xác suất cao nhất
            predicted_class_idx = np.argmax(ensemble_pred)
            confidence = ensemble_pred[0][predicted_class_idx]
            
            # Chuyển đổi chỉ số lớp thành tên lớp
            predicted_class = models['label_encoder'].inverse_transform([predicted_class_idx])[0]
            
            return predicted_class, confidence
    
        return None, 0.0
def camera_thread():
    global camera_active, last_prediction
    global bad_posture_tracking

    # Thêm import time ở đầu file nếu chưa có
    import time

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_count = 0
    prediction_interval = 10

    while camera_active:
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        
        if frame_count % prediction_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                features = extract_features_from_landmarks(results.pose_landmarks.landmark)
                predicted_class, confidence = predict_posture(features, models)

                # Cập nhật last_prediction
                last_prediction = {
                    "class": predicted_class,
                    "confidence": float(confidence)
                }

                # Xử lý theo dõi tư thế sai
                if predicted_class not in ["straight_back", "vai_thang"]:
                    if bad_posture_tracking["current_bad_posture"] != predicted_class:
                        bad_posture_tracking["current_bad_posture"] = predicted_class
                        bad_posture_tracking["start_time"] = time.time()
                    else:
                        current_time = time.time()
                        duration = current_time - bad_posture_tracking["start_time"]
                        if duration >= 5:
                            # Ở đây bạn có thể thêm logic xử lý khi phát hiện tư thế sai quá 5 giây
                            print(f"Cảnh báo: {predicted_class} trong {duration} giây")
                else:
                    # Reset tracking khi về tư thế đúng
                    bad_posture_tracking["current_bad_posture"] = None
                    bad_posture_tracking["start_time"] = None

        # Đưa frame vào queue
        try:
            if not frame_queue.full():
                frame_queue.put_nowait(frame)
        except queue.Full:
            pass

    cap.release()


@app.get("/get_audio/{posture}")
async def get_audio(posture: str):
    if posture in POSTURE_AUDIO_MAPPING:
        audio_path = os.path.join(AUDIO_DIR, POSTURE_AUDIO_MAPPING[posture])
        
        if os.path.exists(audio_path):
            return FileResponse(
                path=audio_path, 
                media_type="audio/mpeg", 
                filename=POSTURE_AUDIO_MAPPING[posture]
            )
    
    return JSONResponse(
        status_code=404, 
        content={"message": "Không tìm thấy file âm thanh"}
    )

last_alert_time = 0
ALERT_COOLDOWN = 10  # Thời gian chờ giữa các lần cảnh báo (giây)
@app.get("/check_posture_alert")
async def check_posture_alert():
    global last_alert_time
    current_bad_posture = bad_posture_tracking["current_bad_posture"]
    current_time = time.time()
    
    if current_bad_posture and current_bad_posture not in ["straight_back", "vai_thang"]:
        duration = current_time - bad_posture_tracking["start_time"]
        
        if duration >= 5 and (current_time - last_alert_time >= ALERT_COOLDOWN):
            last_alert_time = current_time
            return JSONResponse({
                "alert": True, 
                "posture": translate_posture_to_vi(current_bad_posture),
                "audio_url": f"/get_audio/{current_bad_posture}",
                "duration": duration
            })
    
    return JSONResponse({
        "alert": False, 
        "posture": None, 
        "audio_url": None, 
        "duration": 0
    })

@app.get("/start_camera")
async def start_camera():
    global camera_active
    if not camera_active:
        camera_active = True
        threading.Thread(target=camera_thread, daemon=True).start()
    return JSONResponse({"status": "Camera started"})

@app.get("/stop_camera")
async def stop_camera():
    global camera_active
    camera_active = False
    return JSONResponse({"status": "Camera stopped"})

@app.get("/video_feed")
async def video_feed():
    async def generate():
        while camera_active:
            try:
                frame = frame_queue.get(timeout=1)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except queue.Empty:
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Lỗi video feed: {e}")
                break

    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/get_prediction")
async def get_prediction():
    global last_prediction
    prediction_vi = last_prediction.copy()
    prediction_vi['class_vi'] = translate_posture_to_vi(last_prediction.get('class', ''))
    return JSONResponse(content=prediction_vi)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000)
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf  # O PyTorch si usas otro framework
import socket
import json
from tensorflow.keras import losses

# Carga el modelo especificando las funciones estándar
model = tf.keras.models.load_model("my_model.h5", custom_objects={"mse": losses.MeanSquaredError()})


# Conexión al socket del NAO
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("127.0.0.1", 6000))  # IP del servidor NAO/controlador

# Inicialización MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

def normalize_data(position, orientation):
    position_max = [3000, 3000, 3000]  # Máximos de los valores de posición (en milímetros)
    orientation_max = [180, 180, 180]  # Máximos de grados para la orientación

    # Normalización
    position_norm = [p / m for p, m in zip(position, position_max)]
    orientation_norm = [r / m for r, m in zip(orientation, orientation_max)]
    
    return position_norm + orientation_norm

# Función para extraer la posición y orientación del efector final (mano)
def extract_hand_position_and_orientation(holistic_results):
    # Puntos clave de los brazos
    left_wrist = holistic_results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST]
    right_wrist = holistic_results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]
    
    # Extraer posición de la mano (efector final)
    left_hand_position = [left_wrist.x, left_wrist.y, left_wrist.z]
    right_hand_position = [right_wrist.x, right_wrist.y, right_wrist.z]
    
    # Aquí puedes calcular la orientación basada en los codos y hombros
    # Para simplificar, asumimos que usamos solo la posición para la predicción, pero puedes agregar la orientación usando rotaciones.
    return left_hand_position, right_hand_position

# Captura de video y procesamiento
with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir la imagen a RGB para MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar la imagen con MediaPipe
        holistic_results = holistic.process(image_rgb)

        annotated_frame = frame.copy()
        mp_drawing.draw_landmarks(annotated_frame, holistic_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        
        # Si se detecta cuerpo, extraemos la posición y orientación del efector final
        if holistic_results.pose_landmarks:
            left_hand_position, right_hand_position = extract_hand_position_and_orientation(holistic_results)
            
            # Normalizar las posiciones y orientaciones
            input_data_left = normalize_data(left_hand_position, [0, 0, 0])  # La orientación se puede calcular, pero por ahora ponemos [0, 0, 0]
            input_data_right = normalize_data(right_hand_position, [0, 0, 0])

            # Predicción de los ángulos usando el modelo
            input_array = np.array([input_data_left + input_data_right])  # Combina ambas manos si es necesario
            predicted_angles = model.predict(input_array)

            # Mapeo de los valores predichos a los ángulos de las articulaciones
            angles = {
                "LShoulderPitch": predicted_angles[0][0],
                "LShoulderRoll": predicted_angles[0][1],
                "LElbowYaw": predicted_angles[0][2],
                "LElbowRoll": predicted_angles[0][3],
                "LWristYaw": predicted_angles[0][4],
                "RShoulderPitch": predicted_angles[0][5],
                "RShoulderRoll": predicted_angles[0][6],
                "RElbowYaw": predicted_angles[0][7],
                "RElbowRoll": predicted_angles[0][8],
                "RWristYaw": predicted_angles[0][9]
            }

            # Enviar los ángulos al robot NAO a través del socket
            try:
                payload = json.dumps(angles)
                print("Enviando al NAO:", payload)
                sock.sendall(payload.encode("utf-8"))
                print("-->>> Ángulos enviados.")
            except Exception as e:
                print("Error al enviar ángulos:", e)
        
        # Mostrar la imagen con las marcas de MediaPipe
        cv2.imshow("NAO Tracker", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
        
        if cv2.waitKey(1) & 0xFF == 27:  # Esc para salir
            break

cap.release()
cv2.destroyAllWindows()
sock.close()

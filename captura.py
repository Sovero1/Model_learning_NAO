import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model


# === CONFIGURACIÓN ===

# Cargar modelo entrenado
modelo = load_model('my_model.h5', compile=False)


# Media y desviación estándar usadas en el entrenamiento (ajusta con tus valores)
media = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])  # ejemplo: centro de imagen
std = np.array([0.2, 0.2, 0.2, 1.0, 1.0, 1.0])    # ejemplo

# Inicializar MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


# Inicializar cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Extraer landmarks necesarios
        shoulder = np.array([lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                             lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                             lm[mp_pose.PoseLandmark.LEFT_SHOULDER].z])
        elbow = np.array([lm[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                          lm[mp_pose.PoseLandmark.LEFT_ELBOW].y,
                          lm[mp_pose.PoseLandmark.LEFT_ELBOW].z])
        wrist = np.array([lm[mp_pose.PoseLandmark.LEFT_WRIST].x,
                          lm[mp_pose.PoseLandmark.LEFT_WRIST].y,
                          lm[mp_pose.PoseLandmark.LEFT_WRIST].z])

        # Calcular posición media como [Px, Py, Pz]
        position = (shoulder + wrist) / 2

        # Calcular orientación: vector del hombro al codo
        orientation = elbow - shoulder
        norm = np.linalg.norm(orientation)
        if norm == 0:
            continue  # evitar división por cero
        orientation /= norm  # Rx, Ry, Rz

        entrada = np.concatenate([position, orientation])  # [Px, Py, Pz, Rx, Ry, Rz]

        # Normalizar la entrada
        entrada_norm = (entrada - media) / std  
        entrada_norm = np.array(entrada_norm, dtype=np.float32).reshape(1, -1)
                #print("Entrada para el modelo:", entrada_norm.shape, entrada_norm.dtype)


        # Predicción
        
        angulos = modelo.predict(entrada_norm)[0]  # [θ1, θ2, θ3, θ4, θ5]

        # Mostrar resultados
        for i, ang in enumerate(angulos):
            cv2.putText(frame, f"Theta {i+1}: {ang:.3f} rad",
                        (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        # Dibujar los puntos en pantalla
        h, w, _ = frame.shape
        for pt in [shoulder, elbow, wrist]:
            cx, cy = int(pt[0] * w), int(pt[1] * h)
            cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1)

    cv2.imshow('Brazo izquierdo + predicción', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()

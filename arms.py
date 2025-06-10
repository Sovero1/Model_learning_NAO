import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Cargar modelo entrenado
modelo = load_model('my_model.h5', compile=False)

# Media y desviación estándar (ajusta con los valores de entrenamiento reales)
media = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
std = np.array([0.2, 0.2, 0.2, 1.0, 1.0, 1.0])

# Configurar MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

# Longitudes de segmentos del brazo
long_upper_arm = 0.1
long_forearm = 0.1

# Matrices de rotación
def rot_x(theta):
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])

def rot_y(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def rot_z(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta),  np.cos(theta), 0],
                     [0, 0, 1]])

# Inicializar visualización
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        shoulder = np.array([lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                             lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                             lm[mp_pose.PoseLandmark.LEFT_SHOULDER].z])
        elbow = np.array([lm[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                          lm[mp_pose.PoseLandmark.LEFT_ELBOW].y,
                          lm[mp_pose.PoseLandmark.LEFT_ELBOW].z])
        wrist = np.array([lm[mp_pose.PoseLandmark.LEFT_WRIST].x,
                          lm[mp_pose.PoseLandmark.LEFT_WRIST].y,
                          lm[mp_pose.PoseLandmark.LEFT_WRIST].z])

        position = (shoulder + wrist) / 2
        orientation = elbow - shoulder
        norm = np.linalg.norm(orientation)
        if norm == 0:
            continue
        orientation /= norm

        entrada = np.concatenate([position, orientation])
        entrada_norm = (entrada - media) / std
        entrada_norm = np.asarray(entrada_norm, dtype=np.float32).reshape(1, -1)

        angulos = modelo.predict(entrada_norm, verbose=0)[0]

        # Visualización 3D del brazo NAO
        ax.clear()
        ax.set_xlim([-0.2, 0.2])
        ax.set_ylim([-0.2, 0.2])
        ax.set_zlim([-0.2, 0.2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Simulación brazo NAO (tiempo real)')

        base = np.array([0, 0, 0])
        shoulder_rot = rot_y(angulos[0]) @ rot_x(angulos[1])
        upper_arm_vector = shoulder_rot @ np.array([long_upper_arm, 0, 0])
        elbow_pos = base + upper_arm_vector

        elbow_rot = rot_z(angulos[2]) @ rot_x(angulos[3])
        forearm_vector = elbow_rot @ np.array([long_forearm, 0, 0])
        wrist_pos = elbow_pos + forearm_vector

        ax.plot([base[0], elbow_pos[0]], [base[1], elbow_pos[1]], [base[2], elbow_pos[2]], 'b-', label='Brazo sup.')
        ax.plot([elbow_pos[0], wrist_pos[0]], [elbow_pos[1], wrist_pos[1]], [elbow_pos[2], wrist_pos[2]], 'r-', label='Antebrazo')
        ax.scatter(*base, color='k', label='Hombro')
        ax.scatter(*elbow_pos, color='g', label='Codo')
        ax.scatter(*wrist_pos, color='m', label='Muñeca')
        ax.legend()
        plt.pause(0.001)

    cv2.imshow('MediaPipe Pose', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()

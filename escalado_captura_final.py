import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe para la estimación de poses
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Inicializar OpenCV para capturar video desde la cámara
cap = cv2.VideoCapture(1)

# Función para escalar las coordenadas a un espacio cartesiano del robot
def scale_to_robot_space(x, y, z, scale_x=1.0, scale_y=1.0, scale_z=1.0):
    robot_x = (x - 0.5) * scale_x  # Escala para el eje X
    robot_y = (y - 0.5) * scale_y  # Escala para el eje Y
    robot_z = (z - 0.5) * scale_z  # Escala para el eje Z
    return robot_x, robot_y, robot_z

# Función para calcular el vector de orientación
def calculate_orientation(shoulder, elbow, wrist):
    # Vectores entre hombro, codo y muñeca
    vector_shoulder_elbow = np.array([elbow.x - shoulder.x, elbow.y - shoulder.y, elbow.z - shoulder.z])
    vector_elbow_wrist = np.array([wrist.x - elbow.x, wrist.y - elbow.y, wrist.z - elbow.z])
    
    # Normalización de los vectores
    vector_shoulder_elbow /= np.linalg.norm(vector_shoulder_elbow)
    vector_elbow_wrist /= np.linalg.norm(vector_elbow_wrist)
    
    # Producto cruzado para calcular la orientación
    cross_product = np.cross(vector_shoulder_elbow, vector_elbow_wrist)
    orientation = cross_product / np.linalg.norm(cross_product)  # Normalización final
    return orientation

# Visualización de los datos de entrada para el modelo
def show_input_data(humanoid_coords, robot_coords, orientation):
    print("Datos de Entrada (Humano) - Normalizados:")
    for idx, (h_coord, r_coord) in enumerate(zip(humanoid_coords, robot_coords)):
        print(f"Articulación {idx+1} - Humano (X: {h_coord[0]:.3f}, Y: {h_coord[1]:.3f}, Z: {h_coord[2]:.3f}) -> Robot (X: {r_coord[0]:.3f}, Y: {r_coord[1]:.3f}, Z: {r_coord[2]:.3f})")
    print(f"Orientación (Vectores R_x, R_y, R_z): {orientation}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Convertir la imagen a RGB para usarla en MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Dibujar los resultados de los landmarks en el frame
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)  # Convertir de nuevo a BGR para OpenCV
    if results.pose_landmarks:
        # Extraer landmarks de la persona (brazos)
        landmarks = results.pose_landmarks.landmark
        
        # Extraer las coordenadas de la muñeca, codo y hombro del brazo izquierdo (índices específicos)
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

        # Coordenadas del brazo izquierdo en el espacio de la cámara (normalizadas entre 0 y 1)
        left_shoulder_coords = (left_shoulder.x, left_shoulder.y, left_shoulder.z)
        left_elbow_coords = (left_elbow.x, left_elbow.y, left_elbow.z)
        left_wrist_coords = (left_wrist.x, left_wrist.y, left_wrist.z)

        # Escalar las coordenadas al espacio cartesiano del robot
        left_shoulder_robot = scale_to_robot_space(left_shoulder.x, left_shoulder.y, left_shoulder.z)
        left_elbow_robot = scale_to_robot_space(left_elbow.x, left_elbow.y, left_elbow.z)
        left_wrist_robot = scale_to_robot_space(left_wrist.x, left_wrist.y, left_wrist.z)

        # Calcular la orientación (vectores R_x, R_y, R_z) del efector final
        orientation = calculate_orientation(left_shoulder, left_elbow, left_wrist)

        # Mostrar los datos de entrada que se usarán para el modelo
        humanoid_coords = [left_shoulder_coords, left_elbow_coords, left_wrist_coords]
        robot_coords = [left_shoulder_robot, left_elbow_robot, left_wrist_robot]
        show_input_data(humanoid_coords, robot_coords, orientation)

        # Visualización de los landmarks en el video
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Dibujar círculos en los puntos de interés (muñeca, codo, hombro)
        cv2.circle(frame, (int(left_wrist.x * frame.shape[1]), int(left_wrist.y * frame.shape[0])), 5, (0, 255, 0), -1)
        cv2.circle(frame, (int(left_elbow.x * frame.shape[1]), int(left_elbow.y * frame.shape[0])), 5, (255, 0, 0), -1)
        cv2.circle(frame, (int(left_shoulder.x * frame.shape[1]), int(left_shoulder.y * frame.shape[0])), 5, (0, 0, 255), -1)

    # Mostrar el frame
    cv2.imshow('Pose Estimation', frame)

    # Para cerrar la ventana, presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Iniciar captura de video
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Voltear la imagen horizontalmente para facilitar el uso
    frame = cv2.flip(frame, 1)

    # Convertir imagen BGR a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Procesar la imagen y obtener resultados
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujar los landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extraer las posiciones de los landmarks de la mano
            for landmark in hand_landmarks.landmark:
                # Cada landmark tiene x, y, z (coordenadas en el espacio de la imagen)
                print(f"x: {landmark.x}, y: {landmark.y}, z: {landmark.z}")

    # Mostrar el video con las manos detectadas
    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

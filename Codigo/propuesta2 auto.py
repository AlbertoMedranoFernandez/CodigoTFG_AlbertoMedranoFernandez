import cv2
import mediapipe as mp
import math
import time

def main():
    # Variables
    parpadeo = False
    contParpadeos = 0
    tiempo = 0
    inicio = 0
    final = 0
    contSueños = 0

    calibrado = False
    cont = 0
    umbral = 0

    # Encender cámara
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    # Para dibujar malla facial
    drawingSpec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

    # Para crear la malla facial
    faceMesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)

    # Bucle principal
    while True:

        ret, frame = cap.read()
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Corrección de color

        resultados = faceMesh.process(frameRGB)

        if resultados.multi_face_landmarks:  # Si se detecta alguna cara
            for rostros in resultados.multi_face_landmarks:  # Mostrar cara detectada
                mp.solutions.drawing_utils.draw_landmarks(frame, rostros, mp.solutions.face_mesh.FACEMESH_CONTOURS, None, drawingSpec)

                # Extraer los puntos de la cara detectada
                lista = extraerPuntos(frame, rostros.landmark)

                # Ojo derecho
                longitud1 = calcularDistancia(lista[145], lista[159])

                # Ojo izquierdo
                longitud2 = calcularDistancia(lista[374], lista[386])

                # Frente y barbilla
                longitud3 = calcularDistancia(lista[10], lista[152])

                # Normalizar
                longitud1, longitud2, longitud3 = normalizar(longitud1, longitud2, longitud3)

                # Calibrado automatico
                if calibrado == False:
                    cv2.putText(frame, f'Calibrando...', (75, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
                    if cont < 60:
                        cv2.putText(frame, f'IMPORTANTE: Mantenga los ojos ABIERTOS.',
                                    (75, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        cont += 1
                        umbral += longitud1 + longitud2
                    else:
                        calibrado = True
                        umbral = obtenerUmbral(umbral, cont * 2)

                # Cuando ya está calibrado
                else:
                    # Controlar inicio y final del parpadeo
                    print(longitud3, longitud1, longitud2, umbral)
                    parpadeo, finalParpadeo, inicio, final = parpadear(longitud1, longitud2, parpadeo, inicio, final, umbral)

                    # Comporbar si ha sido micro sueño
                    if finalParpadeo == True:
                        tiempo, contParpadeos, contSueños, finalParpadeo = esMicroSueño(inicio, final, contParpadeos, contSueños)

                    # Mostrar datos
                    mostrarDatos(frame, contParpadeos, contSueños, tiempo)

        cv2.imshow('Frame', frame)
        key = cv2.waitKey(25) & 0xFF

        # Pulsar "q" para acabar
        if key == ord("q"):
            break

    # Cerrar la cámara
    cap.release()
    cv2.destroyAllWindows()


# Función para pasar las coordenadas de los puntos de referencia a píxeles
def extraerPuntos(frame, landmarks):
    lista = []

    for id in range(len(landmarks)):
        al, an, c = frame.shape  # alto, ancho, canales
        x, y = int(landmarks[id].x * an), int(landmarks[id].y * al)
        lista.append([x, y])
        if len(lista) == 468:
            return lista

# Función para calcular la distancia entre dos puntos de referencia
def calcularDistancia(punto1, punto2):
    return math.hypot(punto2[0] - punto1[0], punto2[1] - punto1[1])

# Función para normalizar las distancias entre 0 y 1
def normalizar(longitud1, longitud2, longitud3):
    return longitud1 / longitud3, longitud2 / longitud3, 1

# Función para calcular el umbral
def obtenerUmbral(umbral, cont):
    return (umbral / cont) * 0.9

# Función para controlar cuando se abren y cierran los ojos
def parpadear(longitud1, longitud2, parpadeo, inicio, final, umbral):

    finalParpadeo = False

    if longitud1 <= umbral and longitud2 <= umbral and parpadeo == False:
        parpadeo = True
        inicio = time.time()

    if longitud1 > umbral and longitud2 > umbral and parpadeo == True:
        parpadeo = False
        final = time.time()
        finalParpadeo = True

    return parpadeo, finalParpadeo, inicio, final

# Función para determinar si se ha producido un parpadeo o un micro sueño
def esMicroSueño(inicio, final, contParpadeos, contSueños):
    # Tiempo entre que se abre y cierra el ojo
    tiempo = round(final - inicio, 1)

    # Contador de Parpadeos
    if tiempo < 1:
        contParpadeos = contParpadeos + 1

    # Contador de Micro Sueños
    elif tiempo >= 1:
        contSueños = contSueños + 1

    finalParpadeo = False

    return tiempo, contParpadeos, contSueños, finalParpadeo

# Función para mostrar en la ventana el número de parpadeos y de micro sueños y el tiempo
def mostrarDatos(frame, contParpadeos, contSueños, tiempo):
    cv2.putText(frame, f'Parpadeos: {int(contParpadeos)}', (75, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 3)
    cv2.putText(frame, f'Micro sueños: {int(contSueños)}', (75, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 3)
    cv2.putText(frame, f'Duracion: {str(tiempo)}', (75, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                3)


if __name__ == "__main__":
    main()
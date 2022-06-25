from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
 
def eye_aspect_ratio(eye):
# Calcula las distancias euclidianas entre los dos conjuntos de
# Señales verticales del ojo (x, y) -coordenadas
  A = dist.euclidean(eye[1], eye[5])
  B = dist.euclidean(eye[2], eye[4])
 
# Calcular la distancia euclidiana entre la horizontal
  C = dist.euclidean(eye[0], eye[3])
 
# calcula AR del ojo
  ear = (A + B) / (2.0 * C)
 
# devuelve AR del ojo
  return ear
 
mStart=48
mEnd=68
jStart=0
jEnd= 17
rlStart=17
rlEnd= 22
leStart=22
leEnd= 27
nStart=27
nEnd= 36
 
# Se definen dos constantes, una para la relación de aspecto del ojo para indicar
# el parpadeo y luego una segunda constante para el número de
# Frames en que el ojo debe estar por debajo del umbral
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 2
sleep1_FRAMES = 3*5
sleep2_FRAMES = 7*5
sleep3_FRAMES = 8*5
sleep4_FRAMES = 10*5
sleep01_FRAMES = 6
sleep00_FRAMES = 2
 
 
# contadores para parpadeo
COUNTER = 0
TOTAL = 0
sleep=0;
gradsleep=0
 
# Inicializar la detección de cara con la libreria dlib (HOG-based) y luego
# usa el predictor del hito facial
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
 
#se selecionan los indices del los ojos
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(jStart,jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
(reStart,reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
(leStart,leEnd)= face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(nStart,nEnd)= face_utils.FACIAL_LANDMARKS_IDXS["nose"]
 
 
cap = cv2.VideoCapture(2)
fileStream = False
time.sleep(1.0)
 
# ciclo de procesado ppal
while True:
# Si se trata de un archivo de flujo de vídeo, entonces tenemos que comprobar si
# Hay más cuadros dejados en el búfer para procesar
# Si fileStream y no vs.more ():
# descanso
 
# Agarrar el marco de la secuencia de archivo de vídeo de rosca, cambiar el tamaño
# It, y convertirlo a escala de grises
# Canales)
  ret, frame = cap.read()
  #frame = imutils.resize(frame, width=450)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
# detecta caras en la imagen en escala de grises
  rects = detector(gray, 0)
 
# ciclo sobre las detecciones de la cara
  for rect in rects:
# Determina las marcas faciales para la región de la cara, luego
# Convierte el punto de referencia facial (x, y) - a coordenada NumPy
#Array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
 
# Extrae las coordenadas de los ojos izquierdo y derecho y calcula
# la relación de aspecto (AR) del ojo para ambos ojos
 
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    mouth = shape[mStart:mEnd]
    jaw = shape[jStart:jEnd]
    re = shape[reStart:reEnd]
    le = shape[leStart:leEnd]
    nose= shape[nStart:nEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
 
# media de AR para los dos ojos
    ear = (leftEAR + rightEAR) / 2.0
 
# hace la convex hull para los dos ojos
# se dibujan los dos ojos
    leftEyeHull = cv2.convexHull(leftEye)
    mouthHull = cv2.convexHull(mouth)
    jawHull = cv2.convexHull(jaw)
    reHull = cv2.convexHull(re)
    leHull = cv2.convexHull(le)
    noseHull = cv2.convexHull(nose)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
# cv2.drawContours(frame, [mouthHull], -1, (255, 255, 0), 1)
# cv2.drawContours(frame, [jaw], -1, (255, 55, 0), 1)
# cv2.drawContours(frame, [reHull], -1, (255, 2, 60), 1)
# cv2.drawContours(frame, [leHull], -1, (255, 255, 200), 1)
# cv2.drawContours(frame, [noseHull], -1, (5, 255, 200), 1)
 
# Compruebe si la relación de aspecto del ojo está por debajo del parpadeo
#, Y si es así se incrementa el contador del marco intermitente
    if ear < EYE_AR_THRESH:
       COUNTER += 1
 
# De lo contrario, la relación de aspecto del ojo no está por debajo del parpadeo
# límite
    else:
# Si los ojos estaban cerrados por un número suficiente de
# Luego incrementar el número total de parpadeos</pre>
 
      if COUNTER >= EYE_AR_CONSEC_FRAMES:
         TOTAL += 1
      if COUNTER >= sleep00_FRAMES:
         gradsleep = 0.2
      if COUNTER >= sleep01_FRAMES:
         gradsleep = 0.5
      if COUNTER >= sleep1_FRAMES:
         gradsleep = 1
      if COUNTER >= sleep2_FRAMES:
         gradsleep = 2
      if COUNTER >= sleep3_FRAMES:
         gradsleep = 4
      if COUNTER >= sleep4_FRAMES:
         gradsleep = 10
 
# reseteo del contador
      COUNTER = 0
 
# Dibuja el número total de destellos en el marco junto con
# la relación de aspecto calculada del ojo para el marco
    cv2.putText(frame, "Parpadeo: {}".format(TOTAL), (10, 30),
       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "sleeping grade (0:10): {:.2f}".format(gradsleep), (300, 30),
       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
# mostramos el frame
  cv2.imshow("Frame", frame)
  key = cv2.waitKey(1) & 0xFF
 
# si pulsa q se rompe el ciclo
  if key == ord("q"):
     break
 
# limpiamos un poco
cap.release()
cv2.destroyAllWindows()
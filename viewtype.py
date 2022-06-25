import numpy as np
import cv2
import requests
import os
import dlib
import imutils

#Descargamos los clasificadores
face = "haarcascade.xml"
if not os.path.isfile(face):
	url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
	content = requests.get(url).content
	f = open(face, "wb")
	f.write(content)

eye = "eyecascade.xml"
if not os.path.isfile(eye):
	url_ = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
	content_ = requests.get(url_).content
	f = open(eye, "wb")
	f.write(content_)


#Clasificadores
face_cascade = cv2.CascadeClassifier(face)
eye_cascade = cv2.CascadeClassifier(eye)

print("--------------------------------------")
print(type(eye_cascade))
print(type(face_cascade))	

#Captura de video
cap = cv2.VideoCapture(0)

cv2.namedWindow("capturador", cv2.WINDOW_NORMAL)

while 1:
	ret, frame = cap.read()
	#Imagen en escala de grises
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	#Deteccion de rostros con nuestro clasificador
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	
	#(coordenadas en xy, ancho y alto)
	for (x,y,w,h) in faces:
		#Dibujar rectangulo(imagen, coordenadas, ancho y alto, color en RGB, grosor)
		img = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
		cv2.putText(img = img,
                text = "humano",
                org = (x-10, y-10),
                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.6,
                color = (0,0,255),
                thickness= 1
                )
		#Region del rostro en la imagen en escala de grises
		roi_gray = gray[y:y+h, x:x+w]
		
		#Region del rostro en la imagen a colores
		roi_color = img[y:y+h, x:x+w]
		
		#Deteccion de ojos por cada rostro
		#Recuerda que el kernel o filtro solo funciona con imagenes en escala de grises
		eyes = eye_cascade.detectMultiScale(roi_gray)
		
		#(coordenadas en xy, ancho y alto)
		for (ex,ey,ew,eh) in eyes:
			#Dibuja el rectangulo en la region del rostro
			img = cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (255,0,0), 2)
			
	
	cv2.imshow("frame", frame)
	cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
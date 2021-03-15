import cv2

carregaAlgoritmo = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
image = cv2.imread('images/image7.jpg')

greyImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = carregaAlgoritmo.detectMultiScale(greyImage, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

print(faces)

for (x, y, l, a) in faces:
    cv2.rectangle(image, (x, y), (x + l, y + a), (255, 0, 255), 2)

cv2.imshow("Faces ", image)
cv2.waitKey()

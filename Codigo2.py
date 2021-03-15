import cv2
carregaFace = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
carregaOlho = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

image = cv2.imread('images/image3.jpg')

greyImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = carregaFace.detectMultiScale(greyImage)


print(faces)

for (x, y, l, a) in faces:
    cv2.rectangle(image, (x, y), (x + l, y + a), (255, 0, 255), 2)

cv2.imshow("Faces ", image)
cv2.waitKey()

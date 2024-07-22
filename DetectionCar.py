import cv2

video_src = 'resource/video480.avi'

cap = cv2.VideoCapture(video_src)
plate_cascade = cv2.CascadeClassifier('cascades/haarcascade_licence_plate_rus_16stages.xml')

while True:
    ret, img = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in plates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('video-plate', img)

    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
cap.release()

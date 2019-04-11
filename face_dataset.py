import cv2
import os
import time

cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

global index
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id

# face_id = input('\n enter user id end press <return> ==>  ')
face_id = 1

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count


# Initialize the first fps
t_start = time.time()
fps = 0
sfps = 0
def get_faces( img ):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    return faces, gray

def draw_frame( img, faces, gray):
    global index
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        index += 1
        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(index) + ".jpg", gray[y:y + h, x:x + w])
    cv2.imshow('image', img)

if __name__ == '__main__':
    global index
    index = 0
    count = 0
    while (True):
        #get the images data
        ret, img = cam.read()

        #skip the p
        if count%5==0:
            faces, gray= get_faces(img)
        count = (count+1)%5

        draw_frame(img, faces, gray)

        k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
        elif index >= 30:  # Take 30 face sample and stop video
            # print("ok")
            break

        cv2.putText(img, "FPS : " + str(int(sfps)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)



        # Calculate and show the FPS
        # fps = fps + 1
        # sfps = fps / (time.time() - t_start)
        # print(sfps,fps)


    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()



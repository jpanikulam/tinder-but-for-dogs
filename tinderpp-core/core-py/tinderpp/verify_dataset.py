import sys
import os
import cv2
from vision import ImageScanner
fpath = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    dataset_location = os.path.join(fpath, '..', '..', 'data')
    imscan = ImageScanner()

    people = os.listdir(dataset_location)
    for person in people:
        fullpath = os.path.join(dataset_location, person)
        if not os.path.isdir(fullpath):
            print "Skipping ", person
            continue

        print 'Viewing {}'.format(person)
        for image_name in os.listdir(fullpath):
            img = cv2.imread(os.path.join(fullpath, image_name))
            img = cv2.resize(img, (img.shape[1] / 2, img.shape[0] / 2), interpolation=cv2.INTER_CUBIC)

            detection = imscan.detect_face(img)
            if detection is not None:
                x0, y0, x1, y1 = detection
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(img, "{} at {}".format(person, image_name), (x0, y1 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 50))
            else:
                print image_name, 'no face detected'

            cv2.imshow("image", img)
            key = cv2.waitKey(0) & 0xff
            if key == ord('q'):
                sys.exit()

import cv2
import os
import fr_tools
from facedet.detector import CascadedDetector
from pulse_detect.processors_noopenmdao import findFaceGetPulse
from vision import ImageScanner
from haar import Detector
fpath = os.path.dirname(os.path.abspath(__file__))


class App(object):
    def __init__(self, model, cascade_filename):
        self.imscan = ImageScanner()
        self.cap = cv2.VideoCapture(0)

    def run(self):
        k = 0
        while(True):
            k += 1
            ret, frame = self.cap.read()
            img = cv2.resize(frame, (frame.shape[1] / 2, frame.shape[0] / 2), interpolation=cv2.INTER_CUBIC)
            imgout = img.copy()

            # Get BPM
            bpm = self.imscan.get_bpm(img)

            # Get person's name
            face_detection = self.imscan.recognize_face(img)
            if face_detection is not None:
                person_name, bounding_box, is_smiling = face_detection
                x0, y0, x1, y1 = bounding_box
                cv2.rectangle(imgout, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(imgout, "{} Hz".format(bpm), (x1 + 5, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 50))
                if is_smiling != []:
                    print "Smiling!"
                else:
                    print "Not smiling"
                    # x01, y01, x11, y11 = is_smiling[0]
                    # cv2.rectangle(imgout, (x01 + x0, y01 + y0), (x11 + x1, y11 + y1), (100, 100, 0), 2)

            print self.imscan.likeness

            # print self.smile_detector.classify(face)
            cv2.imshow("Display", imgout)
            key = cv2.waitKey(10) & 0xff
            if key == ord('q'):
                break

if __name__ == '__main__':
    # dataset_location = os.path.join(fpath, '..', '..', 'data')
    # image_size = (100, 100)  # TODO: Fiddle with this

    # [images, labels, subject_names] = fr_tools.read_images(dataset_location, image_size)
    # Zip us a {label, name} dict from the given data:
    # list_of_labels = list(xrange(max(labels) + 1))
    # subject_dictionary = dict(zip(list_of_labels, subject_names))
    # Get the model we want to compute:
    # model = fr_tools.get_model(image_size=image_size, subject_names=subject_dictionary)

    # TODO: Do this only once and cache the model
    # model.compute(images, labels)
    # And save the model, which uses Pythons pickle module:
    print "Saving the model..."
    # save_model(model_filename, model)
    # print os.listdir(os.path.join(fpath, '..', 'facerec/py/apps/videofacerec'))
    p = App(None, os.path.join(fpath, '..', 'facerec/py/apps/videofacerec/haarcascade_frontalface_alt2.xml'))
    p.run()

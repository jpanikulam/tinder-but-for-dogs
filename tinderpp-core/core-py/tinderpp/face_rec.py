import cv2
import os
import fr_tools
from facedet.detector import CascadedDetector
from pulse_detect.processors_noopenmdao import findFaceGetPulse
from vision import ImageScanner
from haar import Detector
fpath = os.path.dirname(os.path.abspath(__file__))


class App(object):
    def __init__(self):
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
                cv2.putText(imgout, "{} Hz".format(bpm), (x1 + 5, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 50))
                cv2.putText(imgout, person_name, (x0, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 20, 255))
                if is_smiling != []:
                    print "Smiling!"
                else:
                    print "Not smiling"

            print self.imscan.likeness
            cv2.imshow("Display", imgout)
            key = cv2.waitKey(10) & 0xff
            if key == ord('q'):
                break

if __name__ == '__main__':
    print "Saving the model..."
    p = App()
    p.run()

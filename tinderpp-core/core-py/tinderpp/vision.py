from collections import deque
import numpy as np
import cv2
import os
import fr_tools
from facedet.detector import CascadedDetector
from pulse_detect.processors_noopenmdao import findFaceGetPulse
from haar import Detector
fpath = os.path.dirname(os.path.abspath(__file__))


class ImageScanner(object):
    _face_cascade = os.path.join(fpath, '..', 'facerec/py/apps/videofacerec/haarcascade_frontalface_alt2.xml')
    _dataset_location = os.path.join(fpath, '..', '..', 'data')
    _smile_weight = 1.0
    _heartbeat_weight = 0.1

    def __init__(self):
        self.bpm_detector = findFaceGetPulse(
            bpm_limits=[50, 160],
            data_spike_limit=2500.,
            face_detector_smoothness=10.,
            cascade_path=self._face_cascade)
        self.bpm_attempts = 0
        self.last_bpm = 61.2
        self.face_detector = self.make_face_detector()
        self.smile_detector = Detector('smile')
        self.heartbeat_deque = deque()
        self.likeness = 0

    def make_face_detector(self):
        image_size = (100, 100)  # TODO: Fiddle with this
        images, labels, subject_names = fr_tools.read_images(self._dataset_location, image_size)
        list_of_labels = list(xrange(max(labels) + 1))
        subject_dictionary = dict(zip(list_of_labels, subject_names))
        self.face_model = fr_tools.get_model(image_size=image_size, subject_names=subject_dictionary)
        self.face_model.compute(images, labels)

        face_detector = CascadedDetector(cascade_fn=self._face_cascade, minNeighbors=5, scaleFactor=1.1)
        return face_detector

    def downsample(self, image):
        img = cv2.resize(image, (image.shape[1] / 2, image.shape[0] / 2), interpolation=cv2.INTER_CUBIC)
        return img

    def detect_face(self, img):
        detections = self.face_detector.detect(img)
        for i, bounding_box in enumerate(detections):
            x0, y0, x1, y1 = bounding_box
            return x0, y0, x1, y1

    def recognize_face(self, img):
        # TODO: Accumulate/make probabilistic
        detections = self.face_detector.detect(img)
        for i, bounding_box in enumerate(detections):
            if i >= 1:
                print "Ignoring face at index > 1"
                break

            x0, y0, x1, y1 = bounding_box
            face = img[y0:y1, x0:x1]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            # print len(self.smile_detector.classify(face))
            face = cv2.resize(face, self.face_model.image_size, interpolation=cv2.INTER_CUBIC)
            smiling = self.smile_detector.classify(face)
            if smiling != []:
                self.likeness += self._smile_weight
            else:
                self.likeness -= self._smile_weight * 0.3
            prediction = self.face_model.predict(face)[0]
            person_name = self.face_model.subject_names[prediction]
            return person_name, bounding_box, smiling
        else:
            return None

    def get_bpm(self, image):
        self.bpm_attempts += 1
        # TODO: Check if we've detected a face
        # Try a few times first
        self.bpm_detector.frame_in = image
        self.bpm_detector.find_faces = bool(self.bpm_attempts < 5)
        self.bpm_detector.run(1)

        if self.bpm_attempts > 5:
            bpm = self.bpm_detector.bpm
            if bpm == 0:
                bpm = self.last_bpm
            elif bpm > 105:
                bpm = bpm / 2.0
            else:
                self.last_bpm = bpm
        else:
            bpm = 61.2 + np.random.normal(3.0)

        if len(self.heartbeat_deque) > 15:
            self.heartbeat_deque.popleft()
        self.heartbeat_deque.append(bpm)

        average = np.average(self.heartbeat_deque)

        self.likeness += (average - 60.0) * self._heartbeat_weight

        return average

import os
import cv2
fpath = os.path.dirname(os.path.abspath(__file__))


class Detector(object):
    _scaleFactor = 1.3
    _minNeighbors = 4

    # Try 25, 25
    _minSize = (30, 30)

    def __init__(self, cascade_type):
        self.cascades = {
            'eye': os.path.join(fpath, 'cascades', 'haarcascade_eye.xml'),
            'face': os.path.join(fpath, 'cascades', 'haarcascade_frontalface_alt.xml'),
            'smile': os.path.join(fpath, 'cascades', 'haarcascade_smile.xml'),
        }

        assert cascade_type in self.cascades.keys(), "Unknown cascade"
        self.classifier = cv2.CascadeClassifier(self.cascades[cascade_type])

    def classify(self, image):
        assert len(image.shape) == 2, "{} is too weird".format(image.shape)
        # classifications = self.classifier(image)
        detected = list(self.classifier.detectMultiScale(
            image,
            scaleFactor=self._scaleFactor,
            minNeighbors=self._minNeighbors,
            minSize=self._minSize,
            flags=cv2.CASCADE_SCALE_IMAGE))

        detected.sort(key=lambda a: a[-1] * a[-2])
        return detected


if __name__ == '__main__':
    d = Detector('smile')

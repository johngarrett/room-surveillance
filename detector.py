"""One CPU model instance; inference accepts full camera frames."""
class Detector:
    def __init__(self, model=None):
        if model is None:
            from ultralytics import YOLO
            model = YOLO('yolo11n.pt')
        self.model = model

    def has_person(self, frame, confidence: float) -> bool:
        results = self.model.predict(frame, imgsz=320, conf=confidence,
                                     classes=[0], device='cpu', verbose=False)
        return any(int(box.cls.item()) == 0 and float(box.conf.item()) >= confidence
                   for result in results for box in result.boxes)

# detectors/rfdetr_adapter.py

from rfdetr import RFDETRSegMedium

from .detector_output import DetectorOutput


class RFDETRSegAdapter:
    def __init__(
        self,
        weights_path,
        num_classes=11,
        threshold=0.3,
        optimize_for_inference=True,
    ):
        self.weights_path = weights_path
        self.num_classes = num_classes
        self.threshold = threshold

        print("Loading RF-DETRSeg detector...")

        self.model = RFDETRSegMedium(
            pretrain_weights=weights_path,
            num_classes=num_classes,
        )

        if optimize_for_inference:
            self.model.optimize_for_inference()

        self.class_names = self.model.class_names

        print("RF-DETR classes:")
        print(self.class_names)

    def predict(self, image_pil, image_np=None):
        detections = self.model.predict(
            image_pil,
            threshold=self.threshold,
        )

        return DetectorOutput.from_supervision(
            detections=detections,
            class_names=self.class_names,
        )

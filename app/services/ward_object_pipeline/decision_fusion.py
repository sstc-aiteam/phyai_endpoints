# decision_fusion.py


class ClassificationFusion:
    def __init__(
        self,
        min_rfdetr_confidence=0.50,
        min_dinov2_similarity=0.75,
        require_agreement=True,
    ):
        """
        Final classification rule:

        If RF-DETR and DINOv2 both pass and agree:
            final_class = class name

        Otherwise:
            final_class = unknown
        """

        self.min_rfdetr_confidence = min_rfdetr_confidence
        self.min_dinov2_similarity = min_dinov2_similarity
        self.require_agreement = require_agreement

    def classify_one(self, item):
        rfdetr_class = item.get("class_name", "unknown")
        rfdetr_conf = float(item.get("rfdetr_confidence", 0.0))

        dinov2_class = item.get("dinov2_label", "unknown")
        dinov2_score = float(item.get("dinov2_score", 0.0))

        rfdetr_valid = (
            rfdetr_class != "unknown"
            and rfdetr_conf >= self.min_rfdetr_confidence
        )

        dinov2_valid = (
            dinov2_class != "unknown"
            and dinov2_score >= self.min_dinov2_similarity
        )

        agreement = (
            rfdetr_valid
            and dinov2_valid
            and rfdetr_class == dinov2_class
        )

        if self.require_agreement:
            if agreement:
                final_class = rfdetr_class
                reason = "RF-DETR and DINOv2 agree"
            else:
                final_class = "unknown"
                reason = "RF-DETR and DINOv2 did not both pass and agree"

        else:
            if agreement:
                final_class = rfdetr_class
                reason = "RF-DETR and DINOv2 agree"

            elif rfdetr_valid and not dinov2_valid:
                final_class = rfdetr_class
                reason = "RF-DETR valid, DINOv2 inconclusive"

            elif dinov2_valid and not rfdetr_valid:
                final_class = dinov2_class
                reason = "DINOv2 valid, RF-DETR inconclusive"

            else:
                final_class = "unknown"
                reason = "Neither RF-DETR nor DINOv2 passed"

        return {
            **item,
            "final_class": final_class,
            "classification_reason": reason,
            "agreement": agreement,
            "rfdetr_valid": rfdetr_valid,
            "dinov2_valid": dinov2_valid,
        }

    def classify_all(self, verified_items):
        return [
            self.classify_one(item)
            for item in verified_items
        ]


# Backward-compatible alias in case older files still import DecisionFusion.
DecisionFusion = ClassificationFusion

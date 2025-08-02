from object_detection.utils import visualization_utils as viz_utils

def draw_boxes(image, detections, category_index, threshold=0.5):
    num_detections = int(detections.pop("num_detections"))
    detections = {k: v[0, :num_detections].numpy() for k, v in detections.items()}
    detections["num_detections"] = num_detections
    detections["detection_classes"] = detections["detection_classes"].astype("int64")

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image,
        detections["detection_boxes"],
        detections["detection_classes"],
        detections["detection_scores"],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=30,
        min_score_thresh=threshold,
        agnostic_mode=False,
    )
    return image

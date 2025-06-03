#!/usr/bin/env python3

"""
AprilTag Detection Module for Precision Landing

This module handles AprilTag detection in images and provides
augmented images with detection visualizations.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any
import base64

# Get logger
logger = logging.getLogger("precision-landing")

try:
    import apriltag
    APRILTAG_AVAILABLE = True
except Exception as e:
    APRILTAG_AVAILABLE = False
    logger.warning(f"AprilTag library not available: {str(e)}. Please install with: pip install apriltag")


def detect_april_tags(image: np.ndarray, tag_family: str, target_id: int, include_augmented_image: bool) -> Dict[str, Any]:
    """
    Detect AprilTags in an image and return the tag with the lowest ID

    Args:
        image: Input image as numpy array (BGR format from OpenCV)
        tag_family: AprilTag family to detect
        target_id: Target AprilTag ID to detect (-1 means detect any ID, 0+ means detect only that specific ID)
        include_augmented_image: Whether to return augmented image with detection boxes

    Returns:
        Dictionary containing:
        - success: bool indicating if detection was successful
        - detection: Single detection data for the tag with lowest ID (or None if no tags found)
        - image_base64: Base64 encoded image (augmented if include_augmented_image=True, original if False, empty if no image requested)
        - message: Status message
    """

    if not APRILTAG_AVAILABLE:
        return {
            "success": False,
            "message": "AprilTag library not available. Please install with: pip install apriltag",
            "detection": None,
            "image_base64": ""
        }

    # logging prefix for all messages from this function
    logging_prefix_str = "detect_april_tags:"

    try:
        # Create detector with optimized options for precision landing
        options = apriltag.DetectorOptions(
            families=tag_family,
            border=1,
            nthreads=4,
            quad_decimate=1.0,
            quad_blur=0.0,
            refine_edges=True,
            refine_decode=False,
            refine_pose=False,
            debug=False,
            quad_contours=True
        )
        detector = apriltag.Detector(options)

        # Convert to grayscale for detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect AprilTags
        detections = detector.detect(gray)

        # Process detections and find the one with lowest ID (or matching target_id)
        valid_detections = []
        all_detected_ids = []

        for detection in detections:
            tag_id = detection.tag_id
            all_detected_ids.append(int(tag_id))

            # Filter by target_id if specified
            if target_id != -1 and tag_id != target_id:
                continue  # Skip this detection if it doesn't match target_id

            center = detection.center
            corners = detection.corners
            corner_array = np.array(corners)
            width = np.max(corner_array[:, 0]) - np.min(corner_array[:, 0])
            height = np.max(corner_array[:, 1]) - np.min(corner_array[:, 1])
            diagonal = np.sqrt(width**2 + height**2)

            # Normalize diagonal by image diagonal for relative size
            image_diagonal = np.sqrt(image.shape[1]**2 + image.shape[0]**2)

            # Prevent division by zero
            if image_diagonal > 0:
                relative_size = diagonal / image_diagonal
            else:
                logger.warning(f"{logging_prefix_str} image diagonal is zero, setting relative size to 0")
                relative_size = 0.0

            # Store detection data
            detection_info = {
                "tag_id": int(tag_id),
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "corners": [[float(corner[0]), float(corner[1])] for corner in corners],
                "width": float(width),
                "height": float(height),
                "diagonal": float(diagonal),
                "relative_size": float(relative_size),
                "confidence": float(detection.decision_margin) if hasattr(detection, 'decision_margin') else 1.0
            }
            valid_detections.append(detection_info)

            # log detection details
            logger.debug(f"{logging_prefix_str} detected ID {tag_id}: center=({detection_info['center_x']}, {detection_info['center_y']})")

        # Find the tag with the lowest ID
        if valid_detections:
            lowest_id_detection = min(valid_detections, key=lambda x: x["tag_id"])
        else:
            lowest_id_detection = None

        # Handle image encoding based on parameters
        image_base64 = ""
        if include_augmented_image:
            # Create augmented image with detection box for the selected tag
            augmented_image = image.copy()
            if lowest_id_detection:
                corners = np.array(lowest_id_detection["corners"])
                corners_int = corners.astype(int)
                cv2.polylines(augmented_image, [corners_int], True, (0, 0, 255), 3)  # Red color in BGR

                # Draw center point
                center_int = (int(lowest_id_detection["center_x"]), int(lowest_id_detection["center_y"]))
                cv2.circle(augmented_image, center_int, 5, (0, 0, 255), -1)

            # Encode augmented image as base64
            _, buffer = cv2.imencode('.jpg', augmented_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Generate appropriate message
        if lowest_id_detection and all_detected_ids:
            message = f"Detected AprilTag with ID: {lowest_id_detection['tag_id']} (found IDs: {sorted(all_detected_ids)})"
        elif all_detected_ids:
            message = f"Found AprilTags {sorted(all_detected_ids)} but looking for ID: {target_id}"
        else:
            message = f"No AprilTags detected (looking for ID {target_id})"

        # log message
        logger.debug(f"{logging_prefix_str} {message}")

        # return success or failure
        success = lowest_id_detection is not None
        return {
            "success": success,
            "message": message,
            "detection": lowest_id_detection,
            "image_base64": image_base64
        }

    except Exception as e:
        logger.exception(f"Error during AprilTag detection: {str(e)}")
        return {
            "success": False,
            "message": f"AprilTag detection failed: {str(e)}",
            "detection": None,
            "image_base64": ""
        }

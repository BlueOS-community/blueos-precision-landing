#!/usr/bin/env python3

"""
Image Capture Module for Precision Landing

This module handles RTSP camera connection and frame capture functionality.

Methods include:

- `test_rtsp_connection`: Tests RTSP connection and captures a frame.


Based on the working implementation from https://github.com/mzahana/siyi_sdk
"""

import cv2
import base64
import logging
import threading
import time
import queue
from typing import Dict, Any

# Get logger
logger = logging.getLogger("precision-landing")

# Import AprilTag detection module with error handling
try:
    from app import april_tags
    APRIL_TAGS_AVAILABLE = True
except ImportError as e:
    logger.error(f"AprilTag detection module not available: {str(e)}")
    APRIL_TAGS_AVAILABLE = False
    april_tags = None

# video capture objects
video_capture = None
video_capture_rtsp_url = None
video_capture_mutex = threading.Lock()


# test RTSP connection using OpenCV with FFMPEG backend
# called from index.html's Test button
def test_rtsp_connection(rtsp_url: str) -> Dict[str, Any]:
    """
    Test RTSP connection and capture a frame with AprilTag detection.
    Returns connection status and basic stream information.
    """

    try:
        # capture single frame from RTSP stream
        frame_result = capture_frame_from_stream(rtsp_url)

        if not frame_result["success"]:
            return {
                "success": False,
                "message": f"RTSP connection failed: {frame_result['message']}",
                "error": frame_result.get("error", "Frame capture failed")
            }

        # Get frame data
        frame = frame_result["frame"]
        width = frame_result["width"]
        height = frame_result["height"]

        # Perform AprilTag detection with default settings and include augmented image
        if APRIL_TAGS_AVAILABLE and april_tags is not None:
            result = april_tags.detect_april_tags(
                frame,
                tag_family="tag36h11",
                target_id=-1,
                include_augmented_image=True
            )
            detection_result = {"success": result["success"], "message": result["message"], "detections": [result["detection"]] if result["detection"] else []}
            image_base64 = result["image_base64"]
        else:
            # Encode original frame as base64 if AprilTags not available
            _, buffer = cv2.imencode('.jpg', frame)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            detection_result = {
                "success": False,
                "message": "AprilTag detection module not available",
                "detections": []
            }

        return {
            "success": True,
            "message": f"RTSP connection successful ({width}x{height}). Method: {rtsp_url}",
            "connection_method": rtsp_url,
            "resolution": f"{width}x{height}",
            "image_base64": image_base64,
            "april_tag_detection": detection_result
        }

    except Exception as e:
        logger.exception(f"test_rtsp_connection: exception {str(e)}")
        return {
            "success": False,
            "message": f"Error testing RTSP connection: {str(e)}. Method: {rtsp_url}",
            "error": str(e)
        }


# captures a single frame from an RTSP stream
def capture_frame_from_stream(rtsp_url: str) -> Dict[str, Any]:
    """
    Capture a single frame from an RTSP stream.
    This is a simplified version focused on just getting one frame quickly.

    Args:
        rtsp_url: The RTSP URL to connect to

    Returns:
        Dictionary with success status and frame data (numpy array)
    """
    global rtsp_stream_reader

    # logging prefix for all messages from this function
    logging_prefix_str = "capture_frame_from_stream:"

    try:
        # Ensure video capture is thread-safe
        with video_capture_mutex:
            # Initialize reader if needed
            if not rtsp_stream_reader.start(rtsp_url):
                logger.error(f"{logging_prefix_str} failed to start RTSP stream {rtsp_url}")
                return {
                    "success": False,
                    "message": "Failed to start RTSP stream reader",
                    "error": "RTSP stream reader start failed"
                }

            # Wait for 0.1 seconds for a frame to be available
            time.sleep(0.1)

            # Get latest frame from the background thread
            ret, frame = rtsp_stream_reader.get_latest_frame()

        # check if frame was read successfully
        if not ret or frame is None:
            logger.error(f"{logging_prefix_str} no frame available from RTSP stream")
            return {
                "success": False,
                "message": "No frame available from video stream",
                "error": "Frame read failed"
            }

        # get dimensions and return frame
        height, width = frame.shape[:2]
        return {
            "success": True,
            "message": f"Frame captured successfully ({width}x{height})",
            "frame": frame,
            "resolution": f"{width}x{height}",
            "width": width,
            "height": height
        }

    except Exception as e:
        logger.exception(f"{logging_prefix_str} exception {str(e)}")
        return {
            "success": False,
            "message": f"Error capturing frame: {str(e)}",
            "error": str(e)
        }


# Clean up video capture resources
def cleanup_video_capture():
    """
    Clean up the video capture object and release resources.
    Should be called when stopping precision landing or when done.
    """
    global rtsp_stream_reader

    with video_capture_mutex:
        # Stop the RTSP stream reader
        rtsp_stream_reader.stop()
        logger.info("Cleaning up video capture")


# RTSPStreamReader class continuously reads frames from an RTSP stream, keeping only the latest frame
class RTSPStreamReader:
    def __init__(self):
        self.cap = None  # VideoCapture object
        self.rtsp_url = None  # Current RTSP URL
        self.q = queue.Queue(maxsize=1)  # Small queue for latest frames
        self.running = False
        self.thread = None
        self.logging_prefix_str = "RTSPStreamReader:"

    # start reading from the RTSP stream
    # returns True on success, False on failure
    def start(self, rtsp_url):

        # check if already running
        if self.running:

            # if the RTSP URL is the same, no need to restart
            if self.rtsp_url == rtsp_url:
                return True

            # if the RTSP URL has changed, stop the current reader
            self.stop()

        # initialise video capture with the new RTSP URL and timeout parameters
        self.cap = cv2.VideoCapture(rtsp_url,
                                    cv2.CAP_FFMPEG,
                                    [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                                     cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])

        if self.cap.isOpened():
            self.running = True
            self.rtsp_url = rtsp_url
            self.thread = threading.Thread(target=self._reader)
            self.thread.daemon = True
            self.thread.start()
            logger.info(f"{self.logging_prefix_str} started reader for {rtsp_url}")
            return True
        else:
            logger.error(f"{self.logging_prefix_str} failed to open RTSP stream: {rtsp_url}")
            self.cap.release()
            return False

    # stop reading from the RTSP stream
    def stop(self):
        if self.running:
            logger.info(f"{self.logging_prefix_str} stopped")
            self.running = False
            if self.thread:
                self.thread.join(timeout=2.0)  # Don't wait forever
            if self.cap:
                self.cap.release()
                self.cap = None
            self.rtsp_url = None

    # get the latest frame from the queue
    def get_latest_frame(self):
        try:
            return True, self.q.get_nowait()
        except queue.Empty:
            return False, None

    # background thread that reads frames from the RTSP stream
    def _reader(self):
        logger.info(f"{self.logging_prefix_str} thread started")
        first_frame = True
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    # Keep only the latest frame - remove old frame if queue is not empty
                    if not self.q.empty():
                        try:
                            self.q.get_nowait()  # Remove old frame
                        except queue.Empty:
                            pass
                    try:
                        self.q.put_nowait(frame)
                        if first_frame:
                            logger.debug(f"{self.logging_prefix_str} started receiving frames")
                            first_frame = False
                    except queue.Full:
                        # Queue is full, skip this frame
                        logger.warning(f"{self.logging_prefix_str} queue full, skipping")
                        pass
                elif not self.running:
                    # Normal shutdown
                    break
                else:
                    # Failed to read frame, short sleep and retry
                    time.sleep(0.01)
            except Exception as e:
                if self.running:  # Only log if we're supposed to be running
                    logger.error(f"{self.logging_prefix_str} error in reader thread: {e}")
                    time.sleep(0.1)  # Short delay before retry

        logger.info(f"{self.logging_prefix_str} thread stopped")


# Global stream reader
rtsp_stream_reader = RTSPStreamReader()

#!/usr/bin/env python3

# Precision Landing Python backend
# Implements these features required by the index.html frontend:
# - Save camera settings including type and RTSP URL
# - Get camera settings including last used settings
# - Save/get precision landing enabled state (persistent across restarts)
# - "Test" button to view the live video and capture the april tag location
# - "Run" button to enable the precision landing including sending MAVLink messages to the vehicle
# - Status endpoint to check if precision landing is currently running

import logging.handlers
import sys
import asyncio
import time
import math
from math import tan, atan, radians, degrees
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi import Query
from typing import Dict, Any

# Import the local modules
from app import settings
from app import image_capture
from app import mavlink_interface
from app import optflow_lkt

# Configure console logging
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Create logger
logger = logging.getLogger("precision-landing")
logger.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

app = FastAPI()

# Global exception handler to ensure all errors return JSON
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception in {request.url}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": f"Internal server error: {str(exc)}",
            "error": "Internal server error"
        }
    )

# Global variables
precision_landing_running = False  # True if precision landing is currently running
gimbal_down_q = [0.7071, 0, -0.7071, 0]  # Quaternion used to check for downward facing gimbal

# log that the backend has started
logger.info("Precision Landing backend started")


# Auto-start precision landing if it was previously enabled
async def startup_auto_restart():
    """Check if precision landing was previously enabled and auto-restart if needed"""
    try:
        enabled = settings.get_precision_landing_enabled()
        if enabled:
            logger.info("Auto-restarting precision landing")
            # Get last used settings for auto-restart
            last_used = settings.get_last_used()
            camera_type = last_used.get("camera_type", "siyi-a8")
            rtsp_url = last_used.get("rtsp", "rtsp://192.168.144.25:8554/main.264")
            # Don't await this so startup doesn't block
            asyncio.create_task(start_precision_landing_internal(camera_type, rtsp_url))
    except Exception as e:
        logger.error(f"Error during auto-restart: {str(e)}")


# Internal function to start precision landing
async def start_precision_landing_internal(camera_type: str, rtsp_url: str):
    """Internal function to start the precision landing process"""
    global precision_landing_running

    # logging prefix for all messages from this function
    logging_prefix_str = "Precision landing:"

    try:
        logger.info(f"{logging_prefix_str} started")
        precision_landing_running = True

        # Get camera setting
        camera_hfov_deg = settings.get_camera_horizontal_fov(camera_type)

        # Get AprilTag target ID from settings
        target_apriltag_id = settings.get_apriltag_target_id()

        # Get MAVLink target system ID from settings
        target_system_id = settings.get_mavlink_sysid()

        # Get gimbal attitude usage setting
        use_gimbal_attitude = settings.get_gimbal_attitude_settings()

        # log settings used
        tag_id_str = str(target_apriltag_id)
        if target_apriltag_id == -1:
            tag_id_str = "any"
        logger.info(f"{logging_prefix_str} Camera type: {camera_type}, HFOV: {camera_hfov_deg}, RTSP:{rtsp_url}, TagId:{tag_id_str}, SysId:{target_system_id}, UseGimbalAttitude:{use_gimbal_attitude}")

        # Test MAV2Rest connection by sending a test LANDING_TARGET message
        mav_test = mavlink_interface.send_landing_target_msg(
            angle_x=0.0,
            angle_y=0.0,
            distance=0.0,
            size_x=0.0,
            size_y=0.0,
            target_num=0,
            sysid=target_system_id
        )
        if not mav_test["success"]:
            logger.error(f"{logging_prefix_str} MAV2Rest connection test failed: {mav_test['message']}")
            precision_landing_running = False
            return

        # get gimbal attitude, if not available request it
        gimbal_result = mavlink_interface.get_gimbal_attitude(target_system_id)
        if not gimbal_result["success"]:
            # request gimbal attitude status at 1hz
            mavlink_interface.request_gimbal_attitude_status(target_system_id, 1)
            logger.info(f"{logging_prefix_str} Requested GIMBAL_DEVICE_ATTITUDE_STATUS at 1hz")

        # Precision landing main loop
        last_frame_time = time.time()
        last_log_time = last_frame_time
        last_send_time = last_frame_time
        frame_count = 0
        sent_count = 0

        while precision_landing_running:
            try:
                # Capture frame from RTSP stream
                frame_result = image_capture.capture_frame_from_stream(rtsp_url)
                capture_time = time.time()

                if not frame_result["success"]:
                    logger.warning(f"{logging_prefix_str} failed to capture frame: {frame_result['message']}")

                    # if no frames captured in 10 seconds, restart the capture
                    if capture_time - last_frame_time > 10:
                        logger.error(f"{logging_prefix_str} No frames captured in 10 seconds, restarting capture")
                        image_capture.cleanup_video_capture()
                        last_frame_time = capture_time

                    # Wait 0.01 seconds before retrying
                    await asyncio.sleep(0.01)
                    continue

                # record success
                last_frame_time = capture_time
                frame_count += 1

                # Check if we should use gimbal attitude and if gimbal is facing downward
                # Defaults to sending flow values if gimbal attitude is unavailable
                should_send_target = True
                if use_gimbal_attitude:
                    gimbal_result = mavlink_interface.get_gimbal_attitude(target_system_id)
                    if gimbal_result["success"]:
                        # Use quaternion to check if gimbal is facing downward within 10 degrees
                        gimbal_attitude_dict = gimbal_result["quaternion"]
                        # Convert dictionary format to array format [w, x, y, z]
                        gimbal_attitude_q = [
                            gimbal_attitude_dict["w"],
                            gimbal_attitude_dict["x"],
                            gimbal_attitude_dict["y"],
                            gimbal_attitude_dict["z"]
                        ]
                        angle_diff_rad = angle_between_quaternions(gimbal_down_q, gimbal_attitude_q)
                        angle_diff_deg = degrees(angle_diff_rad)
                        logger.debug(f"{logging_prefix_str} Gimbal attitude angle_diff:{angle_diff_deg:.1f}°")
                        if angle_diff_rad > radians(10):
                            should_send_target = False
                            logger.debug(f"{logging_prefix_str} Gimbal attitude angle_diff:{angle_diff_deg:.1f}° > 10°, skipping target")
                    else:
                        # If we can't get gimbal attitude but it's required, don't send target
                        logger.warning(f"{logging_prefix_str} gimbal attitude unavailable, sending anyway")

                # if not sending target sleep for 1 second
                if not should_send_target:
                    await asyncio.sleep(1)
                    continue

                # Get the captured frame
                frame = frame_result["frame"]
                width = frame_result["width"]
                height = frame_result["height"]

                # Perform optical flow calculation
                opticalflow_result = optflow_lkt.get_optical_flow(frame, capture_time, False)

                # calculate hfov
                camera_vfov_deg = calculate_vertical_fov(camera_hfov_deg, width, height)

                if opticalflow_result.get("success"):
                    # Extract flow values
                    flow_x = opticalflow_result.get("flow_x", 0.0)
                    flow_y = opticalflow_result.get("flow_y", 0.0)
                    dt = opticalflow_result.get("dt", 0.0)

                    # Convert pixel flow to angular flow (radians/second)
                    # Assuming a simple conversion based on camera FOV and frame rate
                    pixels_per_radian_h = width / math.radians(camera_hfov_deg)
                    pixels_per_radian_v = height / math.radians(camera_vfov_deg)

                    if dt > 0:
                        flow_rate_x = flow_x / (pixels_per_radian_h * dt)  # rad/s
                        flow_rate_y = flow_y / (pixels_per_radian_v * dt)  # rad/s
                    else:
                        # Avoid division by zero
                        flow_rate_x = 0.0
                        flow_rate_y = 0.0

                    logger.debug(f"{logging_prefix_str} Frame:{frame_count} sent OPTICAL_FLOW "
                                    f"(flow_x={flow_rate_x:.4f} rad/s, flow_y={flow_rate_y:.4f} rad/s) dt={dt:.4f}s")

                    # Send OPTICAL_FLOW message
                    send_result = mavlink_interface.send_optical_flow_msg(
                        sysid=target_system_id,  # system ID of the target vehicle
                        flow_x=0.0,            # flow_x as integer (not used)
                        flow_y=0.0,            # flow_y as integer (not used)
                        flow_comp_m_x=0.0,     # flow_comp_m_x (gimbaled camera so always zero)
                        flow_comp_m_y=0.0,     # flow_comp_m_y (gimbaled camera so always zero)
                        quality=255,           # quality (0 to 255, 255 means good quality)
                        ground_distance=-1,    # ground distance is unknown
                        flow_rate_x=flow_rate_x,  # flow rate in x direction (rad/s)
                        flow_rate_y=flow_rate_y  # flow rate in y direction (rad/s)
                    )

                    if send_result["success"]:
                        sent_count += 1
                        logger.debug(f"{logging_prefix_str} Frame:{frame_count} sent OPTICAL_FLOW "
                                     f"(flow_x={flow_rate_x:.4f} rad/s, flow_y={flow_rate_y:.4f} rad/s)")
                    else:
                        logger.warning(f"{logging_prefix_str} Failed to send OPTICAL_FLOW: {send_result['message']}")

                    # record last send time
                    last_send_time = time.time()
                else:
                    logger.debug(f"{logging_prefix_str} Optical flow calculation failed: {opticalflow_result.get('message', 'Unknown error')}")

                # log every 10 seconds
                current_time = time.time()
                if current_time - last_log_time > 10:
                    update_rate_hz = frame_count / (current_time - last_log_time)
                    logger.info(f"{logging_prefix_str} rate:{update_rate_hz:.1f}hz frames:{frame_count} MsgsSent:{sent_count}")
                    last_log_time = current_time
                    frame_count = 0
                    sent_count = 0

                # sleep to reduce CPU load
                await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"{logging_prefix_str} loop error {str(e)}")
                await asyncio.sleep(1)  # Wait 1 second before retrying

    except Exception as e:
        logger.error(f"{logging_prefix_str} error {str(e)}")
    finally:
        # Clean up video capture when stopping
        image_capture.cleanup_video_capture()
        precision_landing_running = False
        logger.info(f"{logging_prefix_str} stopped")


# helper function to calculate the vertical FOV in degrees based on the horizontal FOV, image width, and height (in pixels)
def calculate_vertical_fov(hfov_deg: float, width: int, height: int) -> float:
    """Calculate vertical FOV based on horizontal FOV, image width, and height
       tan(vfov/2) = tan(hfov/2) * (height/width)
    """

    # logging prefix for all messages from this function
    logging_prefix_str = "calculate_vertical_fov:"

    # Validate inputs to prevent mathematical errors
    if width <= 0:
        logger.error(f"{logging_prefix_str} invalid image width: {width}")
        return 0.0

    if height <= 0:
        logger.error(f"{logging_prefix_str} invalid image height: {height}")
        return 0.0

    if hfov_deg <= 0 or hfov_deg >= 180:
        logger.error(f"{logging_prefix_str} invalid horizontal FOV: {hfov_deg} (must be between 0 and 180)")
        return 0.0

    try:
        # Convert horizontal FOV from degrees to radians
        hfov_rad = radians(hfov_deg)

        # Calculate aspect ratio
        aspect_ratio = height / width

        # Use trigonometric relationship to calculate vertical FOV
        vfov_rad = 2 * atan(tan(hfov_rad / 2) * aspect_ratio)

        # Convert back to degrees
        vfov_deg = degrees(vfov_rad)

        # Sanity check result
        if vfov_deg <= 0 or vfov_deg >= 180:
            logger.error(f"{logging_prefix_str} calculated invalid VFOV: {vfov_deg}")
            return 0.0

        return vfov_deg
    except (ValueError, OverflowError) as e:
        logger.error(f"{logging_prefix_str} mathematical error calculating VFOV: {e}")
        return 0.0

# calculate the angle in radians between two quaternions
def angle_between_quaternions(q1, q2):
    # Ensure both quaternions are unit length
    dot = abs(q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3])
    dot = min(1.0, max(-1.0, dot))  # Clamp for acos
    return 2 * math.acos(dot)  # in radians

# Precision Landing API Endpoints

# Load precision landing settings
@app.post("/precision-landing/get-settings")
async def get_precision_landing_settings() -> Dict[str, Any]:
    """Get saved camera settings"""
    logger.debug("Getting precision landing settings")

    try:
        # Get the last used camera settings
        last_used = settings.get_last_used()

        # Get RTSP URLs and FOV values for all camera types
        cameras = {}
        for camera_type in ["siyi-a8", "siyi-zr10", "siyi-zt6-ir", "siyi-zt6-rgb"]:
            rtsp_url = settings.get_camera_rtsp(camera_type)
            horizontal_fov = settings.get_camera_horizontal_fov(camera_type)
            cameras[camera_type] = {
                "rtsp": rtsp_url,
                "horizontal_fov": horizontal_fov
            }

        # Get AprilTag settings
        apriltag_settings = {
            "family": settings.get_apriltag_family(),
            "target_id": settings.get_apriltag_target_id()
        }

        # Get MAVLink settings
        mavlink_settings = {
            "flight_controller_sysid": settings.get_mavlink_sysid()
        }

        # Get gimbal settings
        gimbal_settings = {
            "use_gimbal_attitude": settings.get_gimbal_attitude_settings()
        }

        return {
            "success": True,
            "last_used": last_used,
            "cameras": cameras,
            "apriltag": apriltag_settings,
            "mavlink": mavlink_settings,
            "gimbal": gimbal_settings
        }
    except Exception as e:
        logger.exception(f"Error getting precision landing settings: {str(e)}")
        return {"success": False, "message": f"Error: {str(e)}"}


# Save precision landing settings
@app.post("/precision-landing/save-settings")
async def save_precision_landing_settings(
    type: str = Query(...),
    rtsp: str = Query(...),
    fov: float = Query(...),
    apriltag_family: str = Query(...),
    tag_id: int = Query(...),
    flight_controller_sysid: int = Query(...),  # Keep this for HTML compatibility
    use_gimbal_attitude: bool = Query(True)     # Default to True
) -> Dict[str, Any]:
    """Save camera settings and other precision landing settings to persistent storage (using query parameters)"""
    # Map flight_controller_sysid to sysid for internal use
    sysid = flight_controller_sysid
    logger.info(f"Saving precision landing settings: camera_type={type}, rtsp_url={rtsp}, fov={fov}, "
                f"apriltag_family={apriltag_family}, tag_id={tag_id}, sysid={sysid}, "
                f"use_gimbal_attitude={use_gimbal_attitude}")

    # Save camera settings
    camera_success = settings.update_camera_settings(type, rtsp, fov)

    # Save AprilTag settings
    apriltag_success = settings.update_apriltag_settings(apriltag_family, tag_id)

    # Save MAVLink settings
    mavlink_success = settings.update_mavlink_sysid(sysid)

    # Save gimbal attitude settings
    gimbal_success = settings.update_gimbal_attitude_settings(use_gimbal_attitude)

    if camera_success and apriltag_success and mavlink_success and gimbal_success:
        return {"success": True, "message": f"Settings saved for {type}"}
    else:
        return {"success": False, "message": "Failed to save some settings"}


# Get precision landing enabled state
@app.get("/precision-landing/get-enabled-state")
async def get_precision_landing_enabled_state() -> Dict[str, Any]:
    """Get saved precision landing enabled state (supports both GET and POST)"""
    logger.debug("Getting precision landing enabled state")

    try:
        enabled = settings.get_precision_landing_enabled()
        return {
            "success": True,
            "enabled": enabled
        }
    except Exception as e:
        logger.exception(f"Error getting precision landing enabled state: {str(e)}")
        return {"success": False, "message": f"Error: {str(e)}", "enabled": False}


# Save precision landing enabled state
@app.post("/precision-landing/save-enabled-state")
async def save_precision_landing_enabled_state(enabled: bool = Query(...)) -> Dict[str, Any]:
    """Save precision landing enabled state to persistent storage (using query parameter)"""
    logger.info(f"Precision landing enabled state: {enabled}")
    success = settings.update_precision_landing_enabled(enabled)

    if success:
        return {"success": True, "message": f"Enabled state saved: {enabled}"}
    else:
        return {"success": False, "message": "Failed to save enabled state"}


# Test image retrieval from the RTSP stream and AprilTag detection
@app.post("/precision-landing/test")
async def test_precision_landing(type: str = Query(...), rtsp: str = Query(...)) -> Dict[str, Any]:
    """Test precision landing functionality with RTSP connection"""
    logger.info(f"Testing with camera_type={type}, rtsp={rtsp}")

    try:
        # Run the RTSP connection test in a thread to avoid blocking
        def run_test():
            return image_capture.test_rtsp_connection(rtsp)

        # Run the test in an executor to avoid blocking the async loop
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_test)
            result = future.result(timeout=60)  # 60 second timeout should be sufficient

        if result["success"]:
            logger.info(f"RTSP test successful for {type}: {result['message']}")
            # Add camera type to the response
            result["camera_type"] = type
            result["rtsp_url"] = rtsp
        else:
            logger.warning(f"RTSP test failed for {type}: {result['message']}")

        return result

    except concurrent.futures.TimeoutError:
        logger.error(f"RTSP test timed out for {type} camera")
        return {
            "success": False,
            "message": "Test timed out - unable to connect to camera within 60 seconds",
            "error": "Connection timeout"
        }
    except Exception as e:
        logger.exception(f"Error during precision landing test: {str(e)}")
        return {"success": False, "message": f"Test failed: {str(e)}"}


# Get precision landing running status
@app.get("/precision-landing/status")
async def get_precision_landing_status() -> Dict[str, Any]:
    """Get precision landing running status"""
    logger.debug("Getting precision landing status")

    try:
        return {
            "success": True,
            "running": precision_landing_running,
            "message": "Running" if precision_landing_running else "Stopped"
        }
    except Exception as e:
        logger.exception(f"Error getting precision landing status: {str(e)}")
        return {"success": False, "message": f"Error: {str(e)}", "running": False}


# Start precision landing (this is called by the frontend's "Run" button)
@app.post("/precision-landing/start")
async def start_precision_landing(type: str = Query(...), rtsp: str = Query(...)) -> Dict[str, Any]:
    """Start precision landing"""
    global precision_landing_running

    logger.info(f"Start precision landing request received for type={type}, rtsp={rtsp}")

    try:
        if precision_landing_running:
            return {"success": False, "message": "Precision landing is already running"}

        # Start the precision landing process
        asyncio.create_task(start_precision_landing_internal(type, rtsp))

        return {
            "success": True,
            "message": f"Precision landing started successfully with {type} camera"
        }
    except Exception as e:
        logger.exception(f"Error starting precision landing: {str(e)}")
        return {"success": False, "message": f"Failed to start: {str(e)}"}


# Stop precision landing (this is called by the frontend's "Stop" button)
@app.post("/precision-landing/stop")
async def stop_precision_landing() -> Dict[str, Any]:
    """Stop precision landing"""
    global precision_landing_running

    logger.info("Stop precision landing request received")

    try:
        # Stop the precision landing process
        precision_landing_running = False

        # Clean up video capture when manually stopping
        image_capture.cleanup_video_capture()

        return {
            "success": True,
            "message": "Precision landing stopped successfully"
        }
    except Exception as e:
        logger.exception(f"Error stopping precision landing: {str(e)}")
        return {"success": False, "message": f"Failed to stop: {str(e)}"}


# Test MAV2Rest MAVLink connection
@app.post("/precision-landing/test-mavlink")
async def test_mavlink_connection() -> Dict[str, Any]:
    """Test MAV2Rest MAVLink connection"""
    try:
        target_system_id = settings.get_mavlink_sysid()
        use_gimbal_attitude = settings.get_gimbal_attitude_settings()

        # Test LANDING_TARGET message sending
        result = mavlink_interface.send_landing_target_msg(
            angle_x=0.0,
            angle_y=0.0,
            distance=0.0,
            size_x=0.0,
            size_y=0.0,
            target_num=0,
            sysid=target_system_id
        )

        # get gimbal attitude, if not available request it
        gimbal_result = mavlink_interface.get_gimbal_attitude(target_system_id)
        if not gimbal_result["success"]:
            # request gimbal attitude status at 1hz
            mavlink_interface.request_gimbal_attitude_status(target_system_id, 1)
            logger.info("Requested GIMBAL_DEVICE_ATTITUDE_STATUS at 1hz")
            # wait for 1 second and try again
            await asyncio.sleep(1)
            gimbal_result = mavlink_interface.get_gimbal_attitude(target_system_id)

        # Prepare response with both tests
        response = {
            "landing_target_test": result,
            "gimbal_attitude_test": gimbal_result
        }

        if result["success"]:
            logger.info("LANDING_TARGET test successful")
            base_message = f"LANDING_TARGET sent to SysID {target_system_id}"

            if gimbal_result["success"]:
                logger.info("Gimbal attitude retrieval successful")
                # Calculate gimbal attitude difference from downward
                gimbal_attitude_dict = gimbal_result["quaternion"]
                gimbal_attitude_q = [
                    gimbal_attitude_dict["w"],
                    gimbal_attitude_dict["x"],
                    gimbal_attitude_dict["y"],
                    gimbal_attitude_dict["z"]
                ]
                angle_diff_rad = angle_between_quaternions(gimbal_down_q, gimbal_attitude_q)
                angle_diff_deg = degrees(angle_diff_rad)
                gimbal_info = f"Gimbal attitude q0:{gimbal_attitude_q[0]:.4f}, q1:{gimbal_attitude_q[1]:.4f}, "
                gimbal_info += f"{gimbal_attitude_q[2]:.5f}, q3:{gimbal_attitude_q[3]:.4f}"
                gimbal_info += f"\nGimbal angle from downward:{angle_diff_deg:.1f} deg"
                if not use_gimbal_attitude or angle_diff_deg <= 10.0:
                    gimbal_info += " targets will be sent"
                else:
                    gimbal_info += " targets will NOT be sent"

                response.update({
                    "success": True,
                    "message": base_message + "\n" + gimbal_info
                })
            else:
                logger.warning(f"Gimbal attitude unavailable")
                gimbal_warning = f"Gimbal attitude unavailable"
                response.update({
                    "success": True,  # Still success since LANDING_TARGET worked
                    "message": base_message + "\n" + gimbal_warning
                })
        else:
            logger.warning(f"MAV2Rest LANDING_TARGET test failed: {result['message']}")
            response.update({
                "success": False,
                "message": f"MAV2Rest connected but test message failed: {result['message']}"
            })

        return response

    except Exception as e:
        logger.exception(f"Error during MAV2Rest test: {str(e)}")
        return {"success": False, "message": f"MAV2Rest test failed: {str(e)}"}


# Initialize auto-restart task
@app.on_event("startup")
async def on_startup():
    """Application startup event handler"""
    await startup_auto_restart()


# Mount static files AFTER defining API routes
# Use absolute path to handle Docker container environment
static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

# Set up logging for the app
log_dir = Path('/app/logs')
log_dir.mkdir(parents=True, exist_ok=True)
fh = logging.handlers.RotatingFileHandler(log_dir / 'lumber.log', maxBytes=2**16, backupCount=1)
logger.addHandler(fh)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

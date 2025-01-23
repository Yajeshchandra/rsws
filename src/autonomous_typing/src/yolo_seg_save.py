#!/usr/bin/env python3

# Import required libraries
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from tf2_geometry_msgs import do_transform_point
import tf2_ros
import cv2
import numpy as np
from ultralytics import YOLO
import json


class YoloPixelSegmentationNode:
    """
    A ROS node for pixel segmentation using YOLO model.
    This node processes images from a ROS topic, detects objects, and performs transformations.
    """

    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('yolo_pixel_segmentation_node', anonymous=True)

        # Parameters
        self.input_topic = rospy.get_param('~input_topic', '/camera_gripper/image_raw')
        self.camera_info_topic = rospy.get_param('~camera_info_topic', '/camera_gripper/camera_info')
        self.output_topic = rospy.get_param('~output_topic', '/camera_gripper/processed_image')
        self.class_to_detect = rospy.get_param('~class_to_detect', 66)
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_link2')
        self.base_frame = rospy.get_param('~base_frame', 'base_link')

        # Real-world keyboard dimensions (in millimeters)
        self.KEYBOARD_LENGTH = 354.076
        self.KEYBOARD_WIDTH = 123.444

        # Initialize YOLO model for segmentation
        self.model = YOLO('yolov8n-seg.pt')

        # Camera parameters (updated from CameraInfo callback)
        self.camera_matrix = None
        self.dist_coeffs = None

        # Initialize ROS communication
        self.bridge = CvBridge()
        self.subscriber = rospy.Subscriber(self.input_topic, Image, self.image_callback)
        self.camera_info_sub = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback)
        self.publisher = rospy.Publisher(self.output_topic, Image, queue_size=10)

        # TF Buffer and Listener for transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Load keyboard layout from JSON file
        with open('keyboard_layout.json', 'r') as f:
            self.keyboard_points_dict = json.load(f)

        rospy.loginfo("YoloPixelSegmentationNode initialized successfully.")

    def camera_info_callback(self, msg):
        """
        Callback function to update camera parameters from the CameraInfo topic.
        """
        self.camera_matrix = np.array(msg.K).reshape(3, 3)  # Camera intrinsic matrix
        self.dist_coeffs = np.array(msg.D)  # Distortion coefficients

    def get_transform(self, camera_point, camera_frame="camera_link1"):
        """
        Get the transformation of a point from the camera frame to the world frame.

        :param camera_point: Point in the camera frame.
        :param camera_frame: Name of the camera frame.
        :return: Transformed point in the world frame or None if failed.
        """
        try:
            # Wait for the transformation to be available
            transform = self.tf_buffer.lookup_transform(
                "world", camera_frame, rospy.Time(0), rospy.Duration(1.0)
            )
            # Transform the point
            world_point = do_transform_point(camera_point, transform)
            return world_point
        except tf2_ros.LookupException as e:
            rospy.logerr(f"Transform lookup failed: {e}")
            return None

    def process_image(self, cv_image, results):
        """
        Process the input image, detect objects, and transform detected points.

        :param cv_image: Input OpenCV image.
        :param results: YOLO detection results.
        :return: Processed image with visualizations.
        """
        # Create a copy of the input image for visualization
        vis_image = cv_image.copy()

        if len(results) == 0:  # No results detected
            return vis_image

        # Use the first result from YOLO
        result = results[0]
        if len(result.boxes) == 0:  # No bounding boxes detected
            return vis_image

        # Get the camera-to-base transformation matrix
        T_base_camera = self.get_transform()
        if T_base_camera is None:  # Transformation failed
            return vis_image

        # Loop through detected bounding boxes
        for idx in range(len(result.boxes)):
            class_id = int(result.boxes.cls[idx].item())  # Class ID of the object
            if class_id == self.class_to_detect:  # Check if detected class matches the target class
                # Get bounding box coordinates
                box = result.boxes.xyxy[idx].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)

                # Scale keyboard layout points to the bounding box dimensions
                keyboard_points = np.array(list(self.keyboard_points_dict.values()))
                box_length = x2 - x1
                box_width = y2 - y1
                scaled_points = (
                    (keyboard_points / np.array([self.KEYBOARD_LENGTH, self.KEYBOARD_WIDTH]))
                    * np.array([box_length, box_width])
                    + np.array([x1, y1])
                )

                if self.camera_matrix is not None:  # If camera parameters are available
                    try:
                        # Create 3D model points
                        model_points = np.zeros((len(keyboard_points), 3))
                        model_points[:, 0] = keyboard_points[:, 0]
                        model_points[:, 1] = keyboard_points[:, 1]

                        # Estimate pose using SolvePnP
                        success, rvec, tvec = cv2.solvePnP(
                            model_points, scaled_points,
                            self.camera_matrix, self.dist_coeffs,
                            flags=cv2.SOLVEPNP_ITERATIVE
                        )
                        if success:
                            # Draw axes and save projected points
                            self.draw_axes(vis_image, rvec, tvec, self.camera_matrix, self.dist_coeffs)
                            projected_points, _ = cv2.projectPoints(
                                model_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
                            )
                            projected_points = projected_points.reshape(-1, 2)

                            # Save points for further processing
                            points_in_base_frame = []  # Transformation logic can be added here
                            np.save('projected_points.npy', points_in_base_frame)

                            for point in points_in_base_frame:
                                rospy.loginfo(f"Point in base frame: {point}")
                    except Exception as e:
                        rospy.logwarn(f"PnP estimation failed: {e}")

        return vis_image

    def image_callback(self, msg):
        """
        Callback function to process the incoming image message from the ROS topic.
        """
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv_image = cv2.resize(cv_image, (640, 640))  # Resize image for YOLO

            # Perform object detection
            results = self.model(cv_image)

            # Process the detected results
            processed_image = self.process_image(cv_image, results)

            # Convert processed image back to ROS format and publish
            ros_image = self.bridge.cv2_to_imgmsg(processed_image, encoding="bgr8")
            self.publisher.publish(ros_image)
        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")


if __name__ == '__main__':
    try:
        node = YoloPixelSegmentationNode()
        rospy.spin()  # Keep the node running
    except rospy.ROSInterruptException:
        pass

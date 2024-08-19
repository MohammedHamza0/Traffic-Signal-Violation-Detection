# Traffic Signal Violation Detection System

## Overview

This project is an AI-based Traffic Signal Violation Detection System that uses the YOLOv8 object detection model and OpenCV to detect vehicles violating traffic signals in real-time. The system monitors traffic lights, identifies active red lights, and flags vehicles that cross a designated region of interest (ROI) during a red signal.

## Features

- **Real-time Object Detection**: Detects vehicles such as cars, bicycles, motorcycles, buses, and trucks using the YOLOv8 model.
- **Traffic Light Monitoring**: Analyzes the brightness of traffic lights to determine whether the red light is active.
- **Violation Detection**: Identifies and highlights vehicles that cross the ROI during a red light, flagging them as traffic violators.
- **Visual Feedback**: Displays visual cues on the video feed, including bounding boxes, polygons, and text overlays, to indicate the status of vehicles and traffic signals.
- **Configurable Confidence Threshold**: Adjusts the confidence level for object detection, allowing for fine-tuning of the system's accuracy.

## Installation

### Prerequisites

- Python 3.7+
- OpenCV
- Ultralytics YOLOv8
- NumPy

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/MohammedHamza0/traffic-signal-violation-detection.git
   cd traffic-signal-violation-detection
   ```

2. **Install Dependencies**

   Install the required Python packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file include:
   ```text
   opencv-python
   ultralytics
   numpy
   ```

3. **Download YOLOv8 Model**

   Download the pre-trained YOLOv8 model (`yolov8m.pt`) and place it in the project directory.

## Usage

1. **Prepare the Video**

   Ensure that your video file (`tr.mp4`) is placed in the project directory.

2. **Run the Script**

   Execute the detection script by running:

   ```bash
   python traffic_violation_detection.py
   ```

3. **Monitor the Output**

   The video will be displayed with detected objects, highlighted ROIs, and any violations flagged in real-time. Press `Esc` to exit the video display.

## Project Structure

- **`traffic_violation_detection.py`**: The main script for running the violation detection system.
- **`requirements.txt`**: A list of Python dependencies required for the project.
- **`yolov8m.pt`**: The pre-trained YOLOv8 model (to be downloaded separately).
- **`tr.mp4`**: Sample video file .

## Customization

### Adjusting the ROI and Traffic Lights

Modify the following sections in the script to change the coordinates of the ROI and traffic light polygons:

```python
RedLight = np.array([[998, 125],[998, 155],[972, 152],[970, 127]])
GreenLight = np.array([[971, 200],[996, 200],[1001, 228],[971, 230]])
ROI = np.array([[910, 372],[388, 365],[338, 428],[917, 441]])
```

### Changing the Target Labels

You can specify which objects to detect by modifying the `TargetLabels` list:

```python
TargetLabels = ["bicycle", "car", "motorcycle", "bus", "truck", "traffic light"]
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.


## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/yolov8)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)

## Contact

For any inquiries or further information, please contact Mohammed Hamza Moawad Khalifa  [LinkedIn](https://www.linkedin.com/in/mohammed-hamza-moawad-khalifa/).

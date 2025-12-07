## 🚦 CNN-based Road Traffic Sign and Light Recognition

This repository contains a Convolutional Neural Network (CNN) model developed for the real-time **detection and localization** of road traffic signs and lights. This system is a foundational component for applications in autonomous driving and intelligent traffic monitoring, aiming to enhance both safety and decision-making capabilities in automated vehicles.

The core of this project employs **Transfer Learning** using the **ResNet50** architecture, which is adapted to perform **Bounding Box Regression**—predicting the precise location of objects in an image.

***

### ✨ Key Technical Features and Architecture

* **Transfer Learning with ResNet50:** The model utilizes the pre-trained weights of the **ResNet50** (Residual Network) model from Keras Applications. ResNet50 acts as a highly effective **feature extractor**, leveraging deep, complex patterns learned from the vast ImageNet dataset. The base network is integrated, and a custom head is added on top. * **Bounding Box Regression:** The task is formulated as a **regression problem**. The custom output head of the CNN is designed to predict four continuous, normalized floating-point values `[x_min, y_min, x_max, y_max]` corresponding to the coordinates of the object's bounding box.
* **Preprocessing Pipeline:** Includes robust functions for loading and processing images and corresponding text-based label files, ensuring data is ready for the CNN input layers.

***

### 🛠️ Prerequisites and Dependencies

To run the Jupyter Notebook (`CNN-based-Road-Traffic-Sign-and-Light-Recognition.ipynb`), you need a Python environment with the following libraries:

* **Deep Learning:** `tensorflow` / `keras` (specifically for `ResNet50` and model layers)
* **Data Processing:** `numpy`, `sklearn` (for `train_test_split`)
* **Image Processing:** `opencv-python` (`cv2`)
* **Visualization:** `matplotlib`

***

### 📊 Dataset and Data Preparation

The model is trained on a car detection dataset. The data is organized into `images` (JPEG/PNG) and `labels` (TXT).

**Preprocessing Details:**

1.  **Image Resizing and Normalization:** Images are loaded, resized to a consistent **(224, 224)** pixel size, and normalized by dividing the pixel values by 255.0. This scales the input to the range **[0, 1]**, which is standard practice for CNNs.
2.  **Label Loading:** The labels are assumed to be in a format similar to **YOLO** (You Only Look Once), typically containing `[class_id, x_min, y_min, x_max, y_max]` values. The loading function extracts the four normalized bounding box coordinates, skipping the class ID.
3.  **Data Split:** The loaded data (`X` for images, `y` for bounding boxes) is split into training, validation, and test sets using `train_test_split`.

***

### 🧠 Model Compilation

The model is compiled with the following settings, which are appropriate for a regression-based object localization task:

* **Optimizer:** **Adam**.
* **Loss Function:** **Mean Squared Error (`'mse'`)**. As the model predicts continuous coordinate values (a regression task), MSE is the standard loss function for measuring the average squared difference between the predicted box and the ground-truth box.

***

### 💡 Future Works and Enhancements

1.  **Integrate Classification Head:**
    * The current model is configured for **localization (bounding box regression)** only. To achieve "Recognition" (as per the title), a second output head must be added for **classification** (e.g., classifying *what* was detected: "stop sign," "red light," "speed limit 50," etc.). This would transform the model into a **multi-task network**.

2.  **Implement IoU (Intersection over Union) as a Metric:**
    * **IoU** is the definitive metric for object detection, measuring the degree of overlap between the predicted and ground-truth bounding boxes. A custom IoU metric should be implemented in Keras to evaluate localization performance more accurately than simple MSE loss.

3.  **Explore Advanced Object Detection Frameworks:**
    * For production-ready, real-time performance, investigate and migrate the solution to specialized object detection frameworks like **YOLOv8** or **EfficientDet**. These models offer superior speed and accuracy compared to using a single-shot regression head on a frozen backbone.

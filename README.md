# LearnMachine
This project showcases the use of Machine Learning (ML) for real-time object detection and classification of Arduino and ESP32 boards.


# Machine Learning for Object Detection of Arduino & ESP32 Using YOLOv8

## Description
This project demonstrates the use of **Machine Learning (ML)** for **real-time object detection and classification** of **Arduino** and **ESP32** boards. Using a combination of **image annotation**, **data augmentation**, and training a **YOLOv8 model**, this project enables high-accuracy detection of these devices. 

The main components of the project include:
- **Data Annotation**: Annotating images of Arduino and ESP32 boards using the **LabelImg** tool.
- **Data Augmentation**: Enhancing the dataset with transformations like rotation, flipping, and scaling to improve the model’s robustness.
- **YOLOv8 Model**: Training a **YOLOv8** object detection model to detect and classify Arduino and ESP32 boards in real-time.

The trained model can be deployed for applications such as inventory management, quality control, and educational purposes.

## Features
- **Real-Time Object Detection**: Detect and classify Arduino and ESP32 boards in images or video streams.
- **Data Annotation Tool**: Use the **LabelImg** tool to annotate images for training the model.
- **Data Augmentation**: Apply transformations like rotation, scaling, and flipping to the dataset to improve model performance.
- **YOLOv8 Model**: Train an object detection model using YOLOv8, optimized for both speed and accuracy.
- **Deployment**: The trained model can be deployed on various platforms for real-time inference.

## Technologies Used
- **YOLOv8**: You Only Look Once Version 8, for fast and accurate object detection.
- **Python**: For data preprocessing, augmentation, and model training.
- **OpenCV**: For image processing and augmentation.
- **LabelImg**: A tool for annotating images with bounding boxes for object detection.
- **Arduino & ESP32**: The objects being detected and classified.

## How It Works
1. **Data Annotation**: Images of Arduino and ESP32 boards are captured and annotated with the **LabelImg** tool. The boards are labeled with bounding boxes to indicate their position in the image.
2. **Data Augmentation**: The dataset is augmented by applying transformations such as rotation, flipping, and scaling to create a more diverse and robust training set.
3. **Model Training**: The augmented dataset is used to train the **YOLOv8** object detection model. This process involves optimizing the model to detect and classify Arduino and ESP32 boards with high accuracy.
4. **Object Detection**: The trained model is capable of detecting and classifying Arduino and ESP32 boards in images or video streams in real-time.
5. **Deployment**: Once trained, the model can be deployed on various platforms for real-time inference.


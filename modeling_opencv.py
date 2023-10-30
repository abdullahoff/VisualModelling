import cv2
import numpy as np
import tensorflow as tf

# Load the MobileNetV2 model with pre-trained weights
model = tf.keras.applications.MobileNetV2(weights="imagenet")

def classify_image(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    
    # Resize the image to the size MobileNetV2 expects (224x224)
    resized_image = cv2.resize(image, (224, 224))
    
    # Normalize the image pixels to be between 0 and 1
    normalized_image = resized_image / 255.0
    
    # Add an additional dimension to the image tensor (batch size dimension)
    input_tensor = np.expand_dims(normalized_image, axis=0)
    
    # Perform inference on the image
    predictions = model.predict(input_tensor)
    
    # Decode predictions to class names
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions.numpy())
    
    # Take the top prediction
    top_prediction = decoded_predictions[0][0]
    class_name, class_description, score = top_prediction
    
    # Display the image with the classification result
    cv2.putText(image, f"{class_description} ({score:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Classification Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "path_to_your_image.jpg"  # replace with your image path
    classify_image(image_path)

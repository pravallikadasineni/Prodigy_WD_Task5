
https://github.com/user-attachments/assets/a43d31d8-21b1-486a-914e-7156d83ef237
It is code for the avove video :
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np

# Function to load and preprocess images
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# Function to display images
def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title: plt.title(title)
    plt.axis('off')
content_image = load_img('content.jpg')
imshow(content_image, title="Content Image")
style_image = load_img('style.jpg')
imshow(style_image, title="Style Image")

https://github.com/user-attachments/assets/2f37e57d-3c9b-4c16-9e83-cc27d4f37819
It  is the code for the above video :
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np

# Function to load and preprocess images
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img
    # Function to display images
def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title: plt.title(title)
    plt.axis('off')
content_image = load_img('content.jpg')
imshow(content_image, title="Content Image")
style_image = load_img('style.jpg')
imshow(style_image, title="Style Image")

https://github.com/user-attachments/assets/87678511-0e72-4367-92d1-de540f154dcb
It is the code for the above video :
# Load the pre-trained model from TensorFlow Hub
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Load your images
content_image = load_img('content.jpg')
style_image = load_img('style.jpg')

# Perform style transfer
# The model returns a stylized image
stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

# Display the results
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')

plt.subplot(1, 2, 2)
imshow(stylized_image, 'Stylized Image')

plt.show()

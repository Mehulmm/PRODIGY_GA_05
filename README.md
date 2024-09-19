# PRODIGY_GA_05

Steps of the project: 
Here’s a concise version of the steps for the Neural Style Transfer project in Google Colab, presented as individual messages:
### Step 1: Install Required LibrariesInstall TensorFlow and Matplotlib:
```python!pip install tensorflow matplotlib
```
### Step 2: Import LibrariesImport the necessary libraries:
from tensorflow.keras.applications import vgg19from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras import backend as Kimport numpy as np
import matplotlib.pyplot as plt
### Step 3: Load and Process Images
Define functions to load and process images:
def load_and_process_image(img_path):    img = keras_image.load_img(img_path, target_size=(224, 224))
    img = keras_image.img_to_array(img)    return np.expand_dims(vgg19.preprocess_input(img), axis=0)
def deprocess_image(img):
    img = img[0]    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779    img[:, :, 2] += 123.68
    return np.clip(img, 0, 255).astype('uint8')
### Step 4: Load Content and Style ImagesLoad your content and style images:
style_image = load_and_process_image('path_to_your_style_image.jpg')
### Step 5: Define Model
Load the VGG19 model and specify the layers to use:``python
model = vgg19.VGG19(weights='imagenet', include_top=False)content_layer = 'block5_conv2'
style_layers = [f'block{i}_conv1' for i in range(1, 6)]outputs = [model.get_layer(layer).output for layer in [content_layer] + style_layers]
model = tf.keras.Model(inputs=model.input, outputs=outputs)``
### Step 6: Define Loss Functions
Define the content and style loss functions:
def get_content_loss(base_content, target): return K.sum(K.square(base_content - target))def gram_matrix(x): return K.dot(K.batch_flatten(K.permute_dimensions(x, (2, 0, 1))), K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))) / tf.cast(K.shape(x)[1] * K.shape(x)[2] * int(x.shape[-1]), K.floatx())
def get_style_loss(base_style, target): return K.sum(K.square(gram_matrix(base_style) - gram_matrix(target)))
### Step 7: Compute Total Loss
Calculate the total loss:
def total_loss(content_weight, style_weight, content_output, style_outputs):    return content_weight * get_content_loss(content_output, content_image) + \
           style_weight * K.add_n([get_style_loss(style_output, style_image) for style_output in style_outputs])
### Step 8: Initialize Generated Image and Optimizer
Initialize the generated image and optimizer:
generated_image = tf.Variable(content_image, dtype=tf.float32)optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)
### Step 9: Training Step FunctionDefine the training step function:
def train_step():    with tf.GradientTape() as tape:
        outputs = model(generated_image)        loss = total_loss(1e3, 1e-2, outputs[0], outputs[1:])
    grads = tape.gradient(loss, generated_image)    optimizer.apply_gradients([(grads, generated_image)])
    return loss
### Step 10: Run Optimization
Execute the optimization loop:
num_iterations = 1000for i in range(num_iterations):
    loss = train_step()    if i % 100 == 0:
        print(f"Iteration: {i}, Loss: {loss.numpy()}")
### Step 11: Display Result
Visualize the generated image:
result_image = deprocess_image(generated_image.numpy())plt.imshow(result_image)
plt.axis('off')plt.show()

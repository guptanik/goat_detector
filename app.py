import numpy as np
from flask import Flask, request, render_template
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import gdown
import os
import urllib.request
from PIL import Image
import cv2

app = Flask(__name__)

model = None

# load label to names mapping for visualization purposes
labels_to_names = {0: 'Goat'}

# Set threshold
THRES_SCORE = 0.1
counter = 0

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    get_model()

    global counter
    counter = counter + 1
    print("Value of counter is - {}".format(counter))
    if (counter == 1) or (counter > 10):
        clear_all_files()
        counter = 1

    # working url = https://www.cips.org/PageFiles/161907/goat%20770%20x%20400.jpg.ashx?width=600&height=315&mode=max
    url = request.form['imageURL']

    # Get image from url
    input_image = get_image(url)
    input_image_path = 'static/input_image_{}.jpg'.format(counter)
    print("Input image path - {}".format(input_image_path))
    cv2.imwrite(input_image_path, input_image)

    # Make inference
    output_image = img_inference(input_image)
    output_image_path = 'static/output_image_{}.jpg'.format(counter)
    print("Output image path - {}".format(output_image_path))
    cv2.imwrite(output_image_path, output_image)

    return_val = "<img src='{}'/> <img src='{}'>".format(input_image_path, output_image_path)
    print("Return value from predict function is - {}".format(return_val))
    return return_val


def get_model():
    global model
    if model:
        pass
    else :
        return download_model()


def download_model():
    # download trained retinanet model
    url = 'https://drive.google.com/uc?export=download&id=1FCNO5h8lzKMLOWuzYXkTi11BkbTXEhbP'
    output = 'resnet50_csv_79.h5'
    gdown.download(url, output, quiet=True)

    # load retinanet model
    global model
    model_path = os.path.join(output)
    model = models.load_model(model_path, backbone_name='resnet50')
    model = models.convert_model(model)


def get_image(img_path):
    img = urllib.request.urlopen(img_path)
    image = np.ascontiguousarray(Image.open(img).convert('RGB'))
    image = image[:, :, ::-1]
    return image


def img_inference(image):
    # copy to draw on
    draw = image.copy()
    # draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < THRES_SCORE:
            continue

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)

        draw_caption(draw, b, caption)

    return draw


def clear_all_files():
    folder = 'static'
    for filename in os.listdir(folder):
        if filename.find('.jpg') == -1:
            continue
        file_path = os.path.join(folder, filename)
        try:
            os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


if __name__ == "__main__":
    app.run(debug=True)

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
import pickle
import gdown
import os

app = Flask(__name__)

# download trained retinanet model
url = 'https://drive.google.com/file/d/1FCNO5h8lzKMLOWuzYXkTi11BkbTXEhbP/view?usp=sharing'
output = 'resnet50_csv_79.h5'
gdown.download(url, output, quiet=True)

# load retinanet model
model_path = os.path.join(output)
model = models.load_model(model_path, backbone_name='resnet50')
model = models.convert_model(model)

# load label to names mapping for visualization purposes
labels_to_names = pd.read_csv(CLASSES_FILE,header=None).T.loc[0].to_dict()

# Set threshold
THRES_SCORE = 0.1

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    url = request.form['imageURL']

    output_image = img_inference(url)

    return render_template('index.html', output_image=output_image)


def img_inference(img_path):
  img = urllib.request.urlopen(img_path)
  image = np.ascontiguousarray(Image.open(img).convert('RGB'))
  image = image[:, :, ::-1]

  # copy to draw on
  draw = image.copy()
  draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

  # preprocess image for network
  image = preprocess_image(image)
  image, scale = resize_image(image)

  # process image
  start = time.time()
  boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
  print("processing time: ", time.time() - start)

  # correct for image scale
  boxes /= scale

  # visualize detections
  for box, score, label in zip(boxes[0], scores[0], labels[0]):
      # scores are sorted so we can break
      print("Score is - {}".format(score))
      if score < THRES_SCORE:
          continue

      color = label_color(label)

      b = box.astype(int)
      draw_box(draw, b, color=color)

      caption = "{} {:.3f}".format(labels_to_names[label], score)

      draw_caption(draw, b, caption)

  return draw


if __name__ == "__main__":
    app.run(debug=True)

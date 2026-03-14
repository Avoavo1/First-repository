from PIL import Image
import tensorflow as tf
import cv2
import os

from fontTools.misc.cython import returns

DATASET_PATH = 'dataset/'
num_classes = len(os.listdir(DATASET_PATH))#KOLICHESTVO PAPOK-KlASSOV
class_mode = 'binary'if num_classes == 2 else 'categorical'

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f'Virus: Fail has not detected in {image_path}')
        return
    try:
        img = Image.open(image_path)
        img.verify()
        img = Image.open(image_path)
    except (OSError , IOError):
        print(f"Virus: Povrezdenoe izobrazenoe - {image_path}")
        returns
    model = tf.keras.models.load_model("image_classifier.h5")
    img = cv2.imread(image_path)

    if img is None:
        print(f"Virus: Ne ydalos prochitat izobrazenie - {image_path}")
        return

    img=cv2.resize(img,(128,128))
    img = img / 255
    img= tf.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_names = os.listdir(DATASET_PATH)
    if class_names == "binary":
        prediction_class = class_names[int(bool(prediction[0] > 0.5))]
    else:
        prediction_class = class_names[tf.argmax(prediction, axis=-1).numpy()[0]]
    print(f"Model opredelila: {prediction_class}")


predict_image("dataset/imgg_.jpg")
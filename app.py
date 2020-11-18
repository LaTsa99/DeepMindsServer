import os
import gdown
from flask import Flask, request, jsonify, render_template, flash, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '.'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
gdown.cached_download('https://drive.google.com/file/d/1lJ9tPyfO2IJ0OwlfSUEnqq-3l2KfSd_F/view?usp=sharing', 'model.hdf5')
model = load_model('model.hdf5')

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_input(img):
	return img/.255

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	filename = ''
	imagefile = flask.request.files['image']
	if imagefile.filename == '':
		flash('No selected file')
		return redirect(request.url)
	if imagefile and allowed_file(imagefile.filename):
		filename = secure_filename(imagefile.filename)
		imagefile.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
	
	input_img = image.load_img(os.path.join(app.config['UPLOAD_FOLDER'], filename), target_size=(299, 299))
	x = image.img_to_array(input_img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	pred = model.predict(x)
	os.remove(filename)
	if pred > 0.5:
		return 1
	else:
		return 0


if __name__ = '__main__':
	app.run()
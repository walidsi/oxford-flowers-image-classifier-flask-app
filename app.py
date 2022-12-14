from fileinput import filename
from flask import Flask, render_template, redirect, url_for, jsonify
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, DateTimeField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename
import predict
import os
import json
import datetime
import tzlocal  


app = Flask(__name__)
app.config['SECRET_KEY'] = 'SUPERSECRETKEY'
app.config['UPLOAD_FOLDER'] = 'static/files/'

class UploadFileForm(FlaskForm):
    file = FileField('File', validators=[InputRequired()])
    submit = SubmitField('Upload')

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    form = UploadFileForm()

    now = datetime.datetime.now(tzlocal.get_localzone()).strftime("%Y-%m-%d %I:%M:%S %p %z")

    if form.validate_on_submit():
        file = form.file.data
        local_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(local_file_path)
        #return {'message': 'File uploaded successfully'}
                
        result = predict.predict(local_file_path)

        return render_template('index.html', form=form, filename=secure_filename(file.filename), result=result, date=now)

    return render_template('index.html', form=form, date=now)

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='files/' + filename), code=301)

@app.route('/about')
def about():
    return render_template('about.html')

@app.after_request
def add_header(response):
    response.cache_control.max_age = 0  # set this to 0 to force a reload every time
    return response

# if __name__ == '__main__':
#     app.run(debug=True) 
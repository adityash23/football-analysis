from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__)

VIDEO_PATH = os.path.join(app.root_path, 'static', 'videos')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/videos/<filename>')
def video(filename):
    return send_from_directory(VIDEO_PATH, filename)

if __name__ == '__main__':
    app.run(debug=True)

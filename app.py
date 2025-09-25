from flask import Flask, render_template, request, url_for, redirect, session
from werkzeug.utils import secure_filename
import os
from datetime import timedelta
from botchat import du_doan_benh

app = Flask(__name__)
app.secret_key = "replace_this_with_a_random_secret"
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

@app.route("/", methods=["GET"])
def index():
    session.permanent = True
    if 'messages' not in session:
        session['messages'] = []
    return render_template("chat.html", messages=session.get('messages', []))

@app.route("/send", methods=["POST"])
def send():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        from datetime import datetime
        filename = datetime.now().strftime("%Y%m%d%H%M%S_") + filename
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        messages = session.get('messages', [])
        messages.append({"sender": "user", "type": "image", "content": f"uploads/{filename}"})

        ket_qua = du_doan_benh(save_path)
        messages.append({"sender": "bot", "type": "text", "content": ket_qua["text"]})
        session['messages'] = messages

    return redirect(url_for('index'))

@app.route("/clear")
def clear():
    session.pop('messages', None)
    return redirect(url_for('index'))

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)


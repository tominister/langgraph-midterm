from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import os
from rag import RAG

app = Flask(__name__)
app.secret_key = "your_secret_key"

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize RAG
rag = RAG(model_name="google/flan-t5-base")

@app.route("/", methods=["GET"])
def index():
    files_info = []
    uploaded_files = os.listdir(app.config["UPLOAD_FOLDER"])
    for f in uploaded_files:
        status = "Indexed" if rag.is_indexed(f) else "Not indexed"
        points = rag.count_points_for_file(f) if status == "Indexed" else 0
        files_info.append({"filename": f, "status": status, "points": points})

    chat_history = session.get("chat_history", [])
    return render_template("index.html", files_info=files_info, chat_history=chat_history)

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return redirect(url_for("index"))
    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("index"))
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)
    return redirect(url_for("index"))

@app.route("/index_file/<filename>", methods=["POST"])
def index_file(filename):
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    rag.index_file(file_path)
    return redirect(url_for("index"))

@app.route("/delete_file/<filename>", methods=["POST"])
def delete_file(filename):
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    rag.delete_file(filename)
    return redirect(url_for("index"))

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form["question"]
    file_filter = request.form.get("file_filter")
    context_chunks = rag.retrieve(question, file_filter=file_filter)
    answer = rag.generate_answer(question, context_chunks)
    chat_history = session.get("chat_history", [])
    chat_history.append({"question": question, "answer": answer})
    session["chat_history"] = chat_history
    return redirect(url_for("index"))

@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    session["chat_history"] = []
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)

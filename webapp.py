from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def submission():
    return render_template("index.html")

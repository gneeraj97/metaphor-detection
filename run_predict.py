import pandas as pd
from run_test_live import predict
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"])
def home():
    Disp = "Select the word and insert the paragraph that you want to check"
    return render_template("index.html",  pred = Disp)

@app.route("/pred", methods = ["POST"])
def pred():
    para = ""
    result = ""
    # word = ""
    if request.method == "POST":
        para = request.form.get("paragraph")
        word = request.form.get("metaphor")
    result = predict(word, para)
    return render_template("index.html",  pred = result, text = para, selected_metaphor = word )


if __name__ =="__main__":
    app.run(debug = True)

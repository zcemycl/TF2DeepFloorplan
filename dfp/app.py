import multiprocessing as mp
import os
import random
from argparse import Namespace

import matplotlib.image as mpimg
import numpy as np
import requests
from flask import Flask, request, send_file

import dfp._paths
from dfp.deploy import main

print(dfp._paths)
app = Flask(__name__)


def saveStreamFile(stream, fnum):
    stream.save(fnum + ".jpg")


def saveStreamURI(stream, fnum):
    with open(fnum + ".jpg", "wb") as handler:
        handler.write(stream)


@app.route("/")
def home():
    return {"message": "Hello Flask!"}


@app.route("/process", methods=["POST"])
def process_image():
    fnum = str(random.randint(0, 10000))
    finname = "resources/30939153.jpg"
    postprocess = True
    colorize = True
    foutname = fnum + "-out.jpg"
    output = "/tmp"

    # input image: either local file or uri
    if "file" in request.files:
        print("File mode...")
        try:
            saveStreamFile(request.files["file"], fnum)
            finname = fnum + ".jpg"
            print("files: ", request.files)
            print(request.files["file"])
        except KeyError:
            print("KeyError!")
            return {"message": "input error"}, 400

    if request.json and "uri" in request.json.keys():
        print("URI mode...")
        uri = request.json["uri"]
        try:
            data = requests.get(uri).content
            saveStreamURI(data, fnum)
            finname = fnum + ".jpg"
        except KeyError:
            print("KeyError!")
            return {"message": "input error"}, 400

    # postprocess
    if "postprocess" in request.form.keys():
        print("Detect postprocess mode 1...")
        print(request.form.getlist("postprocess")[0])
        postprocess = bool(int(request.form.getlist("postprocess")[0]))

    if request.json and "postprocess" in request.json.keys():
        print("Detect postprocess mode 2...")
        postprocess = bool(request.json["postprocess"])

    # colorize
    if "colorize" in request.form.keys():
        print(request.form.getlist("colorize")[0])
        colorize = bool(int(request.form.getlist("colorize")[0]))

    if request.json and "colorize" in request.json.keys():
        colorize = bool(request.json["colorize"])

    # output path
    if "output" in request.form.keys():
        print(request.form.getlist("output")[0])
        output = str(request.form.getlist("output")[0]).strip()

    if request.json and "output" in request.json.keys():
        output = str(request.json["output"])

    args = Namespace(
        image=finname,
        weight="log/store/G",
        loadmethod="log",
        postprocess=postprocess,
        colorize=colorize,
        save=os.path.join(output, foutname),
    )
    print(args)

    with mp.Pool() as pool:
        result = pool.map(main, [args])[0]

    print("Output Image shape: ", np.array(result).shape)

    if args.save:
        mpimg.imsave(args.save, np.array(result).astype(np.uint8))

    try:
        callback = send_file(
            os.path.join(output, foutname), mimetype="image/jpg"
        )
        return callback, 200
    except FileNotFoundError:
        return {"message": "input error"}, 400
    finally:
        os.system("rm " + os.path.join(output, foutname))
        if finname != "resources/30939153.jpg":
            os.system("rm " + finname)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=1111)

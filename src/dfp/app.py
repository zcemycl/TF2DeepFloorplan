import multiprocessing as mp
import os
import random
from argparse import Namespace

import matplotlib.image as mpimg
import numpy as np
import requests
from flask import Flask, request, send_file
from werkzeug.datastructures import FileStorage

from .deploy import main
from .utils.settings import overwrite_args_with_toml

app = Flask(__name__)
app.config["UPLOAD_EXTENSIONS"] = [".jpg", ".png", ".gif"]

args = Namespace(tomlfile="docs/app.toml")
args = overwrite_args_with_toml(args)
finname = "resources/30939153.jpg"
output = "/tmp"


def saveStreamFile(stream: FileStorage, fnum: str):
    stream.save(fnum + ".jpg")


def saveStreamURI(stream: bytes, fnum: str):
    with open(fnum + ".jpg", "wb") as handler:
        handler.write(stream)


@app.route("/")
def home():
    return {"message": "Hello Flask!"}


@app.route("/upload", methods=["POST"])
def dummy():
    finname = "resources/30939153.jpg"
    fnum = str(random.randint(0, 10000))
    foutname = fnum + "-out.jpg"
    if "file" in request.files:
        saveStreamFile(request.files["file"], fnum)
        finname = fnum + ".jpg"

    postprocess = (
        False
        if "postprocess" not in request.form.keys()
        else bool(int(request.form.getlist("postprocess")[0]))
    )
    colorize = (
        False
        if "colorize" not in request.form.keys()
        else bool(int(request.form.getlist("colorize")[0]))
    )

    args.image = finname
    args.postprocess = postprocess
    args.colorize = colorize
    args.save = os.path.join(output, foutname)
    app.logger.info(args)
    with mp.Pool() as pool:
        result = pool.map(main, [args])[0]

    app.logger.info(f"Output Image shape: {np.array(result).shape}")
    if args.save:
        mpimg.imsave(args.save, np.array(result).astype(np.uint8))

    try:
        callback = send_file(
            os.path.join(output, foutname), mimetype="image/jpg"
        )
        return callback, 200
    except Exception:
        return {"message": "send error"}, 400
    finally:
        os.system("rm " + os.path.join(output, foutname))
        if finname != "resources/30939153.jpg":
            os.system("rm " + finname)
    return {"message": "hello"}


@app.route("/uri", methods=["POST"])
def process_image():
    fnum = str(random.randint(0, 10000))
    finname = "resources/30939153.jpg"
    foutname = fnum + "-out.jpg"
    postprocess = (
        bool(request.json["postprocess"])
        if request.json and "postprocess" in request.json.keys()
        else False
    )
    colorize = (
        bool(request.json["colorize"])
        if request.json and "colorize" in request.json.keys()
        else False
    )

    # input image: uri
    if request.json and "uri" in request.json.keys():
        app.logger.info("URI mode...")
        uri = request.json["uri"]
        try:
            data = requests.get(uri).content
            saveStreamURI(data, fnum)
            finname = fnum + ".jpg"
        except Exception:
            return {"message": "input error"}, 400

    args.image = finname
    args.postprocess = postprocess
    args.colorize = colorize
    args.save = os.path.join(output, foutname)
    app.logger.info(args)

    with mp.Pool() as pool:
        result = pool.map(main, [args])[0]

    app.logger.info(f"Output Image shape: {np.array(result).shape}")

    if args.save:
        mpimg.imsave(args.save, np.array(result).astype(np.uint8))

    try:
        callback = send_file(
            os.path.join(output, foutname), mimetype="image/jpg"
        )
        return callback, 200
    except Exception:
        return {"message": "send error"}, 400
    finally:
        os.system("rm " + os.path.join(output, foutname))
        if finname != "resources/30939153.jpg":
            os.system("rm " + finname)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=1111)

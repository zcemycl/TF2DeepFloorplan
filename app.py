from flask import Flask, jsonify, request, send_file
from deploy import *
import argparse
from argparse import Namespace
import multiprocessing as mp
import random
import requests

app = Flask(__name__)

@app.route("/process",methods=['POST'])
def process_image():
    dct = request.json
    print(request.json)

    # arguments
    uri = dct["uri"] if 'uri' in dct.keys() else None
    postprocess = bool(dct['postprocess']) if 'postprocess' in dct.keys() else True
    colorize = bool(dct['colorize']) if 'colorize' in dct.keys() else True

    # download image
    fnum = str(random.randint(0,10000))
    if uri:
        data = requests.get(uri).content
        with open(fnum+'.jpg','wb') as handler:
            handler.write(data)
        finname = fnum+'.jpg'
    else:
        finname = 'resources/30939153.jpg'
    foutname = fnum+'-out.jpg'

    args = Namespace(image=finname,
            weight='../log/store/G',
            postprocess=postprocess,colorize=colorize,
            save=foutname)

    with mp.Pool() as pool:
        result = pool.map(main,[args])[0]

    print('Output Image shape: ',np.array(result).shape)

    if args.save:
        mpimg.imsave(args.save,np.array(result).astype(np.uint8))

    callback = send_file(foutname,mimetype='image/jpg')

    return callback, 200


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=1111)


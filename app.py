import argparse
import multiprocessing as mp
import random
from argparse import Namespace

import requests
from flask import Flask, jsonify, request, send_file

from deploy import *

app = Flask(__name__)

@app.route("/process",methods=['POST'])
def process_image():
    fnum = str(random.randint(0,10000))
    finname = 'resources/30939153.jpg'
    postprocess = True; colorize = True
    foutname = fnum+'-out.jpg'

    # input image: either local file or uri
    if 'file' in request.files:
        try:
            request.files['file'].save(fnum+'.jpg')
            finname = fnum+'.jpg'
            print('files: ',request.files)
            print(request.files['file'])
        except:
            pass

    if request.json and 'uri' in request.json.keys():
        uri = request.json['uri']
        try: 
            data = requests.get(uri).content
            with open(fnum+'.jpg','wb') as handler:
                handler.write(data)
            finname = fnum+'.jpg'
        except:
            pass

    # postprocess
    if 'postprocess' in request.form.keys():
        postprocess = bool(request.form.getlist('postprocess')[0])

    if request.json and 'postprocess' in request.json.keys():
        postprocess = bool(request.json['postprocess'])


    # colorize
    if 'colorize' in request.form.keys():
        colorize = bool(request.form.getlist('colorize')[0])
    
    if request.json and 'colorize' in request.json.keys():
        colorize = bool(request.json['colorize'])


    args = Namespace(image=finname,
            weight='log/store/G',loadmethod='log',
            postprocess=postprocess,colorize=colorize,
            save=foutname)

    with mp.Pool() as pool:
        result = pool.map(main,[args])[0]

    print('Output Image shape: ',np.array(result).shape)

    if args.save:
        mpimg.imsave(args.save,np.array(result).astype(np.uint8))


    try:
        callback = send_file(foutname,mimetype='image/jpg')
        return callback, 200
    except:
        return {'message': 'input error'}, 400
    finally:
        os.system('rm '+foutname)
        if finname != 'resources/30939153.jpg':
            os.system('rm '+finname)


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=1111)

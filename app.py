from flask import Flask, jsonify, request, send_file
from deploy import *
import argparse
from argparse import Namespace
import multiprocessing as mp

app = Flask(__name__)

def worker(queue):
    ret = queue.get()
    print(ret)
    result = main(ret)
    print(result)
    queue.put(result)


@app.route("/process",methods=['POST'])
def process_image():
    args = Namespace(image='resources/30939153.jpg',
            weight='../log/store/G',
            postprocess=True,colorize=True,
            save='1.jpg')

    with mp.Pool() as pool:
        result = pool.map(main,[args])[0]

    print('Output Image shape: ',np.array(result).shape)

    if args.save:
        mpimg.imsave(args.save,np.array(result).astype(np.uint8))

    callback = send_file('1.jpg',mimetype='image/jpg')

    return callback, 200


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=1111)


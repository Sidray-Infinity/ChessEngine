import chess
from flask import Flask, render_template, request, redirect, url_for
from random import randint
import numpy as np
import traceback

import tensorflow as tf

from dataset import serialize


app = Flask(__name__)
board = chess.Board()

# fromModel = tf.keras.models.load_model("app/TrainedNet/From/chess_model-23.h5")
# toModel = tf.keras.models.load_model("app/TrainedNet/To/chess_model-13.h5")

def create_model():
    inputs = tf.keras.Input(shape=(1, 8, 8, 12))
    x = tf.keras.layers.Conv3D(filters=128, kernel_size=(2, 2, 1), padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv3D(filters=128, kernel_size=(2, 2, 1), padding="same", activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)

    d1 = tf.keras.layers.Dense(1024, activation="relu")(x)
    d1 = tf.keras.layers.Dense(1024, activation="relu")(d1)
    d1 = tf.keras.layers.Dense(64, activation="softmax", name="from")(d1)

    d2 = tf.keras.layers.Dense(1024, activation="relu")(x)
    d2 = tf.keras.layers.Dense(1024, activation="relu")(d2)
    d2 = tf.keras.layers.Dense(64, activation="softmax")(d2)

    model = tf.keras.Model(inputs=inputs, outputs=[d1, d2])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
    #model.summary()


    model.load_weights("model-weights.h5")

    return model

# model = tf.keras.models.load_model("TrainedNet/hybrid-chess_model-4.h5")
# model.save_weights("model-weights.h5")

model = create_model()

def save_board(board):
    svg_data = chess.svg.board(board=board, coordinates=True, size=400)
    cairosvg.svg2png(bytestring=svg_data, write_to="static/board_img.png")


@app.route("/")
def print_board():
    ret = open("templates/index.html").read()
    return ret.replace('start', board.fen())


@app.route("/turn")
def turn():
    
    source = int(request.args.get('from', default=''))
    target = int(request.args.get('to', default=''))
    promotion = True if request.args.get('promote', default='') == 'true' else False

    
    #input()
    move = chess.Move(source, target, promotion=chess.QUEEN if promotion else None)

    try :
        if move in board.legal_moves:
            board.push(move)
        else:
            response = app.response_class(
                response = "invalidMove",
                status = 200
            )
            return response
    except:
        traceback.print_exc()

    if board.is_checkmate():
        response = app.response_class(
            response = "checkmate",
            status = 200
        )
        return response

    
    black_move()

    if board.is_checkmate():
        response = app.response_class(
            response = "checkmate",
            status = 200
        )
        return response

    response = app.response_class(
      response = board.fen(),
      status=200
    )

    return response


def black_move():

    X = serialize(board.fen())
    X = np.expand_dims(X, axis=0)
    X = np.expand_dims(X, axis=1)

    # from_sqrs = fromModel.predict(X)[0]
    # to_sqrs = toModel.predict(X)[0]

    pred = model.predict(X)
    #print('PRED', pred.shape)

    from_sqrs, to_sqrs = pred[0][0], pred[1][0]

    fromSqrs = np.zeros(64)
    toSqrs = np.zeros(64)

    # Make all squares not related to legal moves 0
    for move in board.legal_moves:
        fromSqrs[move.from_square] = from_sqrs[move.from_square]   
        toSqrs[move.to_square] = to_sqrs[move.to_square]   

    # Extract the from square with max probability
    f_square = np.argmax(fromSqrs)

    # Find the corresponding to square with max probability
    max_pred = -1.0
    for move in board.legal_moves:
        if move.from_square == f_square:
            if toSqrs[move.to_square] > max_pred:
                max_pred = toSqrs[move.to_square]

    t_square = np.where(toSqrs == max_pred)
    
    move = chess.Move(f_square, t_square[0][0])

    try:
        board.push(move)
    except:
        traceback.print_exc()

@app.route("/new_game")
def new_game():
    board = chess.Board()
    response = app.response_class(
      response = board.fen(),
      status=200
    )

    return response


# @app.after_request
# def add_header(response):
#     """
#     Add headers to both force latest IE rendering engine or Chrome Frame,
#     and also to cache the rendered page for 10 minutes.
#     """
#     response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
#     response.headers['Cache-Control'] = 'public, max-age=0'
#     return response


if __name__ == "__main__":
    
    app.run()

import chess
import chess.svg
from flask import Flask, render_template, request, redirect, url_for
import cairosvg
from random import randint
import numpy as np
import traceback

import tensorflow as tf

from dataset import serialize


app = Flask(__name__)

fromModel = tf.keras.models.load_model("TrainedNet/From/chess_model-23.h5")
toModel = tf.keras.models.load_model("TrainedNet/To/chess_model-13.h5")


def save_board(board):
    svg_data = chess.svg.board(board=board, coordinates=True, size=400)
    cairosvg.svg2png(bytestring=svg_data, write_to="static/board_img.png")


@app.route("/")
def print_board():
    save_board(board)
    if(board.turn == chess.BLACK):
        return redirect("/getmove_black")

    return render_template('index.html', board_img="static/board_img.png")


@app.route("/", methods=['POST'])
def getmove_white():
    try:
        move = request.form['move']
        move = chess.Move.from_uci(move)
        if move in board.legal_moves:
            board.push(move)
    except:
        return "<H1>SOME ERROR</H1>"

    if board.is_checkmate():
        return "<center><h1>CHECKMATE</h1></center>"

    return redirect("/")



@app.route("/getmove_black")
def getmove_black():

    X = serialize(board.fen())
    X = np.expand_dims(X, axis=0)
    X = np.expand_dims(X, axis=1)

    from_sqrs = fromModel.predict(X)[0]
    to_sqrs = toModel.predict(X)[0]

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
        return "<H1>SOME ERROR</H1>"

    if board.is_checkmate():
        return "<center><h1>CHECKMATE</h1></center>"

    return redirect("/")


@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == "__main__":
    board = chess.Board()
    app.run(port=8000, debug=True, use_reloader=True)

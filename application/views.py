from flask import request, redirect, url_for, render_template, flash, session, jsonify
from application import app
from application.othello_ai_prediction import get_best_move
# from elasticsearch import Elasticsearch
import os
import sys

@app.route('/ai-othello/v1/best-move', methods=['POST'])
def show_ai_othello_best_move():

    # return request.get_data()

    # print('Hello world!', file=sys.stderr)
    # print(str(request), file=sys.stderr)
    # print(str(request.headers), file=sys.stderr)
    # print(str(request.get_data()), file=sys.stderr)
    # print(str(request.get_json()), file=sys.stderr)

    # jsonリクエストから値を取得
    game_status = request.json
    # board_array = game_status.get('board_array')
    # next_player = game_status.get('next_player')
    
    # print(str(game_status), flush=True)
    # print(game_status, file=sys.stdout)
    next_player = game_status['nextplayer']  
    board_array = game_status['boardarray']
    # next_player = game_status['next_player']
    print(next_player, file=sys.stdout)
    print(str(type(next_player)), file=sys.stdout)
    print(board_array, file=sys.stdout)
    print(str(type(board_array)), file=sys.stdout)

    # 最適な手を取得
    best_move = get_best_move(board_array, next_player)

    # jsonレスポンスの作成
    response = jsonify({"bestmove": best_move})
    print(best_move, file=sys.stdout)

    return response
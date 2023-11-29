#!/usr/bin/env python3
import sys
import config as cf
import time
import threading
import global_storage as gs
import signal
import os
import json
from flask import Flask, request, send_from_directory, redirect
from flask_sockets import Sockets

class RemoteController(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)

        app = Flask(__name__, static_url_path='') 
        sockets = Sockets(app)

        @app.route("/")
        def homepage():
            return redirect("/static/index.html")

        @app.route('/static/<path:path>')
        def send_static_file(path):
            return send_from_directory('static', path)

        @sockets.route('/control')
        def echo_socket(ws):
            while not ws.closed:
                message = ws.receive()
                control_params = json.loads(message)
                gs.remote_control_speed = control_params["speed"]
                gs.remote_control_steer_angle = control_params["steer"]
                gs.speed = min(gs.remote_control_speed / gs.remote_control_max_speed * cf.MAX_SPEED, cf.MAX_SPEED)
                gs.steer = -max(min(gs.remote_control_steer_angle / gs.remote_control_max_steer_angle * cf.MIN_ANGLE, cf.MAX_ANGLE), cf.MIN_ANGLE)
                gs.record_videos = control_params["record_videos"]
                if control_params["emergency_stop"]:
                    gs.emergency_stop = control_params["emergency_stop"]
                gs.last_time_control_signal = time.time()
        self.app = app


    def run(self):
        from gevent import pywsgi
        from geventwebsocket.handler import WebSocketHandler
        server = pywsgi.WSGIServer(('0.0.0.0', 5000), self.app, handler_class=WebSocketHandler)
        server.serve_forever()




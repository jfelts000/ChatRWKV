import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from flask import Flask, render_template, session, request
from flask_session import Session
from flask_socketio import SocketIO, emit
import uuid
from chat import process_message
import chat

app = Flask(__name__)
#app.secret_key = str(uuid.uuid4())
socketio = SocketIO(app)
chat.init_chat(socketio) 
@app.route('/')
def home():
    session_id = str(uuid.uuid4())
    return render_template("index.html", session_id=session_id)


@app.route('/favicon.ico')
def favicon():
    return ''

@socketio.on('connect')
def handle_connect():
    session_id = request.headers.get('sessionId')
    print(f'Client connected with session ID: {session_id}')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('user_message')
def handle_message(data):
    session_id = data['session_id']
    message = data['message']
    print(f'UserMessage Info: SessionID: {session_id} message: {message}')
    response = process_message(message, session_id,  socketio)
    emit('bot_response', {'sessionId': session_id, 'response': response})
    print(f'Post Emit: sessionId: {session_id} response: {response}')


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)

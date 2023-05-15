########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import logging
import os, copy, types, gc, sys
import uuid
from flask_socketio import SocketIO
# Import flask session and uuid
from flask import session
import uuid
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{current_path}/../rwkv_pip_package/src')

import numpy as np
from prompt_toolkit import prompt
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
np.set_printoptions(precision=4, suppress=True, linewidth=200)
args = types.SimpleNamespace()

print('\n\nChatRWKV v2 https://github.com/BlinkDL/ChatRWKV')

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
socketio = None
def init_chat(socketio):
    def handle_user_message(data):
        session_id = data['session_id']
        message = data['message']
        response = process_message(session_id, message, socketio)
        socketio.emit('bot_response', {'response': response})

    socketio.on('user_message', handle_user_message)
    

# Tune these below (test True/False for all of them) to find the fastest setting:
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
# torch._C._jit_override_can_fuse_on_cpu(True)
# torch._C._jit_override_can_fuse_on_gpu(True)
# torch._C._jit_set_texpr_fuser_enabled(False)
# torch._C._jit_set_nvfuser_enabled(False)

########################################################################################################
#
# fp16 = good for GPU (!!! DOES NOT support CPU !!!)
# fp32 = good for CPU
# bf16 = less accuracy, supports some CPUs
# xxxi8 (example: fp16i8) = xxx with int8 quantization to save 50% VRAM/RAM, slightly less accuracy
#
# Read https://pypi.org/project/rwkv/ for Strategy Guide
#
########################################################################################################

# args.strategy = 'cpu fp32'
#args.strategy = 'cuda fp16i8 *0+ -> cpu fp32 *1'
# args.strategy = 'cuda:0 fp16 -> cuda:1 fp16'
# args.strategy = 'cuda fp16i8 *10 -> cuda fp16'
args.strategy = 'cuda fp16i8 *0+ -> cpu fp32 *1'
# args.strategy = 'cuda fp16i8 -> cpu fp32 *10'
# args.strategy = 'cuda fp16i8 *10+'

os.environ["RWKV_JIT_ON"] = '1' # '1' or '0', please use torch 1.13+ and benchmark speed
os.environ["RWKV_CUDA_ON"] = '1' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

CHAT_LANG = 'English' # English // Chinese // more to come

# Download RWKV models from https://huggingface.co/BlinkDL
# Use '/' in model path, instead of '\'
# Use convert_model.py to convert a model for a strategy, for faster loading & saves CPU RAM 
if CHAT_LANG == 'English':
    args.MODEL_NAME = 'RWKV-4-Raven-14B-v10-Eng99%-Other1%-20230427-ctx8192-fp16i820.pth'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-raven/RWKV-4-Raven-7B-v10-Eng99%-Other1%-20230418-ctx8192'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-14b/RWKV-4-Pile-14B-20230313-ctx8192-test1050'

elif CHAT_LANG == 'Chinese': # Raven系列可以对话和 +i 问答。Novel系列是小说模型，请只用 +gen 指令续写。
    args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-raven/RWKV-4-Raven-7B-v9x-Eng49%-Chn50%-Other1%-20230418-ctx4096'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-novel/RWKV-4-Novel-7B-v1-ChnEng-20230409-ctx4096'

elif CHAT_LANG == 'Japanese':
    args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-raven/RWKV-4-Raven-14B-v8-EngAndMore-20230408-ctx4096'
    # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-raven/RWKV-4-Raven-7B-v9-Eng86%-Chn10%-JpnEspKor2%-Other2%-20230414-ctx4096'

# -1.py for [User & Bot] (Q&A) prompt
# -2.py for [Bob & Alice] (chat) prompt
PROMPT_FILE = f'{current_path}/prompt/default/{CHAT_LANG}-2.py'

CHAT_LEN_SHORT = 40
CHAT_LEN_LONG = 200
FREE_GEN_LEN = 256

# For better chat & QA quality: reduce temp, reduce top-p, increase repetition penalties
# Explanation: https://platform.openai.com/docs/api-reference/parameter-details
GEN_TEMP = 0.8 # It could be a good idea to increase temp when top_p is low
GEN_TOP_P = 0.5 # Reduce top_p (to 0.5, 0.2, 0.1 etc.) for better Q&A accuracy (and less diversity)
GEN_alpha_presence = 0.2 # Presence Penalty
GEN_alpha_frequency = 0.2 # Frequency Penalty
AVOID_REPEAT = '，：？！'

CHUNK_LEN = 256 # split input into chunks to save VRAM (shorter -> slower)

# args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-raven/RWKV-4-Raven-7B-v9-Eng86%-Chn10%-JpnEspKor2%-Other2%-20230414-ctx4096'
# args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-raven/RWKV-4-Raven-3B-v9x-Eng49%-Chn50%-Other1%-20230417-ctx4096'
# args.MODEL_NAME = '/fsx/BlinkDL/CODE/_PUBLIC_/RWKV-LM/RWKV-v4neo/7-ENZH/rwkv-88'
# args.MODEL_NAME = '/fsx/BlinkDL/CODE/_PUBLIC_/RWKV-LM/RWKV-v4neo/7-JP/rwkv-5'

########################################################################################################
# 1. Add a chat_session dictionary
chat_sessions = {}

# 2. Use request.sid to identify the session
def handle_message(session_id, message):
    session_id = request.sid
    if session_id not in chat_sessions:
        chat_sessions[session_id] = {'bot': copy.deepcopy(bot), 'user': copy.deepcopy(user)}
        bot = chat_sessions[session_id]['bot']
        user = chat_sessions[session_id]['user']

    message = message.strip()
    if message:
        user.add(message)
        response = interface(user, bot, model, pipeline, chunk_len=CHUNK_LEN, top_p=GEN_TOP_P, temp=GEN_TEMP, repetition_penalty=[GEN_alpha_presence, GEN_alpha_frequency], avoid_tokens=AVOID_REPEAT_TOKENS)
        bot.add(response)

    # 3. Include session_id as the first argument
        socketio.emit('bot_response', (session_id, {'response': response}), broadcast=True)


def on_connect():
    session_id = request.sid
    chat_sessions[session_id] = {'bot': copy.deepcopy(bot), 'user': copy.deepcopy(user)}


def on_disconnect():
    session_id = request.sid
    if session_id in chat_session:
        del chat_session[session_id]
        
print(f'\n{CHAT_LANG} - {args.strategy} - {PROMPT_FILE}')
from rwkv.model import RWKV
from rwkv.utils import PIPELINE

def load_prompt(PROMPT_FILE):
    variables = {}
    with open(PROMPT_FILE, 'rb') as file:
        exec(compile(file.read(), PROMPT_FILE, 'exec'), variables)
    user, bot, interface, init_prompt = variables['user'], variables['bot'], variables['interface'], variables['init_prompt']
    init_prompt = init_prompt.strip().split('\n')
    for c in range(len(init_prompt)):
        init_prompt[c] = init_prompt[c].strip().strip('\u3000').strip('\r')
    init_prompt = '\n' + ('\n'.join(init_prompt)).strip() + '\n\n'
    return user, bot, interface, init_prompt

# Load Model

print(f'Loading model - {args.MODEL_NAME}')
model = RWKV(model=args.MODEL_NAME, strategy=args.strategy)
pipeline = PIPELINE(model, f"{current_path}/20B_tokenizer.json")
END_OF_TEXT = 0
END_OF_LINE = 187
# pipeline = PIPELINE(model, "cl100k_base")
# END_OF_TEXT = 100257
# END_OF_LINE = 198

model_tokens = []
model_state = None

AVOID_REPEAT_TOKENS = []
for i in AVOID_REPEAT:
    dd = pipeline.encode(i)
    assert len(dd) == 1
    AVOID_REPEAT_TOKENS += dd

########################################################################################################
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class ChatSession:
    def __init__(self, model, pipeline):
        self.model = model
        self.pipeline = pipeline
        self.model_tokens = []
        self.model_state = None
        self.all_state = {}
        self.user = "user"
        self.bot = "bot"
        self.interface = "interface"
        # Initialize the session
        self.initialize_session()

    def initialize_session(self):
        logging.info("Initializing chat session.")
        user, self.bot, self.interface, init_prompt = load_prompt(PROMPT_FILE)
        out = self.run_rnn(self.pipeline.encode(init_prompt))
        self.save_all_stat('', 'chat_init', out)
        gc.collect()
        torch.cuda.empty_cache()

        srv_list = ['dummy_server']
        for s in srv_list:
            self.save_all_stat(s, 'chat', out)

    def run_rnn(self, tokens, newline_adj=0):
        logging.info("Running RNN with tokens: %s", tokens)
        tokens = [int(x) for x in tokens]
        self.model_tokens += tokens

        while len(tokens) > 0:
            out, self.model_state = self.model.forward(tokens[:CHUNK_LEN], self.model_state)
            tokens = tokens[CHUNK_LEN:]

        out[END_OF_LINE] += newline_adj  # adjust \n probability

        if self.model_tokens[-1] in AVOID_REPEAT_TOKENS:
            out[self.model_tokens[-1]] = -999999999
        return out

    def save_all_stat(self, srv, name, last_out):
        logging.info("Saving state for srv: %s, name: %s", srv, name)
        n = f'{name}_{srv}'
        self.all_state[n] = {}
        self.all_state[n]['out'] = last_out
        self.all_state[n]['rnn'] = copy.deepcopy(self.model_state)
        self.all_state[n]['token'] = copy.deepcopy(self.model_tokens)

    def load_all_stat(self, srv, name):
        logging.info("Loading state for srv: %s, name: %s", srv, name)
        n = f'{name}_{srv}'
        self.model_state = copy.deepcopy(self.all_state[n]['rnn'])
        self.model_tokens = copy.deepcopy(self.all_state[n]['token'])
        return self.all_state[n]['out']

    def reply_msg(self, msg):
        return f'{self.bot}{self.interface} {msg}\n'

    def generate_response(self, message):
        logging.info("Generating Response")
        response = ''
        srv = 'dummy_server'

        msg = message.replace('\\n', '\n').strip()

        x_temp = GEN_TEMP
        x_top_p = GEN_TOP_P
        if ("-temp=" in msg):
            x_temp = float(msg.split("-temp=")[1].split(" ")[0])
            msg = msg.replace("-temp=" + f'{x_temp:g}', "")
        if ("-top_p=" in msg):
            x_top_p = float(msg.split("-top_p=")[1].split(" ")[0])
            msg = msg.replace("-top_p=" + f'{x_top_p:g}', "")
        if x_temp <= 0.2:
            x_temp = 0.2
        if x_temp >= 5:
            x_temp = 5
        if x_top_p <= 0:
            x_top_p = 0
        msg = msg.strip()

        if msg == '+reset':
            out = self.load_all_stat('', 'chat_init')
            self.save_all_stat(srv, 'chat', out)
            return self.reply_msg("Chat reset.")
        elif msg[:8].lower() == '+prompt ':
            response += "Loading prompt..."
            try:
                PROMPT_FILE = msg[8:].strip()
                user, self.bot, self.interface, init_prompt = load_prompt(PROMPT_FILE)
                out = self.run_rnn(self.pipeline.encode(init_prompt))
                self.save_all_stat(srv, 'chat', out)
                response += "Prompt set up."
                gc.collect()
                torch.cuda.empty_cache()
            except:
                response += "Path error."
        elif msg[:5].lower() == '+gen ' or msg[:3].lower() == '+i ' or msg[:4].lower() == '+qa ' or msg[:4].lower() == '+qq ' or msg.lower() == '+++' or msg.lower() == '++':

            if msg[:5].lower() == '+gen ':
                new = '\n' + msg[5:].strip()
                model_state = None
                self.model_tokens = []
                out = run_rnn(pipeline.encode(new))
                self.save_all_stat(srv, 'gen_0', out)

            elif msg[:3].lower() == '+i ':
                msg = msg[3:].strip().replace('\r\n', '\n').replace('\n\n', '\n')
                new = f'''


                # Create a new chat session.



                # Below is an instruction that describes a task. Write a response that appropriately completes the request.

                # Instruction:
{msg}

                # Response:
'''
                chat_session = ChatSession(model, pipeline)
                model_state = None
                self.model_tokens = []
                out = run_rnn(pipeline.encode(new))
                save_all_stat(srv, 'gen_0', out)

            elif msg[:4].lower() == '+qq ':
                new = '\nQ: ' + msg[4:].strip() + '\nA:'
                model_state = None
                self.model_tokens = []
                out = run_rnn(pipeline.encode(new))
                save_all_stat(srv, 'gen_0', out)

            elif msg[:4].lower() == '+qa ':
                out = load_all_stat('', 'chat_init')

                real_msg = msg[4:].strip()
                new = f"{user}{interface} {real_msg}\n\n{self.bot}{self.interface}"
                out = generate_response(pipeline.encode(new))
                save_all_stat(srv, 'gen_0', out)

            elif msg.lower() == '+++':
                try:
                    out = load_all_stat(srv, 'gen_1')
                    save_all_stat(srv, 'gen_0', out)
                except:
                    return

            elif msg.lower() == '++':
                try:
                    out = load_all_stat(srv, 'gen_0')
                except:
                    return

            begin = len(self.model_tokens)
            out_last = begin
            occurrence = {}
            for i in range(FREE_GEN_LEN + 100):
                for n in occurrence:
                    out[n] -= (GEN_alpha_presence + occurrence[n] * GEN_alpha_frequency)
                token = pipeline.sample_logits(
                    out,
                    temperature=x_temp,
                    top_p=x_top_p,
                )
                if token == END_OF_TEXT:
                    break
                if token not in occurrence:
                    occurrence[token] = 1
                else:
                    occurrence[token] += 1

                if msg[:4].lower() == '+qa ':
                    out = run_rnn([token], newline_adj=-2)
                else:
                    out = run_rnn([token])

                xxx = pipeline.decode(self.model_tokens[out_last:])
                if '\ufffd' not in xxx:  # avoid utf-8 display issues
                    response += xxx
                    out_last = begin + i + 1
                    if i >= FREE_GEN_LEN:
                        break
            response += '\n'
            save_all_stat(srv, 'gen_1', out)

        else:
            if msg.lower() == '+':
                try:
                    out = self.load_all_stat(srv, 'chat_pre')
                except:
                    return
            else:
                out = self.load_all_stat(srv, 'chat')
                msg = msg.strip().replace('\r\n', '\n').replace('\n\n', '\n')
                new = f"{self.user}{self.interface} {msg}\n\n{self.bot}{self.interface}"
                out = self.run_rnn(pipeline.encode(new), newline_adj=-999999999)
                self.save_all_stat(srv, 'chat_pre', out)

            begin = len(self.model_tokens)
            out_last = begin
            response += f'{self.bot}{self.interface}'
            occurrence = {}
            for i in range(999):
                if i <= 0:
                    newline_adj = -999999999
                elif i <= CHAT_LEN_SHORT:
                    newline_adj = (i - CHAT_LEN_SHORT) / 10
                elif i <= CHAT_LEN_LONG:
                    newline_adj = 0
                else:
                    newline_adj = min(3, (i - CHAT_LEN_LONG) * 0.25)  # MUST END THE GENERATION

                for n in occurrence:
                    out[n] -= (GEN_alpha_presence + occurrence[n] * GEN_alpha_frequency)
                token = pipeline.sample_logits(
                    out,
                    temperature=x_temp,
                    top_p=x_top_p,
                )

                if token not in occurrence:
                    occurrence[token] = 1
                else:
                    occurrence[token] += 1

                out = self.run_rnn([token], newline_adj=newline_adj)
                out[END_OF_TEXT] = -999999999  # disable 

                xxx = pipeline.decode(self.model_tokens[out_last:])
                if '\ufffd' not in xxx:  # avoid utf-8 display issues
                    response += xxx
                    out_last = begin + i + 1

                send_msg = pipeline.decode(self.model_tokens[begin:])
                if '\n\n' in send_msg:
                    send_msg = send_msg.strip()
                    break

            self.save_all_stat(srv, 'chat', out)
            return response.strip()

########################################################################################################
def process_message(user_message,session_id,socketio):
    # Process the user_message with your chatbot using the generate_response function.
    logging.info("Processing message: %s", user_message)
    cs = ChatSession(model, pipeline)   
    response = cs.generate_response(user_message)
    logging.info("response message: %s", response)
    return response
if CHAT_LANG == 'English':
    HELP_MSG = '''Commands:
say something --> chat with bot. use \\n for new line.
+ --> alternate chat reply
+reset --> reset chat

+gen YOUR PROMPT --> free single-round generation with any prompt. use \\n for new line.
+i YOUR INSTRUCT --> free single-round generation with any instruct. use \\n for new line.
+++ --> continue last free generation (only for +gen / +i)
++ --> retry last free generation (only for +gen / +i)

Now talk with the bot and enjoy. Remember to +reset periodically to clean up the bot's memory. Use RWKV-4 14B (especially https://huggingface.co/BlinkDL/rwkv-4-raven) for best results.
'''
elif CHAT_LANG == 'Chinese':
    HELP_MSG = f'''指令:
直接输入内容 --> 和机器人聊天（建议问机器人问题），用\\n代表换行，必须用 Raven 模型
+ --> 让机器人换个回答
+reset --> 重置对话，请经常使用 +reset 重置机器人记忆

+i 某某指令 --> 问独立的问题（忽略聊天上下文），用\\n代表换行，必须用 Raven 模型
+gen 某某内容 --> 续写内容（忽略聊天上下文），用\\n代表换行，写小说用 testNovel 模型
+++ --> 继续 +gen / +i 的回答
++ --> 换个 +gen / +i 的回答

作者：彭博 请关注我的知乎: https://zhuanlan.zhihu.com/p/603840957
如果喜欢，请看我们的优质护眼灯: https://withablink.taobao.com

中文 Novel 模型，可以试这些续写例子（不适合 Raven 模型）：
+gen “区区
+gen 以下是不朽的科幻史诗长篇巨著，描写细腻，刻画了数百位个性鲜明的英雄和宏大的星际文明战争。\\n第一章
+gen 这是一个修真世界，详细世界设定如下：\\n1.
'''
elif CHAT_LANG == 'Japanese':
    HELP_MSG = f'''コマンド:
直接入力 --> ボットとチャットする．改行には\\nを使用してください．
+ --> ボットに前回のチャットの内容を変更させる．
+reset --> 対話のリセット．メモリをリセットするために，+resetを定期的に実行してください．

+i インストラクトの入力 --> チャットの文脈を無視して独立した質問を行う．改行には\\nを使用してください．
+gen プロンプトの生成 --> チャットの文脈を無視して入力したプロンプトに続く文章を出力する．改行には\\nを使用してください．
+++ --> +gen / +i の出力の回答を続ける．
++ --> +gen / +i の出力の再生成を行う.

ボットとの会話を楽しんでください。また、定期的に+resetして、ボットのメモリをリセットすることを忘れないようにしてください。
'''



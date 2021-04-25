import gym
from MapleEnv_v4 import MapleEnv
from stable_baselines3 import PPO
from maplewrapper import wrapper
#from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, register_policy
import time

MlpPolicy = ActorCriticPolicy

model_name = "model_new.sav"

from win32con import *
from win32api import *
from win32gui import *
from win32process import *

A_KEY = 0x001E0000
D_KEY = 0x00200000
S_KEY = 0x001f0000
LEFT_KEY = 0x014b0000
RIGHT_KEY = 0x014d0000

ms_hwnd = []
target_pid = 0
def windowEnumerationHandler(hwnd, top_windows):
    global target_pid
    # top_windows.append((hwnd, win32gui.GetWindowText(hwnd)))
    # tid,pid = GetWindowThreadProcessId(hwnd)
    # print(hwnd)
    text = GetWindowText(hwnd)
    if "MapleStory" in text:
        ms_hwnd.append(hwnd)
        tid,target_pid = GetWindowThreadProcessId(hwnd)
        print(target_pid)
        print("set target_pid")
    # print(GetWindowModuleFileName(hwnd))

def windowEnumerateSearch(hwnd, top_windows):
    # print("target_pid {}".format(target_pid))
    tid,pid = GetWindowThreadProcessId(hwnd)
    if pid == target_pid:
        print("found! {}".format(hwnd))
        print(pid)

# found! 793002
# 25520
# found! 4263014
# 25520
# found! 2299890
# 25520

top_windows = []
EnumWindows(windowEnumerationHandler, top_windows)
print(ms_hwnd)
EnumWindows(windowEnumerateSearch, top_windows)

# from IPython import embed
# embed()
# hwnd = 528792

# PostMessage(ms_hwnd[0], 0x401, 0x540, 0x1)

for i in range(0, 10):
    PostMessage(ms_hwnd[0], WM_KEYDOWN, 0, D_KEY)
    time.sleep(1)

from IPython import embed
embed()

# for i in range(0, 0xffffffff):
#     PostMessage(ms_hwnd[0], 260, 0, i)
#     print("sending key: {:x}".format(i))
#     time.sleep(0.1)
#     PostMessage(ms_hwnd[0], 260, 0, D_KEY)
#     time.sleep(0.1)


# with wrapper("charlie",["Snail", "Blue Snail", "Shroom", "Red Snail"]) as w:
#     INSPECT = 0
#     if INSPECT:
#         w.inspect('portals')
#     else:
#         env = MapleEnv(w)
#         model = PPO(MlpPolicy, env, verbose=3)
#         try:
#             f = open(model_name, "r")
#             f.close()
#             model = model.load(model_name)
#             model.set_env(env)
#             # from IPython import embed
#             # embed()
#             # model.load()
#             print("loaded")
#         except FileNotFoundError:
#             pass
#         print("starting in 3 seconds..")
#         time.sleep(3)
#         try:
#             pass
#             model.learn(total_timesteps=500000, log_interval=1)
#             model.save(model_name)
#         except KeyboardInterrupt:
#             model.save(model_name)
#             print("[~] Saved")
        # except Exception:
        #     print("[!] Crashed due to unknown exception, saving model with different name to not overwrite any previous model...")
        #     model.save(model_name + '.bak')

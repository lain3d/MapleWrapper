import gym
from MapleEnv_v4 import MapleEnv
from stable_baselines3 import PPO
from maplewrapper import wrapper
#from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, register_policy
import time

MlpPolicy = ActorCriticPolicy

model_name = "model73_cont.sav"

with wrapper("rich",["Snail", "Blue Snail", "Shroom", "Red Snail"]) as w:
    INSPECT = 0
    if INSPECT:
        w.inspect('portals')
    else:
        env = MapleEnv(w)
        model = PPO(MlpPolicy, env, verbose=3)
        try:
            f = open(model_name, "r")
            f.close()
            model = model.load(model_name)
            model.set_env(env)
            # from IPython import embed
            # embed()
            # model.load()
            print("loaded")
        except FileNotFoundError:
            pass
        print("starting in 3 seconds..")
        time.sleep(3)
        try:
            pass
            model.learn(total_timesteps=500000, log_interval=1)
            model.save(model_name)
        except KeyboardInterrupt:
            model.save(model_name)
            print("[~] Saved")
        except Exception:
            print("[!] Crashed due to unknown exception, saving model with different name to not overwrite any previous model...")
            model.save(model_name + '.bak')

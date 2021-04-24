import gym
import time
from gym import spaces
from maplewrapper import wrapper
import numpy as np
import pydirectinput
pydirectinput.FAILSAFE = False
import cv2
import random

np.set_printoptions(precision=3)

MAX_MOBS = 10
SPEEDUP = 3
EPISODE_TIME = 300 #600/SPEEDUP
print("Episode Time: {}".format(EPISODE_TIME))

NORMALIZE_PLAYER_X = float(806)
NORMALIZE_PLAYER_Y = float(629)

NORMALIZE_MOB_X = 806
NORMALIZE_MOB_Y = 629

NORMALIZE_CONNECTS_X = 629
NORMALIZE_CONNECTS_Y = 629

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def get_distance(o1, o2):
    x = o1[0] - o2[0]
    y = o1[3] - o2[3]
    d = pow(pow(x,2) + pow(y,2), 0.5)
    return d

class MapleEnv(gym.Env):
    """
    Description:
        Gym environment of MapleStory v.90 and below using extracted information from maplewrapper.
        See https://github.com/vinmorel/MapleWrapper
    Observation:
        Type: Dict "MapleWrapper" : box(4)
        Num     Observation               Min                     Max
        1       Player X1                 0                       825
        2       Mob X1 (1)                0                       825
        3       Player Facing Direction   0                       1
        4       Attacked                  0                       1

    Actions:
        Type: Discrete(4)
        Num   Action
        0     Walk left
        1     Walk right
        2     Attack 1
        3     Attack 2
    Reward:
        Reward is the sum of gained exp minus health damage, mp consumption and time penalities
    Starting State:
        All observations are intialized according to game information
    Episode Termination:
        Episode terminates every 10 minutes 
    """
    
    metadata = {'render.modes': ['human']}


    def __init__(self,w):
        pydirectinput.PAUSE = 0.0

        self.w = w
        self.lvl, self.max_hp, self.max_mp, self.max_exp = self.w.get_basestats()
        self.B_X = 850 # Bounding box max X 

        array_size = 32 #24
        self.Min = np.array([0] * array_size,dtype=np.float32)
        self.Max = np.array([self.B_X] * 2 + [1] * (array_size-2) ,dtype=np.float32)

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(self.Min, self.Max, dtype=np.float32)

        self.state = None
        self.done = None
        self.penalty = None

        self.actions_d = {
            '0' : 'left',
            '1' : 'right',
            '2' : 'a',
            # '3' : 'shift',
            # '3' : 'd,up',
            '3' : 'up',
            # '5' : 'd',
            '4' : 'd,right,up',
            '5' : 'd,left,up',
            # '6' : 'down,d',
            'hp' : 't',
            'mp' : 'y',
            'pickup' : 's',
        }

        self.reward_threshold = 20.0
        self.trials = 200
        self.steps_counter = 0
        self.id = "MapleBot"
        self.facing = None
        self.random_t_keydown = 0.01
        self.portal_close = False
        self.close_mobs = False
        self.last_action = None
        self.connect_close = False

    def step(self,action):
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action : int
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob : List[int]
                an environment-specific object representing your observation of
                the environment.
            reward : float
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over : bool
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info : Dict
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """ 
        self.take_action(action)
        self.new_p_state, new_stats = self.get_partial_state()
        self.get_reward(self.stats,new_stats)

        # Determine if attacked
        if new_stats[1] < self.stats[1]:
            self.attacked = np.array([1])
        else :
            self.attacked = np.array([0])
        self.stats = new_stats

        # Determine facing dirzzzection 
        if action == 0:
            self.facing = np.array([action])
        if action == 1:
            self.facing = np.array([action])

        self.state = np.concatenate((self.new_p_state,self.facing,self.attacked))

        # self.render()

        # heal if necessary 
        if new_stats[1]/self.max_hp < 0.5:
            self.take_action('hp')
        # mp if necessary
        if new_stats[2]/self.max_mp < 0.2:
            self.take_action('mp')
        # random pickup
        # if np.random.binomial(1,0.3):
        #     self.take_action('pickup')
        # terminate episode if t 
        self.current_time = time.time()
        if int(self.current_time - self.start_time) >= EPISODE_TIME:
            self.done = 1

        return self.state, self.reward, self.done, {}

    def reset(self):
        self.take_action(1)
        self.facing = np.array([1])
        self.attacked = np.array([0])
        self.p_state, self.stats = self.get_partial_state()
        self.state = np.concatenate((self.p_state,self.facing,self.attacked))
        self.done = 0
        self.start_time = time.time()
        return self.state
        
    def get_partial_state(self):
        self.player, stats, self.mobs, self.connects, self.portals = self.w.observe()
        self.portal_close = False
        # print("portals: {}".format(self.portals))
        for portal in self.portals:
            d = get_distance(self.player, portal)
            # print("[PORTAL] Distance to {} from player {}: {}".format(portal, self.player, d))
            if d < 100:
                self.portal_close = True
        
        MAX_CONNECTS = 4
        # todo actually sort
        if len(self.connects):
            for connect in self.connects:
                d = get_distance(self.player, connect)
                print("[CONNECT] Distance to {} from player {}: {}".format(connect, self.player, d))
                if d < 100:
                    self.connect_close = True
            self.connects = self.connects[:, [0,1]]
            self.connects = np.reshape(self.connects, (1, self.connects.size))[0]

            is_y = False
            for i in range(0, len(self.connects)):
                divisor = NORMALIZE_CONNECTS_X
                if is_y:
                    divisor = NORMALIZE_CONNECTS_Y
                self.connects[i] = self.connects[i] / divisor
            while len(self.connects) < MAX_CONNECTS*2:
                self.connects = np.append(self.connects, [1]) # normalized
                # self.connects = normalized(self.connects,order=2)[0]
        else:
            self.connects = np.ones(8)
            # If no connects visible there are no close connects
            self.connects_close = False

        self.player = np.array([self.player[1:3]])
        # self.player = np.array([self.player[2]])
        # self.mobs = self.sort_mobs(self.mobs,self.player)
        # print(self.player)
        self.mobs = self.sort_mobs_v2(self.mobs,self.player[0][0], self.player[0][1])

        # from IPython import embed
        # embed()
        self.player[0][0] = self.player[0][0] / NORMALIZE_PLAYER_Y
        self.player[0][1] = self.player[0][1] / NORMALIZE_PLAYER_X
        # from IPython import embed
        # embed()
        print("actual: {} {}".format(self.player[0][0], self.player[0][1]))
        state = np.concatenate((self.player[0], self.mobs, self.connects))
        return state, stats

    def sort_mobs(self,mob_coords,player_x1):
        if len(mob_coords) == 0:
            mobs_X1 = np.full(1,410 - player_x1)
        else:
            mob_coords = sorted(mob_coords[:,2] - player_x1, key=abs)
            #print(mob_coords)
            mobs_X1 = mob_coords[:1] # max 1 slot
            n_mobs = len(mobs_X1)            

        return mobs_X1

    def sort_mobs_v2(self,mob_coords,player_x1,player_y1):
        if len(mob_coords) == 0:
            # mobs_X1 = np.full(1,410 - player_x1)
            mob_coords = np.zeros((10,2))
            mob_coords.fill(1) # normalized
        else:
            mob_coords = mob_coords[:, [1,2]]
            # sort by total distance to player
            mob_coords_k = {}
            self.close_mobs = False
            for i,coord in enumerate(mob_coords):
                x = coord[1] - player_x1
                y = coord[0] - player_y1
                d = pow(pow(x, 2) + pow(y, 2), 0.5)
                mob_coords_k[i] = d
                if d < 60:
                    self.close_mobs = True
                # print("distance from {} to {},{} = {}".format(coord,player_x1,player_y1,d))

            mob_coords_k = {k: v for k, v in sorted(mob_coords_k.items(), key=lambda item: item[1])}
            mob_coords_new = None
            for k in mob_coords_k.keys():
                coord = mob_coords[k,None]
                coord[0][0] = coord[0][0] / NORMALIZE_MOB_Y #- player_y1
                coord[0][1] = coord[0][1] / NORMALIZE_MOB_X #- player_x1
                if not isinstance(mob_coords_new, np.ndarray):
                    mob_coords_new = coord
                else:
                    mob_coords_new = np.concatenate((mob_coords_new,coord))

            # from IPython import embed
            # embed()

            mob_coords = mob_coords_new

        # concatenate missing mobs
        
        needed_rows = MAX_MOBS - mob_coords.shape[0]
        # todo trim too many mobs
        # todo normalize this
        add_on = np.zeros((needed_rows,2))
        add_on.fill(1) # normalized

        mob_coords = np.concatenate((mob_coords, add_on))
        mob_coords = np.reshape(mob_coords, (1,mob_coords.size))[0]
        # mob_coords = normalized(mob_coords,order=1)[0]
        # from IPython import embed
        # embed()
        return mob_coords

    def take_action(self,action):
        #print(str(action))
        # pydirectinput.keyDown("s")
        self.last_action = action
        if action != None:
            if 'p' in str(action):
                pydirectinput.press(self.actions_d[str(action)])
                return None
            else:
                s = self.actions_d[str(action)]
                # if s == "left" or s == "right" or s == "up" or s == "d,right" or s == "d,left":
                #     # self.random_t_keydown = random.random() * 1
                #     self.random_t_keydown = 0.4
                #     # print("z")
                # else:
                #     self.random_t_keydown = 0.4

                if 'up' in s and self.portal_close:
                    print("portal close!")
                    return

                print("taking action {} : {}".format(s, self.portal_close))
                self.random_t_keydown = (random.random() * 1.4) / SPEEDUP #0.09
                
                key = self.actions_d[str(action)]
                keys = key.split(",")
                # print(keys)
                pydirectinput.keyDown('s')
                for i,k in enumerate(keys):
                    pydirectinput.keyDown(k)
                    if i == 2:
                        time.sleep(0.8 / SPEEDUP)
                    # time.sleep(0.1 / SPEEDUP)

                time.sleep(self.random_t_keydown / SPEEDUP)

                for k in keys:
                    pydirectinput.keyUp(k)

                pydirectinput.keyUp('s')
        

    def get_reward(self,old_stats,new_stats):
        old_stats = np.array(old_stats)
        new_stats = np.array(new_stats)
        self.delta = new_stats - old_stats
        self.d_lvl, self.d_hp, self.d_mp, self.d_exp = self.delta
        
        # Default penality 
        self.reward = -0.1

        # penalty if attack when no mobs closeby
        # print("close_mobs: {} last_action: {}".format(self.close_mobs, self.last_action))
        if not self.close_mobs and self.last_action == 2:
            self.reward -= 0.1
            print("ATTACK, NO CLOSE MOBS")

        # check if connect not close and last action was something other than left or right
        if self.connect_close and (self.last_action != 0 or self.last_action != 1): 
            self.reward -= 0.1
            print("[CONNECT] unneeded action")

        # Penality if too close to map borders
        # print(self.state)
        # if self.new_p_state[1] < 125 or self.new_p_state[1] > 744:
        #     self.reward -= 0.1
        #     print("TOO CLOSE!")

        # Penalty if too close to portal
        # todo how to do this actually?
        # if self.portal_close:
            # self.reward -= 0.5

        # Reward if mob hit
        if self.w.get_hitreg() == True:
            self.reward += 0.5
            print("GOOD HIT!")
        # reward if exp gains 
        if self.d_exp > 0 :
            self.reward += 0.5 + (self.d_exp/self.max_exp) * 250
            print("exp reward")
        # re-extract base stats if level up
        if self.d_lvl >= 1:
            self.lvl, self.max_hp, self.max_mp, self.max_exp = self.w.get_basestats()
            print("LEVEL UP!")
        return self.reward

    def render(self, mode='human',close=False):
        self.w.inspect('frame')

    def close(self):
        pass

if __name__ == "__main__":   
    with wrapper("mrblue",["Horny Mushroom", "Zombie Mushroom"]) as w:
        env = MapleEnv(w)
        env.reset()

        while True:
            env.step(action=None)
            #print(env.w.get_hitreg())
            # print(env.new_p_state[1])
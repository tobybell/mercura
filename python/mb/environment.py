import math
import random
import subprocess
import os.path as path

import torch

class Environment(object):
    def __init__(self):
        dirname = path.dirname(__file__)
        exec_path = path.join(dirname, '../../c/interactive')
        self.sp = subprocess.Popen([exec_path],
                                   stdout=subprocess.PIPE,
                                   stdin=subprocess.PIPE,
                                   stderr=subprocess.PIPE)

    def get_state(self):
        return self.step(0, 0)

    def step(self, action, duration=60.0):
        thrust = 0.1 * action - 0.1
        cmd = "a {} {}\n".format(thrust, duration)
        self.sp.stdin.write(cmd.encode("utf-8"))
        self.sp.stdin.flush()
        out = self.sp.stdout.readline().decode("utf-8").strip()
        pv = torch.tensor(list(map(float, out.split(','))))
        return pv

    def reset(self, *pv):
        cmd = "r {}\n".format(' '.join(map(str, pv)))
        self.sp.stdin.write(cmd.encode("utf-8"))
        self.sp.stdin.flush()
        out = self.sp.stdout.readline().decode("utf-8").strip()
        pv = torch.tensor(list(map(float, out.split(','))))
        return pv

    def reset_rand(self):
        r = 7255000 + random.random() * 1e9
        v = math.sqrt(3.9860044188e14 / r)
        t = random.random() * 2 * math.pi
        return self.reset(r * math.cos(t), r * math.sin(t), 0, -v * math.sin(t), v * math.cos(t), 0)

    def reset_geo(self):
        return self.reset(42241095.67708342, 0, 0, 0.017776962751035255, 3071.8591633446, 0)

    def reset_leo(self):
        return self.reset(7255000.0, 0, 0, 0, 7412.2520611297, 0)

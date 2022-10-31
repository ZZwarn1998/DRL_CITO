import sys
import time
import numpy as np
import math
from RType import RType
from TClass import TClass
from Datum import Datum
import torch
import copy
import numba as nb


class Env(object):
    def __init__(self,
                 ncls,
                 routes,
                 clpmat,
                 id2cls,
                 edges):
        # basic attributes about a static or dynamic program structure
        self.ncls = ncls
        self.routes = routes
        self.clpmat = clpmat
        self.id2cls = id2cls
        self.edges = edges
        self.chgd_clpmat = [[0.0] * self.ncls for _ in range(self.ncls)]
        self.init_chgd_clpmat()
        self.act_spc = np.array([id for id in range(self.ncls)], dtype=np.float32)

        # recoding parameters
        self._act = np.float32(-1)  # reset _act = -1
        self.state = np.array([-1 for i in range(self.ncls)], dtype=np.float32) # reset state = np.array([-1 for i in range(self.ncls)], dtype=np.float32)
        self.steps = 0  # reset steps = 0
        self.selected = [False for i in range(self.ncls)] # reset self.selected = [False for i in range(self.ncls)]
        self.cost = 0  # reset cost = 0
        self.chgd_cost = 0  # reset chgd_cost = 0
        self.ifBuiltGSs = [False for i in range(self.ncls)] # reset builtGSs = [False for i in range(self.ncls)]
        self.infoGSs = []  # reset infoGSs = []
        self.infoSSs = []  # reset infoSSs = []
        self.NGSs = 0  # reset NGSs = 0
        self.NSSs = 0  # reset NSSs = 0
        self.NAttrDeps = 0  # reset NAttrDeps = 0
        self.NMethDeps = 0  # reset NMethDeps = 0
        self.NDeps = 0  # reset NDeps = 0
        self.order = []  # reset order = []
        self.getBest = False  # reset getBest = False
        self.profits = set()  # restart profits = set()
        self.ifstd = False  # restart ifstd = False
        self.rewards = 0  # reset rewards = 0

        # fixed attributes
        self.a = 5.0
        self.b = 0.8
        self.min_val = -1.0
        self.max_val = 5.0

        # best result
        self.min_chgd_cost = sys.maxsize  # reset if we start a new training
        self.min_cost = sys.maxsize  # reset if we start a new training
        self.min_order = []  # reset if we start a new training
        self.min_datum = Datum()  # reset if we start a new training

        # calculate running time
        self.st = time.clock()

        # record
        self.profit = 0

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @nb.jit()
    def init_chgd_clpmat(self):
        for edge in self.edges:
            fromid = edge.getfromid()
            toid = edge.gettoid()
            self.chgd_clpmat[fromid][toid] = 5 * self.clpmat[fromid][toid] \
                if edge.gettype() == RType.AG or edge.gettype == RType.I \
                else self.clpmat[fromid][toid]
            # print(edge.gettype() , self.clpmat[fromid][toid] == self.chgd_clpmat[fromid][toid], self.clpmat[fromid][toid], self.chgd_clpmat[fromid][toid])
        return

    def reset(self):
        """
        We need to reset recording parameters when we start a new round of game in a training.

        :return:
        """
        self._act = np.float32(-1)  # reset _act = -1
        self.state = np.array([-1 for i in range(self.ncls)],
                              dtype=np.float32)  # reset state = np.array([-1 for i in range(self.ncls)], dtype=np.float32)
        self.steps = 0  # reset steps = 0
        self.selected = [False for i in range(self.ncls)]  # reset self.selected = [False for i in range(self.ncls)]
        self.cost = 0  # reset cost = 0
        self.chgd_cost = 0  # reset chgd_cost = 0
        self.ifBuiltGSs = [False for i in range(self.ncls)]  # reset builtGSs = [False for i in range(self.ncls)]
        self.infoGSs = []  # reset infoGSs = []
        self.infoSSs = []  # reset infoSSs = []
        self.NGSs = 0  # reset NGSs = 0
        self.NSSs = 0  # reset NSSs = 0
        self.NAttrDeps = 0  # reset NAttrDeps = 0
        self.NMethDeps = 0  # reset NMethDeps = 0
        self.NDeps = 0  # reset NDeps = 0
        self.order = []  # reset order = []
        self.getBest = False  # reset getBest = False
        self.rewards = 0  # reset rewards = 0

    def restart(self):
        """
        We need to restart our environment when we start a new training.

        :return:
        """
        self.reset()
        self.min_chgd_cost = sys.maxsize  # reset if we start a new training
        self.min_cost = sys.maxsize  # reset if we start a new training
        self.min_order = []  # reset if we start a new training
        self.min_datum = Datum()  # reset if we start a new training
        self.st = time.clock()
        self.profits = set()
        self.ifstd = False

    @nb.jit()
    def step(self, act):
        """
        We execute a new action with current state and obtain corresponding next state, reward and the
        symbol showing whether we finish the game or not.

        :param int act: Selected action
        :return: state
        :rtype: np.ndarray
        """
        state = np.zeros_like(self.state, dtype=np.float32)
        state[:] = self.state
        reward = 0.0
        done = 0
        state_ = np.zeros_like(state, dtype=np.float32)
        state_[:] = copy.deepcopy(state)

        if self.selected[int(act)]:
            state_[int(act)] = np.float32(-10)
            self.steps += 1
            done = 1
            reward = self.calReward(act, punish=True)
        else:
            state_[int(act)] = np.float32(self.steps)
            self.steps += 1
            if self.steps == self.ncls:
                done = 1
            reward = self.calReward(act, punish=False)
        # print(self.state, state_)

        # update
        self.state = np.zeros_like(state_, dtype=np.float32)
        self.state[:] = state_
        self._act = act
        self.order.append(int(act))
        self.selected[int(act)] = True

        reward = np.float32(reward)
        done = np.float32(done)

        return state, state_, reward, done

    @nb.jit()
    def standardize(self, profit):
        uni = torch.from_numpy(np.unique(list(self.profits))).to(self.device)
        std_profit = (profit - uni.mean()) / uni.std()
        return std_profit

    @nb.jit()
    def map_func(self, a, b, x):
        return a * (1 / (1 + math.exp(- 1 / b * x)))

    @nb.jit()
    def calReward(self,
                  act: int,
                  punish: bool = False):
        # func = lambda a, b, x: a * (1 / (1 + math.exp(- 1 / b * x)) - 0.5)
        if punish:
            return self.min_val
        else:
            """Question: Should we consider the effect of only one action while we are calculating the reward? What 
            about considering the effect of actions that we chose in the past?
            """
            reward = 0
            profit = 0
            profit = self.calProfit(act)
            self.profits.add(profit)
            # record
            self.profit = profit

            if not self.ifstd:
                reward = self.rewards + self.map_func(self.a, self.b, profit)
            else:
                reward = self.rewards + self.map_func(self.a, self.b, self.standardize(profit))

            self.rewards = reward

            if self.steps == self.ncls:
                if self.min_cost >= self.cost:
                    self.min_order = copy.deepcopy(self.order)
                    self.min_cost = self.cost
                    self.min_chgd_cost = self.chgd_cost
                    appear = time.clock() - self.st
                    self.min_datum = Datum(order=copy.deepcopy(self.order),
                                           ocplx=float(self.min_cost),
                                           appear=appear,
                                           GStubs=int(self.NGSs),
                                           SStubs=int(self.NSSs),
                                           deps=int(self.NDeps),
                                           attrdeps=int(self.NAttrDeps),
                                           methdeps=int(self.NMethDeps),
                                           infoGStubs=copy.deepcopy(self.infoGSs),
                                           infoSStubs=copy.deepcopy(self.infoSSs)
                                           )
                    self.getBest = True
                    # return self.max_val
                    return reward + self.max_val
            return reward

    @nb.jit()
    def calProfit(self, 
                  act: int):
        acc_chgd_cost = 0.0
        acc_cost = 0.0
        acc_profit = 0.0
        act = int(act)
        
        for i in range(self.ncls):
            if self.routes[act][i] > 0:
                if self.selected[i]:
                    continue
                acc_chgd_cost = acc_chgd_cost + self.chgd_clpmat[act][i]
                acc_cost = acc_cost + self.clpmat[act][i]
                from_cls = self.id2cls.get(act)
                to_cls = self.id2cls.get(i)
                from_cls_name = from_cls.getname()
                to_cls_name = to_cls.getname()
                
                if not self.ifBuiltGSs[i]:
                    self.ifBuiltGSs[i] = True
                    self.NGSs = self.NGSs + 1
                    self.infoGSs.append(str(i) + ":" + to_cls_name)
                    
                self.NSSs = self.NSSs + 1
                self.infoSSs.append(str(i) + "(" + str(act) + ")"+ 
                                    ":" + to_cls_name + "(" + from_cls_name + ")")
                
                if to_cls_name in from_cls.getattrdeps():
                    self.NDeps = self.NDeps + int(from_cls.getattrdeps().get(to_cls_name))
                    self.NAttrDeps = self.NAttrDeps + int(from_cls.getattrdeps().get(to_cls_name))

                if to_cls_name in from_cls.getmethdeps():
                    self.NDeps = self.NDeps + int(from_cls.getmethdeps().get(to_cls_name))
                    self.NMethDeps = self.NMethDeps + int(from_cls.getmethdeps().get(to_cls_name))

            if self.routes[i][act] > 0 and (not self.selected[i]):
                acc_profit = acc_profit + self.chgd_clpmat[i][act]

        self.chgd_cost = self.chgd_cost + acc_chgd_cost
        self.cost = self.cost + acc_cost

        return acc_profit - acc_chgd_cost

    def getActionSpace(self):
        return self.act_spc

    def getNcls(self):
        return self.ncls

    def getBestResult(self):
        return self.min_datum


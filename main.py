import os
import json
import datetime
from Datum import Datum
from InfoLoader import InfoLoader
from Env import Env
from DQN import DQN


def init(pg):
    """
    The function generates initial folder used for storing date.

    :param pg: str
    :return: save_path: str
    """
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_path = os.path.join(os.path.join(os.path.split(os.path.abspath(__file__))[0], "exp_data"), pg + "-" + date)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(os.path.join(save_path, "exps")):
        os.makedirs(os.path.join(save_path, "exps"))

    return save_path, date


def datum2json(datum):
    """
    The function converts class {Datum} into json {string}.

    :param datum: Datum
    :return: json_datum: str
    """
    dirs = dir(datum)
    attrs = list(filter(lambda x: not x.startswith("__"), dirs))
    values = [getattr(datum, attr) for attr in attrs]
    dic = dict(zip(attrs, values))
    json_datum = json.dumps(dic)
    # print(json_datum)
    return json_datum


def json2datum(json_datum):
    """
    The function converts json {string} into class {Datum}.

    :param json_datum: str
    :return: datum: Datum
    """
    dic = json.loads(json_datum)
    datum = Datum()
    for key, val in dic.items():
        if hasattr(datum, key):
            setattr(datum, key, val)
            # print(key, getattr(datum, key))
    return datum


def write_data(data, data_path):
    """
    Write collected data into a json file,

    :param data: Datum list
    :param data_path: str
    :return:
    """
    with open(data_path, "w+") as fw:
        for datum in data:
            json.dump(datum2json(datum), fw)
            fw.write("\n")
        fw.close()
    return


def read_data(data_path):
    """Regenerate Datum list by reading a json file.

    :param data_path: str
    :return: data: Datum list
    """
    data = []
    with open(data_path, "r") as fr:
        json_data = fr.readlines()
        for json_datum in json_data:
            datum = json2datum(json.loads(json_datum))
            data.append(datum)
        fr.close()
    return data


def info2env(pg, mode=0):
    infopath = os.path.join(os.path.join(os.path.split(os.path.abspath(__file__))[0], "infoAbtPg"), pg)
    loader = InfoLoader(infopath, mode)
    ncls = loader.ncls
    routes = loader.routes
    clpmat = loader.clpmat
    edges = loader.edges
    id2cls = loader.id2cls
    return Env(ncls, routes, clpmat, id2cls, edges)


def print_best_datum(index, datum, t):
    order = datum.order
    cost = datum.ocplx
    GStubs = datum.GStubs
    SStubs = datum.SStubs
    if order is not None:
        out_dict = {"no": "No." + str(index) + " " + pg,
                    "t": "%.0fs" % t,
                    "order": ",".join(map(str, order)),
                    "cost": "%.4f" % cost,
                    "gs": str(GStubs),
                    "ss": str(SStubs)}

        print("{0[no]:-^50}\n"
              ">Running Time:{0[t]:<50}\n"
              ">Order:{0[order]:<50}\n>"
              "Cost:{0[cost]:<50}\n"
              ">GStubs:{0[gs]:<50}\n"
              ">SStubs:{0[ss]:<50}".format(out_dict))
    else:
        out_dict = {"no": "gNo." + str(index) + " " + pg,
                    "t": "None",
                    "order": "None",
                    "cost": "None",
                    "gs": "None",
                    "ss": "None"}

        print("{0[no]:-^50}\n"
              ">Running Time:{0[t]:<50}\n"
              ">Order:{0[order]:<50}\n>"
              "Cost:{0[cost]:<50}\n"
              ">GStubs:{0[gs]:<50}\n"
              ">SStubs:{0[ss]:<50}".format(out_dict))


def run(pg, path, mode, times, rounds, date):
    # Environment initialization
    env: Env = info2env(pg, mode)
    # learning rate
    gamma: float = 0.95
    # period
    plotting_period: int = 1000
    update_period: int = 100
    # epsilon-greedy
    max_epsilon: float = 1.0
    min_epsilon: float = 0.001
    x: float = 0.01
    # PER Parameters
    batch_size = 64
    memory_size: int = 20000
    alpha: float = 0.6
    beta: float = 0.4
    prior_eps: float = 1e-10
    n_step: int = 3
    # collector of best data
    best_data = []

    for index in range(times):
        print(pg + "_" +str(index))
        Dqn = DQN(env,  memory_size=memory_size, batch_size=batch_size,
                  update_period=update_period, gamma=gamma, max_epsilon=max_epsilon,
                  min_epsilon=min_epsilon, x=x, alpha=alpha, beta=beta, prior_eps=prior_eps, n_step=n_step)
        data, t = Dqn.run(pg, index, date, rounds, plotting_period)
        data.append(Datum(appear=t))
        best_datum = data[-1] if len(data) == 1 else data[-2]
        best_data.append(best_datum)
        data_path = os.path.join(os.path.join(path, "exps"), f"exp_{index}.txt")
        write_data(data, data_path)
        print_best_datum(index, best_datum, t)
        print(">number:", "%d" % (len(data) - 1))
        print(">appear:", "%.0fs" % best_datum.appear)
        print("-" * 50)
        env.restart()
    # record all best results
    best_data_path = os.path.join(path, f"best.txt")
    write_data(best_data, best_data_path)
    return


if __name__ == "__main__":
    # pgs = ["elevator", "SPM", "ATM", "daisy", "ANT", "email_spl", "BCEL", "DNS", "notepad_spl"]
    # pgs = ["DNS", "notepad_spl", "BCEL"]
    pgs = ["test"]  # programs (you can see more programs in ./infoAbtPg)
    mode = 0  # mode of analysis (static:0, dynamic:1)
    times = 1  # the number of experiments per program
    rounds = 3000  # the number of rounds per experiment

    for pg in pgs:
        print(pg)
        path, date = init(pg)
        run(pg, path, mode, times, rounds, date)

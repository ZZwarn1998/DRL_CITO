# DRL_CITO
## Intro
&emsp;When we test a project, we will undergo a series of testing processes such as unit testing, integrated testing, system testing and so on. The problem of the generation of class integration test order occurs between unit testing and integrated testing. Because of the intricate relationships among classes and the need of creating testing stubs, different integrated testing orders may have different test costs. Therefore, it is of great importance for us to find a way to generate class integration test order whose cost is really low. Considering that Deep Reinforcement Learning is a way combining the advantage of Deep Learning and Reinforcement Learning, and have excellent ability of status-sensing and decision-making, I proposed **DRL_CITO**, a project applying Deep Reinforce Learning to the generation of class integration test order. Compared with other methods, **DRL_CITO** is capable of generating better class integration test orders.

## Usage 
### Environment configuration 
&emsp;Anaconda is recommended to configure virtual environment. When you create a virtual environment, move to the root of the project and use command `pip install -r requirements.txt` to download necessary packages. 

### Set up 
&emsp;Before you run the file `main.py`, you should set up these three parameters `pgs`, `mode`, `time` and `rounds`. You can see detailed introduction about them in [main.py](main.py). Additionally, you can also assign parameters in **run**, a function in **main.py**, with new value. But, you should know that the change of value will influence the performance of **DRL_CITO**. 

### Run  
&emsp;Run python file main.py with command `python main.py`.
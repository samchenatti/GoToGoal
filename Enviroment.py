import vrep, time, random
from Pioneer import Pioneer

class TrajectorySampler():
    def __init__(self, policy=None, timestep_lenght=2, epoch_lenght_inmin=30):
        # Politica estocastica. a_t ~ pi(.|x_1, x_2, .... x_13) = pi(.|s_t)
        self.policy      = policy

        # Duracao do episodio em minutos
        self.EPOCH_LENGHT = epoch_lenght_inmin

        # Duracao do timestep em segundos
        self.TIMESTEP_LENGHT = timestep_lenght


    def action_space(self):
        return self.robot.action_space


    # Wrapper para o metodo real. Assim podemos tratar a interrupcao do teclado
    def generate_trajectorys(self, ):
        try:
            return self.__generate_trajectorys()

        except KeyboardInterrupt:
            self.__stop()


    def __generate_trajectorys(self):
        policy = self.policy

        # Fecha conexoes residuais
        vrep.simxFinish(-1)

        # Conecta ao servidor
        self.clientID = clientID = vrep.simxStart('127.0.0.1',19997,True,True,5000,5)

        # Constante: Numero de episodios/trajetorias
        n_epochs = 1

        # Verifica se obtemos uma conexao
        if clientID!=-1:
            print ('Conectado ao servidor')

            # Para a simulacao, caso esteja rodando
            vrep.simxStopSimulation(clientID,  vrep.simx_opmode_blocking)

            # Definimos um controle para a simulacao, assim podemos pausa-la para realizar
            # os calculos e continua-la para executar a cao
            sim_control = SimulationControl(self.clientID, self.TIMESTEP_LENGHT)

            # Objeto do robo (agente)
            robot = self.robot = Pioneer(clientID, sim_control, continuousWalking=False)

            # Garante que todos os motores inicializem em s0
            # self.robot.reset_actor()


            # Armazenamos a trajetoria nesta lista
            trajectory = {"actions":[], "observations":[], "rewards":[]}

            print("Iniciando episódio ")

            # A primeira observacao do episodio vem da posicao neutra do robo
            robot.reset_actor()
            r, o = robot.step(20)

            # Itera os timesteps
            t = 0
            # Cada time step dura, aproximadamente, TIMESTEP_LENGHT dentro da simulacao
            while (t < (self.EPOCH_LENGHT * 60 / self.TIMESTEP_LENGHT)) and not robot.epoch_failed:
                print(" ")
                print("Timestep %d" %t)
                if self.policy:
                    # Lembrando que a politica eh n deterministica
                    a = policy.sample_action(o)

                else:
                    # Se nao tivermos uma politica, bora fazer um random walk :D
                    a = random.randint(0, 7)

                # Obtemos a recompensa e a observacao para a acao a
                r, o = robot.step(a)

                trajectory["actions"].append(a)
                trajectory["rewards"].append(r)
                trajectory["observations"].append(o)

                t += 1


            print("Finalizando episódio")
            # O stopSimulation retorna a simulacao ao estado inicial
            vrep.simxStopSimulation(clientID,  vrep.simx_opmode_blocking)

            self.__stop()
            # Retorna o set de trajetorias
            return trajectory

        else:
            print("Nao foi possivel obter uma conexao com o servidor")

    def __stop(self):
        print("Forcando interrupcao")
        vrep.simxStopSimulation(self.clientID,  vrep.simx_opmode_blocking)
        self.robot.reset_actor()


class SimulationControl:
    def __init__(self, cid, ds, callback=None):
        self.clientID        = cid
        self.clock           = Clock(cid)

        self.TIMESTEP_LENGHT = ds

        self.call_before_pause = callback

    def pass_time(self):
        # Ligamos a simulacao para executarmos uma acao
        self.clock.set_checkpoint()
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)

        # Esperamos ate que se passe no minimo o tempo do ts dentro da
        # simulacao para executar o proximo time step
        self.clock.wait_until(self.TIMESTEP_LENGHT)

        # Interrompemos a simulacao para efetuar os calculos
        vrep.simxPauseSimulation(self.clientID,  vrep.simx_opmode_blocking)


# Clock para contar o tempo passado dentro da simulacao
class Clock():
    def __init__(self, clientID):
        self.last_checkpoint = 0
        self.clientID        = clientID


    # Seta um novo checkpoint
    def set_checkpoint(self):
        vrep.simxGetLastErrors(self.clientID, vrep.simx_opmode_blocking)
        self.last_checkpoint = self.__tick()

    # Espera ate que ds tenha se passado dentro da simulacao
    def wait_until(self, ds):
        while(not self.reached(ds)):
            self.__wait

    # Verifica se o tempo "time" passou dentro da simulacao
    def reached(self, time):
        t = self.__tick()

        ds = (t - self.last_checkpoint)

        if (ds / 1000 >= time):
            print("Time setp interval reached: %f" %(ds / 1000))
            self.set_checkpoint()
            return True

        return False

    # Que feio, Samuel. Busy wait :<
    def __wait(self):
        return

    # Jogamos uma query qualquer para o simulador so para sabermos qual o tempo
    # atual dentro da simulacao
    def __tick(self):
        vrep.simxGetLastErrors(self.clientID, vrep.simx_opmode_blocking)
        return vrep.simxGetLastCmdTime(self.clientID)

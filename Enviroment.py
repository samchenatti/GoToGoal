import vrep, time, random, threading
from Pioneer import Pioneer

class TrajectorySampler():
    def __init__(self, policy=None):
        # Politica estocastica. a_t ~ pi(.|x_1, x_2, .... x_13) = pi(.|s_t)
        self.policy      = policy
        self.trajectorys = []

        # Duracao do timestep em segundos
        self.TIMESTEP_LENGHT = 2

    def generate_trajectorys(self, ):
        try:
            self.__generate_trajectorys()

        except KeyboardInterrupt:
            self.__stop()

    def __generate_trajectorys(self):
        policy = self.policy

        # Fecha conexoes residuais
        vrep.simxFinish(-1)

        # Conecta ao servidor
        self.clientID = clientID = vrep.simxStart('127.0.0.1',19997,True,True,5000,5)

        # Constante: Numero de episodios/trajetorias
        n_epochs = 10

        # Verifica se obtemos uma conexao
        if clientID!=-1:
            print ('Conectado ao servidor')

            # Para a simulacao, caso esteja rodando
            vrep.simxStopSimulation(clientID,  vrep.simx_opmode_blocking)

            # Definimos um relogio para termos acesso ao tempo passado dentro da simulacao
            clock = Clock(self.clientID)

            # Objeto do robo (agente)
            self.robot = robot = Pioneer(clientID, continuousWalking=False)

            # Garante que todos os motores inicializem em s0
            # self.robot.reset_actor()

            # Itera os episodios
            for e in range(0, n_epochs):
                # Armazenamos a trajetoria nesta lista
                trajectory = {"actions":[], "states":[], "rewards":[]}

                print("Iniciando episódio ")

                # A primeira observacao do episodio vem da posicao neutra do robo
                robot.reset_actor()
                r, o = robot.step(7)

                trajectory["states"].append(o)

                # Itera os timesteps
                t = 0
                # Cada time step dura, aproximadamente, 2s dentro da simulacao
                while (t < 3600) and not robot.epoch_failed:
                    if self.policy:
                        # Lembrando que a politica eh n deterministica
                        a = policy.sample_action(o)

                    else:
                        # Se nao tivermos uma politica, bora fazer um random walk :D
                        a = random.randint(0, 7)


                    # Ligamos a simulacao para executarmos uma acao
                    vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)
                    clock.set_checkpoint()

                    # Obtemos a recompensa e a observacao para a acao a
                    r, o = robot.step(a)

                    trajectory["actions"].append(a)
                    trajectory["rewards"].append(r)
                    trajectory["states"].append(o)


                    # Esperamos ate que se passe no minimo o tempo do ts dentro da
                    # simulacao para executar o proximo time step
                    clock.wait_until(self.TIMESTEP_LENGHT)

                    # Interrompemos a simulacao para efetuar os calculos
                    vrep.simxPauseSimulation(clientID,  vrep.simx_opmode_blocking)


                print("Finalizando episódio")
                # O stopSimulation retorna a simulacao ao estado inicial
                vrep.simxStopSimulation(clientID,  vrep.simx_opmode_blocking)

                self.trajectorys.append(trajectory)


        else:
            print("Nao foi possivel obter uma conexao com o servidor")

    def __stop(self):
        print("Forcando interrupcao")
        vrep.simxStopSimulation(self.clientID,  vrep.simx_opmode_blocking)
        self.robot.reset_actor()
        self.policy.stop()



# Clock para contar o tempo passado dentro da simulacao
class Clock():
    def __init__(self, clientID):
        self.last_checkpoint = 0
        self.clientID        = clientID


    def set_checkpoint(self):
        vrep.simxGetLastErrors(self.clientID, vrep.simx_opmode_blocking)
        self.last_checkpoint = self.__tick()

    def wait_until(self, ds):
        while(not self.reached(ds)):
            self.__wait

    def reached(self, time):
        t = self.__tick()

        ds = (t - self.last_checkpoint)

        if (ds / 1000 >= time):
            print("Time setp interval reached: %f" %(ds / 1000))
            self.set_checkpoint()
            return True

        return False

    def __wait(self):
        return

    def __tick(self):
        vrep.simxGetLastErrors(self.clientID, vrep.simx_opmode_blocking)
        return vrep.simxGetLastCmdTime(self.clientID)

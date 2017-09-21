import vrep, time, random
from Pioneer import Pioneer

class TrajectorySampler():
    def __init__(self, policy=None):
        # Politica estocastica. a_t ~ pi(.|x_1, x_2, .... x_13) = pi(.|s_t)
        self.policy      = policy
        self.trajectorys = []

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

            # Objeto do robo (agente)
            self.robot = robot = Pioneer(clientID, continuousWalking=False)

            # Garante que todos os motores inicializem em s0
            # self.robot.reset_actor()

            # Itera os episodios
            for e in range(0, n_epochs):
                # Armazenamos a trajetoria nesta lista
                trajectory = {"actions":[], "states":[], "rewards":[]}

                print("Iniciando episódio ")
                vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)

                # A primeira observacao do episodio vem da posicao neutra do robo
                r, o = robot.step(7)

                trajectory["states"].append(o)

                # Itera os timesteps
                for t in range(0, 30000):

                    if self.policy:
                        # Lembrando que a politica eh n deterministica
                        a = policy.sample_action(o)

                        # Obtemos a recompensa e a observacao para a acao a
                        r, o = robot.step(a)

                        trajectory["actions"].append(a)
                        trajectory["rewards"].append(r)
                        trajectory["states"].append(o)


                    else:
                        # Se nao tivermos uma politica, bora fazer um random walk :D
                        r, o = robot.step(random.randint(0, 7))

                print("Finalizando episódio")
                vrep.simxStopSimulation(clientID,  vrep.simx_opmode_blocking)

                self.trajectorys.append(trajectory)


        else:
            print("Nao foi possivel obter uma conexao com o servidor")

    def __stop(self):
        print("Forcando interrupcao")
        vrep.simxStopSimulation(self.clientID,  vrep.simx_opmode_blocking)
        self.robot.reset_actor()
        self.policy.stop()

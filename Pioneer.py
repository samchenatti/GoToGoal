import vrep
import numpy as np

# Classe para controlar o robo Pioneer (Carinhosamente apelidado como Robertinho)
class Pioneer:
    def __init__(self, cid, sim_control, continuousWalking=None):
        self.clientID          = cid
        self.motor_handler     = {"right": None, "left": None}
        self.sensor_handler    = []

        self.sim_control       = sim_control

        self.action_space      = [0, 1, 2, 3, 4, 5, 6]
        self.last_pos_since_d  = None
        self.last_position     = 0

        # Constantes de velocidade
        self.DEFAULT_VELOCITY  = 1
        self.SPEED_UP_FACTOR   = 1.5

        # Constantes relacionadas a recompensa
        self.REWARD_BY_DESLOC  = 0.15  # O robo recebe uma recompensa por se deslocar 15cm
        self.PENALTY_PROXIMITY = 0.15 # Penaliza o robo no caso de 0.4 dos sensores indicarem distancia de um obstaculo

        # Usamos estas variaveis para dizer se o robo esta bloqueado, e por quantos
        # timesteps
        self.blocked           = False
        self.steps_blocked     = 0

        # Dizemos que um episodio falhou e o encerramos prematuramente
        self.epoch_failed      = False


        self.__get_handlers()

        if continuousWalking:
            self.__continous_walking(True, 1)


    def get_action_space(self):
        return self.action_space

    # Executa uma acao a_t, retorna a observacao o_t+1 e r_t
    def step(self, action):
        # A primeira modelagem do espaco de acoes era muito complexo
        # if action == 0:
        #     self.__set_motor_velocity("left", self.DEFAULT_VELOCITY)
        #
        # if action == 1:
        #     self.__set_motor_velocity("left", self.DEFAULT_VELOCITY * self.SPEED_UP_FACTOR)
        #
        # if action == 2:
        #     self.__set_motor_velocity("right", self.DEFAULT_VELOCITY)
        #
        # if action == 3:
        #     self.__set_motor_velocity("right", self.DEFAULT_VELOCITY * self.SPEED_UP_FACTOR)
        #
        # if action == 4:
        #     self.__set_motor_velocity("left", 0)
        #
        # if action == 5:
        #     self.__set_motor_velocity("right", 0)
        # # action == 6: do nothing

        # Siga em frente
        if   action == 0 or action == 5:
            self.__set_motor_velocity("left",  self.DEFAULT_VELOCITY)
            self.__set_motor_velocity("right", self.DEFAULT_VELOCITY)

        # Olhe para o lado
        elif action == 1 or action == 6:
            self.__set_motor_velocity("left",  0)
            self.__set_motor_velocity("right", self.DEFAULT_VELOCITY)

        elif action == 2 or action == 7:
            self.__set_motor_velocity("right", 0)
            self.__set_motor_velocity("left",  self.DEFAULT_VELOCITY)

        elif action == 3 or action == 8:
            self.__set_motor_velocity("right", 0)
            self.__set_motor_velocity("left",  0)

        #    action == 4 or action == 9: do nothing


        # Executa a acao e deixa o timestep ocorrer
        self.sim_control.pass_time()
        # Interrompe a simulacao (fim do timestep)

        # Calcula a observacao e a recompensa
        observation = self.__get_sensors_info()
        reward      = self.__get_reward(observation)

        # Se o robo ficar parado por 5 minutos o episodio falhou
        if self.steps_blocked >= 600:
            self.epoch_failed = True

        # Nao lembro pq coloquei isso aqui, mas eh melhor deixar
        self.__desloc()
        self.last_position = self.__get_last_coord()

        return (reward, observation)


    # "Private" methods
    # Reward method
    def __get_reward(self, o):
        # A recompensa padrao eh zero
        reward = 0

        # # Armazena em blocked os sensores a menos de 15cm de um obstaculo
        # blocked = o[np.where(o < self.PENALTY_PROXIMITY)]
        #
        # # Ao encostar na parede 35/100 dos sensores reportam uma distancia < 15cm
        # # Logo, assumindo uma distribuição quase uniforme dos sensores, podemos inferir que esta relacao implica uma colisao e o penalizamos
        # if len(blocked) / len(o) > 0.35:
        #     reward += -2
        #
        #     # Dizemos que o robo esta bloqueado
        #     self.blocked = True
        #
        #     # Contabilizamos por quantos timesteps o robo ficou preso
        #     self.steps_blocked += 1
        #
        #     if self.steps_blocked == 120:
        #         self.epoch_failed = True
        #         print("Episode fail :(")
        #
        # # No caso do robo ter conseguido se desbloquear, damos a ele uma recompensa (invesamente proporcional ao tempo que passou bloqueado)
        # elif self.blocked:
        #     reward += 1
        #     self.blocked = False

        # Verificamos se o robo esta bloqueado, ie, seu deslocamento desde o ultimo ts foi < 1cm
        if self.__is_blocked():
            # Dizemos que o robo esta bloqueado
            self.blocked = True

            print("Pioneer esta bloquedo :( ")
            # Contabilizamos por quantos timesteps o robo ficou preso
            self.steps_blocked += 1

            reward -= 1

        # Se o robo consegue se destravar damos a ele uma recompensa
        elif self.blocked:
            reward += 2

            self.blocked       = False
            self.steps_blocked = 0

        # Damos uma recompensa para o robo por ter andando por 30cm
        # Note que apesar do deslocamento ser calculado atraves da posicao absoluta, poderiamos utilizaar
        # a velocidade angular pra calcula-lo
        if self.__desloc() >= self.REWARD_BY_DESLOC and not self.blocked:
            reward += 1

        return reward


    def __get_last_coord(self):
        return np.array(vrep.simxGetObjectPosition(self.clientID, self.motor_handler["right"], -1, vrep.simx_opmode_buffer)[1])

    # Verifica se o robo esta travado
    def __is_blocked(self):
        p = np.array(vrep.simxGetObjectPosition(self.clientID, self.motor_handler["right"], -1, vrep.simx_opmode_buffer)[1])

        desloc = np.linalg.norm(p - self.last_position)

        self.last_position = p

        # Se o deslocamento eh infinmo, o robo esta travado
        if desloc <= 0.0009:
            return True

        return False

    def __desloc(self):
        p = np.array(vrep.simxGetObjectPosition(self.clientID, self.motor_handler["right"], -1, vrep.simx_opmode_buffer)[1])

        desloc = np.linalg.norm(p - self.last_pos_since_d)

        if desloc >= 0.3:
            self.last_pos_since_d = self.__get_last_coord()

        return desloc


    # "Seta" os handlers dos motores e dos sensores
    def __get_handlers(self):
        # Motor handlers
        left, self.motor_handler["right"] = vrep.simxGetObjectHandle(self.clientID, "Pioneer_p3dx_rightMotor", vrep.simx_opmode_blocking)
        left, self.motor_handler["left"]  = vrep.simxGetObjectHandle(self.clientID, "Pioneer_p3dx_leftMotor",  vrep.simx_opmode_blocking)

        # Sensor handlers
        for i in range(1, 16):
            s = "Pioneer_p3dx_ultrasonicSensor%d" %i
            e, sh = vrep.simxGetObjectHandle(self.clientID, s, vrep.simx_opmode_blocking)

            if e!=0:
                print("Erro: " + s + " :(")
                print(sh)

            self.sensor_handler.append(sh)

        # Coloca os sensores em streaming
        for sensor_handler in self.sensor_handler:
            vrep.simxReadProximitySensor(self.clientID, sensor_handler, vrep.simx_opmode_streaming)

        vrep.simxGetObjectPosition(self.clientID, self.motor_handler["right"], -1, vrep.simx_opmode_streaming)[1]

        self.last_pos_since_d = np.array(vrep.simxGetObjectPosition(self.clientID, self.motor_handler["right"], -1, vrep.simx_opmode_buffer)[1])


    # Retorna a distancia ate o ponto identificado pelo sensor
    def __get_sensors_info(self):
        l = []
        for sensor_handler in self.sensor_handler:
            r = vrep.simxReadProximitySensor(self.clientID, sensor_handler, vrep.simx_opmode_streaming)
            detect   = r[1]

            if detect:
                point    = np.array(r[2])
                distance = np.linalg.norm(point)
                l.append(distance)
            else:
                l.append(1)

        return np.array(l)


    # Seta o robo para andar continuamente
    def __continous_walking(self, flag, v):
        if v:
            vrep.simxSetJointTargetVelocity(self.clientID, self.motor_handler["right"], self.DEFAULT_VELOCITY, vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetVelocity(self.clientID, self.motor_handler["left"], self.DEFAULT_VELOCITY, vrep.simx_opmode_blocking)
        else:
            vrep.simxSetJointTargetVelocity(self.clientID, self.motor_handler["right"], 0, vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetVelocity(self.clientID, self.motor_handler["left"], 0, vrep.simx_opmode_blocking)

    # Define a velocidade de um dos motores
    def __set_motor_velocity(self, motor, velocity):
        vrep.simxSetJointTargetVelocity(self.clientID, self.motor_handler[motor], velocity, vrep.simx_opmode_blocking)

    def reset_actor(self):
        self.__set_motor_velocity("right", 0)
        self.__set_motor_velocity( "left", 0)

        self.steps_blocked = 0
        self.blocked       = False

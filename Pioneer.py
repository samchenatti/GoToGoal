import vrep
import numpy as np
import cv2
from PIL import Image

# Classe para controlar o robo Pioneer (Carinhosamente apelidado como Robertinho)
class Pioneer:
    def __init__(self, cid, sim_control, continuousWalking=None):
        self.clientID          = cid
        self.motor_handler     = {"right": None, "left": None}
        self.sensor_handler    = []
        self.kinect_handler    = []

        self.sim_control       = sim_control

        self.action_space      = [0, 1, 2, 3, 4]
        self.last_pos_since_d  = None
        self.last_position     = 0
        self.last_image        = None

        # Constantes de velocidade
        self.DEFAULT_VELOCITY  = 1
        self.SPEED_UP_FACTOR   = 1.5

        # Constantes relacionadas a recompensa
        self.REWARD_BY_DESLOC  = 1  # O robo recebe uma recompensa por se deslocar 15cm
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

    def set_clientid(self, cid):
        self.clientID = cid

    def get_action_space(self):
        return self.action_space

    # Executa uma acao a_t, retorna a observacao o_t+1 e r_t
    def step(self, action):
        # Siga em frente
        if   action == 0:
            self.__set_motor_velocity("left",  self.DEFAULT_VELOCITY)
            self.__set_motor_velocity("right", self.DEFAULT_VELOCITY)

        # Olhe para o lado
        elif action == 1:
            self.__set_motor_velocity("left",  0)
            self.__set_motor_velocity("right", self.DEFAULT_VELOCITY)

        elif action == 2:
            self.__set_motor_velocity("right", 0)
            self.__set_motor_velocity("left",  self.DEFAULT_VELOCITY)

        # Se liga no...
        elif action == 3:
            self.__set_motor_velocity("right", 0)
            self.__set_motor_velocity("left",  0)

        # action == 4: do nothing


        # Executa a acao e deixa o timestep ocorrer
        self.sim_control.pass_time()
        # Interrompe a simulacao (fim do timestep)

        # Calcula a observacao e a recompensa
        observation = self.__get_sensors_info()
        reward      = self.__get_reward(observation)

        # Se o robo ficar parado por 2.5 minutos o episodio falhou (para ds = 2s)
        if self.steps_blocked >= 15:
            self.epoch_failed = True
            print("Pioneer ficou preso por tempo demais e o episodio falhou")

        # Nao lembro pq coloquei isso aqui, mas eh melhor deixar
        self.__desloc()
        self.last_position = self.__get_last_coord()

        print("Recompensa nesse ts: %d" %reward)
        return (reward, observation)


    # "Private" methods
    # Reward method
    def __get_reward(self, o):
        # A recompensa padrao eh zero
        reward = 0

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

        # Verifica se o robo chegou ao objetivo
        p = self.last_position
        if (p[0] >= 1 and p[0] <= 2) and (p[1] >= -2 and p[1] <= -1):
            print("Pioneer chegou ao destino e o episodio foi um sucesso :D ")
            self.epoch_failed = True #TODO mudar o nome dessa flag
            reward += 20

        return reward


    def __get_last_coord(self):
        return np.array(vrep.simxGetObjectPosition(self.clientID, self.motor_handler["right"], -1, vrep.simx_opmode_buffer)[1])

    # Verifica se o robo esta travado
    def __is_blocked(self):
        p = np.array(vrep.simxGetObjectPosition(self.clientID, self.motor_handler["right"], -1, vrep.simx_opmode_buffer)[1])

        desloc = np.linalg.norm(p - self.last_position)
        print("Isblocked time: %f" %self.steps_blocked)
        self.last_position = p

        # Se o deslocamento eh infinmo, o robo esta travado
        if desloc <= 0.05:
            return True

        return False

    def __desloc(self):
        p = np.array(vrep.simxGetObjectPosition(self.clientID, self.motor_handler["right"], -1, vrep.simx_opmode_buffer)[1])

        desloc = np.linalg.norm(p - self.last_pos_since_d)

        if desloc >= self.REWARD_BY_DESLOC:
            self.last_pos_since_d = self.__get_last_coord()

        return desloc


    # "Seta" os handlers dos motores e dos sensores
    def __get_handlers(self):
        # Motor handlers
        left, self.motor_handler["right"] = vrep.simxGetObjectHandle(self.clientID, "Pioneer_p3dx_rightMotor", vrep.simx_opmode_blocking)
        left, self.motor_handler["left"]  = vrep.simxGetObjectHandle(self.clientID, "Pioneer_p3dx_leftMotor",  vrep.simx_opmode_blocking)
        left, self.kinect_handler         = vrep.simxGetObjectHandle(self.clientID, "kinect_rgb",               vrep.simx_opmode_blocking)

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
        vrep.simxGetVisionSensorImage(self.clientID, self.kinect_handler, 10, vrep.simx_opmode_streaming)

        # Nao lembro o pq disso estar ai, mas nao mexe
        self.last_pos_since_d = np.array(vrep.simxGetObjectPosition(self.clientID, self.motor_handler["right"], -1, vrep.simx_opmode_buffer)[1])


    # Retorna a distancia ate o ponto identificado pelo sensor
    def __get_sensors_info(self):
        l = []
        for sensor_handler in self.sensor_handler:
            r = vrep.simxReadProximitySensor(self.clientID, sensor_handler, vrep.simx_opmode_buffer)
            detect   = r[1]

            if detect:
                point    = np.array(r[2])
                distance = np.linalg.norm(point)
                l.append(distance)
            else:
                l.append(1)


        # Captura a imagem do kinect
        # r, image_res, image = vrep.simxGetVisionSensorImage(self.clientID, self.kinect_handler, 0, vrep.simx_opmode_buffer)

        # self.__process_image(image)

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


    # Varias loucuras de processamento de imagem
    def __process_image(self, image):
        self.__make_rgb_map(image)

    def __make_rgb_map(self, image):
        rgb_image = []

        i = 0
        pixel = []
        for value in image:
            pixel.append(np.absolute(value))
            i += 1

            if i == 3:
                i = 0
                rgb_image.append(pixel)
                pixel = []

        i = 0
        final_image = []
        line        = []
        for pixel in rgb_image:
            line.append(pixel)
            i += 1

            if i == 120:
                i = 0
                final_image.append(line)
                line = []

        rgb_image = np.array(final_image)
        print(rgb_image)
        img = Image.fromarray(np.array(rgb_image), 'RGB')
        img.save('out.png')

        return rgb_image

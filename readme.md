# GoToGoal (wip)

## Portuguese

No GoToGoal, um robo deve percorrer um labirinto e chegar ao objetivo final no menor tempo possível.
A unica referência do ambiente à qual tem acesso são os sensores conicos de proximidade e uma camera de baixa resolução.

Minha solução para o desafio se baseia no algoritmo de Politica do Gradiente com Estimador de Vantagem (ActorCritic)

### Instruções

 - Certifique-se de possuir os pacotes *tensorflow*, *numpy* e *vrep* para o Python 3.
 - De dentro da pasta raiz do projeto, rode os seguintes comandos:

```
# vrep ClientVrep/scenes/goToGoalSolidWall.ttt (use -h para rodar sem gui)
# python train.py
```

 - O script pode ser interrompido a qualquer momento sem risco de perder ou danificar os parametros das redes neurais


 ### Estrutura do projeto

**ModularNN:** Implementação modular de uma MLP. Por fins de estudo, optei por construir o fowardpass manualmente.

**Policy:** Gerencia o estimador de vantagem e a distribuição de probabilidades das ações.

**Enviroment:** Implementa uma interface gym-like para o Vrep, gerando as trajetórias a partir dos resultados da política e as observações do Pioneer

**Pioneer:** Controla o robô dentro da simulação e encapsula o processo de decisão de markov.

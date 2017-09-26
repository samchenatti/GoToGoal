# GoToGoal (wip)

## Portuguese

No GoToGoal, um robo deve percorrer um labirinto e chegar ao objetivo final no menor tempo possível.
A unica referência do ambiente à qual tem acesso são os sensores conicos de proximidade e uma camera de baixa resolução.

Minha solução para o desafio se baseia no algoritmo de Politica do Gradiente com Estimador de Vantagem (ActorCritic)

### Instruções

 - Certifique-se de possuir os pacotes *tensorflow*, *numpy* e *vrep*.
 - De dentro da pasta raiz do projeto, rode os seguintes comandos

```
# vrep ClientVrep/scenes/goToGoalSolidWall.ttt (use -h para rodar sem gui)
# python train.py
```

 - O script pode ser interrompido a qualquer momento sem risco de danos aos parametros da rede

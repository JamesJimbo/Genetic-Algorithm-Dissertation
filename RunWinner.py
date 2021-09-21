import retro
import numpy as np
import cv2
import neat
import pickle

env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')

imgArray = []

xposEnd = 0

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-feedforward')

p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

with open('winner.pkl', 'rb') as input_file:
    genome = pickle.load(input_file)

print(genome)
    
ob = env.reset()
ac = env.action_space.sample()

inx, iny, inc = env.observation_space.shape

inx = int(inx/8)
iny = int(iny/8)

net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

currentMaxFitness = 0
fitnessCurrent = 0
frame = 0
counter = 0
xpos = 0
xposMax = 0

done = False

cv2.namedWindow("main", cv2.WINDOW_NORMAL)

while not done:
    
    env.render()
    frame += 1

    scaledImg = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
    scaledImg = cv2.resize(scaledImg, (iny, inx))

    ob = cv2.resize(ob, (inx, iny))
    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
    ob = np.reshape(ob, (inx, iny))

    cv2.imshow('main', scaledImg)
    cv2.waitKey(1)

    for x in ob:
        for y in x:
            imgArray.append(y)

    nnOutput = net.activate(imgArray)

    ob, rew, done, info = env.step(nnOutput)
    imgArray.clear()

    xpos = info['x']
    xposEnd = info['screen_x_end']

    if xpos > xposMax:
        fitnessCurrent += 1
        xposMax = xpos

    if fitnessCurrent > currentMaxFitness:
        currentMaxFitness = fitnessCurrent
        counter = 0
    else:
        counter += 1

    if done or counter == 250:
        done = True
        print(fitnessCurrent)

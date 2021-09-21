import retro
#Allows classic video games to be supported in Gym environments for AI.
import numpy as np
#Works with arrays and matrices.
import cv2
#Displays the game onto the screen.
import neat
#Neuro-Evolution of Augmented Topologies; used to create a neural network.
import pickle
#Used to serialise and deserialise an object.

zoneNames = ['GreenHillZone','MarbleZone','SpringYardZone','LabyrinthZone','StarLightZone','ScrapBrainZone']
zoneShow = ['Green Hill Zone','Marble Zone','Spring Yard Zone','Labyrinth Zone','Star Light Zone','Scrap Brain Zone']
actNo = ['1','2','3']
scrapBrainNo = ['1', '2']
#The lists needed to allow the user to input a valid zone name and act number.

choice = input("Would you like to see the available zones? (Y/N): ")
print()
if choice.upper() == "Y":
    for x in range(len(zoneNames)):
        print(zoneShow[x])
        print()
    print("Enter in camelcase, such as GreenHillZone \n")
#Allows the user to see what zones are available.

def checkScrapBrain():
    global act
    scrapBrainDone = False
    while scrapBrainDone == False:
        act = input("Enter act: ")
        if act in scrapBrainNo:
            scrapBrainDone = True
        elif act not in scrapBrainNo:
            print("You must input either 1 or 2")
#Scrap Brain Zone only has two acts so a seperate function is needed to allow inputs 1 or 2.

def checkAct():
    global act
    actDone = False
    while actDone == False:
        act = input("Enter act: ")
        if act in actNo:
            actDone = True
        elif act not in actNo:
            print("You must input either 1, 2 or 3")
#If the user inputs an invalid act number, the program repeats until a valid input is inputted.

def checkZone():
    global zone
    zoneDone = False
    while zoneDone == False:
        zone = input("Enter zone: ")
        if zone in zoneNames and zone != "ScrapBrainZone":
            zoneDone = True
            checkAct()
        elif zone == "ScrapBrainZone":
            checkScrapBrain()
            zoneDone = True
        elif zone not in zoneNames:
            print("That is an invalid zone name")
#If the user inputs an invalid zone name, the program repeats until a valid input is inputted.

checkZone()

environment = retro.make(game = 'SonicTheHedgehog-Genesis', state = (zone + '.Act' + act))
#Creates the environment. The game is specified and the state is a concatenation of the users zone and act input.

imgArray = []
xposEnd = 0

resume = True
#Keeps the program running.
#restoreFile = "neat-checkpoint-46"
#This is the most recent checkpoint in the neural network.
#If the program is being ran without any checkpoints then
#restoreFile must be commented out of the program.

def evalGenomes(genomes, configure):
#A genome is an individual in a population.

    for genomeID, genome in genomes:
        ob = environment.reset()
        ac = environment.action_space.sample()
        #If each genome does not make progress then the environment is reset.

        inx, iny, inc = environment.observation_space.shape

        inx = int(inx/8)
        iny = int(iny/8)
        #Feeds the X and Y coordinates to the agent (genome).

        network = neat.nn.recurrent.RecurrentNetwork.create(genome, configure)
        #Creates the neural network.
        
        currentMaxFitness = 0
        fitnessCurrent = 0
        frame = 0
        counter = 0
        xpos = 0
        xposMax = 0
        #Keeps count of the genomes fitness, X position, frame and the
        #current maximum fitness of each generation.
        
        done = False

        while not done:
     
            environment.render()
            frame += 1
            #Renders the game, each frame is incremented.
            
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx,iny))
            #Shapes the size of the array, allowing the game to play.

            imgArray = np.ndarray.flatten(ob)
            #Expects the inputs of the neural network.

            nnOutput = network.activate(imgArray)
            #Activates the neural network.
            
            ob, rew, done, info = environment.step(nnOutput)
            
            xpos = info['x']
            #Sonic's X position.
            xposEnd = info['screen_x_end']
            #The end of the level (before the goal).
            
            if xpos > xposMax:
                fitnessCurrent += 1
                xposMax = xpos
                counter = 0
            else:
                counter += 1
            #Calculates Sonic's exact X position once he is eliminated.
            
            if xpos >= (xposEnd + 360):
            #The end of the screen is right before the goal, an extra
            #360 piexels have been added to this value so Sonic passes it.
                fitnessCurrent = 100000
                print("Winner!")
                done = True
                #If an agent reaches the end of the stage,
                #their current fitness is set to 10,000.
                #This means that they have beaten the level.
            
            fitnessCurrent += rew
            #Adds the value of rew to current fitness.
            
            if fitnessCurrent > currentMaxFitness:
                currentMaxFitness = fitnessCurrent
                counter = 0
            else:
                counter += 1
            #If the current fitness is greater than the highest fitness
            #then that fitness becomes the highest fitness and the counter
            #is set to 0. Otherwsise, the counter increments.
                
            if done or counter == 250:
                done = True
                print(genomeID, fitnessCurrent)
            #If the genome makes no progress after 250 frames, it is reset
            #and the program prints its ID and its fitness.
                
            genome.fitness = fitnessCurrent
                
configure = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

#This configures the neural network depending on the content of the
#'config-feedforward' file, which is a txt file that specifies the fitness
#threshold, maximum population, maximum stagnation, inputs, outputs, etc.

if resume == True:
    #p = neat.Checkpointer.restore_checkpoint(restoreFile)
#else:  
    p = neat.Population(configure)
#Restores the content of the restoreFile; essentially continutes where the
#neural network left off. Otherwise, the population continues for that generation.

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))
#Adds the inputs and statistics of the most fit genome into the file.


winner = p.run(evalGenomes)
#If the genome has a fitness of 100,000 then the program stops and displays its ID.

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
#Creates a pickle file to store its connections and nodes.

showWinner = pickle.load(open('winner.pkl','rb'))
print(showWinner)
#Displays the winning genome's connections and nodes.

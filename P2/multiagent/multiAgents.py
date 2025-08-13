# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
from game import Actions
import math
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        print(" ")
        print("action: " , action)
        succGameState = currentGameState.generatePacmanSuccessor(action)
        print("successorGameState: " , succGameState)
        newPos = succGameState.getPacmanPosition()
        #print("newPos: " , newPos)
        newFood = succGameState.getFood()
        #print("newFood: " , newFood)
        newGhostStates = succGameState.getGhostStates()
        #print("newGhostStates: " , newGhostStates)
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
       #print("newScaredTimes: " , newScaredTimes)
        Score = succGameState.getScore()
        "*** YOUR CODE HERE ***"
       # print("nextPos: " ,nextPos)
        ghostDist = []
        manDist = 0
        correction = 0
        for git in newGhostStates:
            gPos = git.configuration.pos
            #print(gPos)
            gx, gy = gPos
            manDist = manhattanDistance(newPos,(gx,gy))
            ghostDist.append(manDist)            

        cGhost = min(ghostDist)
        
        foodDistNext =[]
        foodPos = newFood.asList()
        cFood = 999
        ScoreFromAction = 0
        for x in foodPos:
            foodDistNext.append(manhattanDistance(newPos,x))
        

        if newPos in foodPos:
            ScoreFromAction = 10
        else:
            ScoreFromAction = -10

        if len (foodDistNext) == 0:
            return 0
            
        if(action == Directions.STOP):
            correction = -100000

        cFood = min(foodDistNext)
        print("cFood : ", (1/.1+cFood))
        print("cGhost: ", math.log(cGhost+1))
        print("ScoreFromAction : ", ScoreFromAction)
        EVOUT =15*(1/(.1+cFood)) + 10*math.log(cGhost+1.1) + Score + 100*ScoreFromAction + correction
        print("EVOUT: " , EVOUT)
        return EVOUT

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        val, action = self.maxAgent(gameState,0)
        return action

    def maxAgent(self,currState,depth):
        #print("maxAgent")
        legalActions = currState.getLegalActions(0)
        MaxValue = -9999999
        CurrValue = -9999999
        MaxAction = Directions.STOP
        targetDepth = self.depth
        if currState.isLose() or currState.isWin() or depth == targetDepth:
            return self.evaluationFunction(currState),MaxAction
        #print(legalActions)
        for action in legalActions:
            nextState = currState.generateSuccessor(0,action)
            CurrValue = self.minAgent(nextState,depth, 1)
            #print( "Choise = ", action, Value, " >= " ,actionOut, lastValue)
            if CurrValue >= MaxValue:
                MaxValue = CurrValue
                MaxAction = action
               # print("Current Chosen Value: ", actionOut ,lastValue)     
        return MaxValue,MaxAction

    def minAgent(self,currState,depth,agent):
        targetDepth = self.depth
        numAgents = currState.getNumAgents()
        #print(numAgents)
        finalState = currState
        MinValue = 9999999
        Values = []
        if currState.isLose() or depth == targetDepth or currState.isWin():
            return self.evaluationFunction(currState)
        if agent == numAgents:
            MinValue, tossAction = self.maxAgent(currState,depth+1)
            return MinValue

        else:
            legalActions = currState.getLegalActions(agent)
            for action in legalActions:
                checkState = currState.generateSuccessor(agent, action)
                MinValue = self.minAgent(checkState,depth,agent+1)
                #print(Value)
                Values.append(MinValue)
            #print(depth,agent)
            #print(Values)
            MinValue = min(Values)
            return MinValue

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = -9999999
        beta = 9999999
        #print("mainLoop Call")
        val, action = self.maxAgent(gameState,0,alpha,beta)
        return action

    def maxAgent(self,currState,depth,alpha,beta):
        #print("maxAgent")
        targetDepth = self.depth
        legalActions = currState.getLegalActions(0)
        MaxValue = -9999999
        CurrValue = -9999999
        MaxAction = Directions.STOP

        if currState.isLose() or currState.isWin() or depth == targetDepth:
            #print("max Terminal")
            return self.evaluationFunction(currState),Directions.STOP

        #print(legalActions)
        for action in legalActions:
            nextState = currState.generateSuccessor(0,action) 
            CurrValue = self.minAgent(nextState,depth,alpha,beta, 1)
            #print( "Choise = ", action, Value, " >= " ,actionOut, lastValue)
            if CurrValue >= MaxValue:
                MaxValue = CurrValue
                MaxAction = action
               # print("Current Chosen Value: ", actionOut ,lastValue) 
            if MaxValue > beta:
                break
            alpha = max(alpha,MaxValue)
        #print("beta: " , beta, "alpha: ", alpha)
        return MaxValue,MaxAction

    def minAgent(self,currState,depth,alpha,beta,agent):
        targetDepth = self.depth
        numAgents = currState.getNumAgents()
        finalState = currState
        MinValue = 9999999
        Values = []
        if currState.isLose() or depth == targetDepth or currState.isWin():
            #print("min Terminal")
            return self.evaluationFunction(currState)
        
        if agent == numAgents:
            MinValue, tossAction = self.maxAgent(currState,depth+1,alpha,beta)
            return MinValue

        else:
            legalActions = currState.getLegalActions(agent)
            for action in legalActions:
                checkState = currState.generateSuccessor(agent, action)
                MinValue = self.minAgent(checkState,depth,alpha,beta,agent+1)
                #print(Value)
                Values.append(MinValue)
            #print(depth,agent)
            #print(Values)
                MinValue = min(Values)
                if MinValue < alpha:
                    break
                beta = min(MinValue,beta)

            return MinValue

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        val, action = self.maxAgent(gameState,0)
        return action

    def maxAgent(self,currState,depth):
        #print("maxAgent")
        legalActions = currState.getLegalActions(0)
        MaxValue = -9999999
        CurrValue = -9999999
        MaxAction = Directions.STOP
        targetDepth = self.depth
        if currState.isLose() or currState.isWin() or depth == targetDepth:
            return self.evaluationFunction(currState),MaxAction
        #print(legalActions)
        for action in legalActions:
            nextState = currState.generateSuccessor(0,action)
            CurrValue = self.exspectAgent(nextState,depth, 1)
            #print( "Choise = ", action, Value, " >= " ,actionOut, lastValue)
            if CurrValue >= MaxValue:
                MaxValue = CurrValue
                MaxAction = action
               # print("Current Chosen Value: ", actionOut ,lastValue)     
        return MaxValue,MaxAction

    def exspectAgent(self,currState,depth,agent):
        targetDepth = self.depth
        numAgents = currState.getNumAgents()
        #print(numAgents)
        finalState = currState
        ExspectedValue = 0
        ExspectOut = 0
        Values = []
        if currState.isLose() or depth == targetDepth or currState.isWin():
            return self.evaluationFunction(currState)
        if agent == numAgents:
            MinValue, tossAction = self.maxAgent(currState,depth+1)
            return MinValue

        else:
            legalActions = currState.getLegalActions(agent)
            succProb = 1.0 / len(legalActions)
            for action in legalActions:
                checkState = currState.generateSuccessor(agent, action)
                ExspectOut = self.exspectAgent(checkState,depth,agent+1)
                ExspectedValue += succProb * ExspectOut
            return ExspectedValue
def betterEvaluationFunction(currState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    playerPos = currState.getPacmanPosition()
    
    ghostPos = currState.getGhostPositions()
    ghostDist = []
    minGhostDist = 9999999

    foodPos = currState.getFood().asList()
    foodCount = len(foodPos)
    foodDist = []
    minFoodDist = 9999999

    capPos = currState.getCapsules()
    capCount = len(capPos)
    capDist = []
    minCapDist = 9999999


    gStates = currState.getGhostStates()
    scaredTimers = [ghostState.scaredTimer for ghostState in gStates]

    

    for ghost in ghostPos:
        ghostDist.append(manhattanDistance(playerPos,ghost))

    if len(ghostDist) > 0:
        minGhostDist = min(ghostDist)
    else: 
        minGhostDist = 0

    for food in foodPos:
        foodDist.append(manhattanDistance(playerPos, food))

    if len(foodDist) > 0: 
        minFoodDist = min(foodDist)
    else:
        minFoodDist = 99999999

    for cap in capPos:
        capDist.append(manhattanDistance(playerPos, cap))
    if len(capDist) > 0:
        minCapDist = min(capDist)
    else:
        minCapDist = 99999999

    if min(scaredTimers) != 0:#minCapDist Has no weight if the scared timer is active
        minCapDist = 99999999

    #start of feture functions
    
    ###GHOST FEATURE###
    #log function from earlier wasn't effective enough it was kind of like a light slap on the wrist
    gFeature = 0
    if minGhostDist < 2:
        gFeature = 99999

    foodFeature = 1/(.01+ minFoodDist)#should probably use 1 but its not as effective

    capFeature = 1/(.01 + minCapDist)

    scoreFeature = currState.getScore()

    evaluation = gFeature + 500*(foodFeature) + 250*(capFeature) + 600*(scoreFeature)
    return evaluation
# Abbreviation
better = betterEvaluationFunction


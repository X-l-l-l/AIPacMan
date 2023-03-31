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

from multiagent.pacman import GameState
from util import manhattanDistance
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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = 0

        closestGhost = min([manhattanDistance(newPos, a.configuration.pos) for a in newGhostStates])

        newFoodPositions = newFood.asList()
        foodDistances = [manhattanDistance(newPos, foodPosition) for foodPosition in newFoodPositions]

        if successorGameState.isWin():
            return float("+inf")

        if len(foodDistances) == 0:
            return 0

        closestFood = min(foodDistances)

        if action == 'Stop':
            score -= 50

        return successorGameState.getScore() + closestGhost / (closestFood * 10) + score


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.constant_depth = int(depth)
        self.level = 0


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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def max_value(gameState, depth):
            Actions = gameState.getLegalActions(0)
            if len(Actions) == 0 or gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState), None)
            w = -(float("inf"))
            Act = None
            for action in Actions:
                sucsValue = min_value(gameState.generateSuccessor(0, action), 1, depth)
                sucsValue = sucsValue[0]
                if (sucsValue > w):
                    w, Act = sucsValue, action
            return (w, Act)

        def min_value(gameState, agentID, depth):
            Actions = gameState.getLegalActions(agentID)
            if len(Actions) == 0:
                return (self.evaluationFunction(gameState), None)
            l = float("inf")
            Act = None
            for action in Actions:

                if (agentID == gameState.getNumAgents() - 1):
                    sucsValue = max_value(gameState.generateSuccessor(agentID, action), depth + 1)
                else:
                    sucsValue = min_value(gameState.generateSuccessor(agentID, action), agentID + 1, depth)

                sucsValue = sucsValue[0]
                if (sucsValue < l):
                    l, Act = sucsValue, action
            return (l, Act)

        max_value = max_value(gameState, 0)[1]
        return max_value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def max_value(gameState, depth, a, b):
            availableActions = gameState.getLegalActions(0)
            if len(availableActions) == 0 or gameState.isWin() or gameState.isLose() or depth == self.depth:
                return (self.evaluationFunction(gameState), None)
            v = (float("-inf"))
            action = ""

            for nextAction in availableActions:
                currValAct = (v, action)
                nextValAct = (min_value(gameState.generateSuccessor(0, nextAction), 1, depth, a, b)[0], nextAction)
                (v, action) = max(currValAct, nextValAct, key=lambda x: x[0])
                if v > b:
                    return (v, action)
                a = max(a, v)
            return (v, action)

        def min_value(gameState, ghostID, depth, a, b):
            availableActions = gameState.getLegalActions(ghostID)
            if len(availableActions) == 0:
                return (self.evaluationFunction(gameState), None)
            v = float("inf")
            action = ""
            for nextAction in availableActions:
                currValAct = (v, action)
                nextValAct = (
                    max_value(gameState.generateSuccessor(ghostID, nextAction), depth + 1, a, b)[0], nextAction) \
                    if (ghostID == gameState.getNumAgents() - 1) else \
                    (min_value(gameState.generateSuccessor(ghostID, nextAction), ghostID + 1, depth, a, b)[0],
                     nextAction)
                (v, action) = min(currValAct, nextValAct, key=lambda x: x[0])

                if (v < a):
                    return (v, action)
                b = min(b, v)

            return (v, action)

        a = -(float("inf"))
        b = float("inf")
        pacman_start = max_value(gameState, 0, a, b)[1]
        return pacman_start


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, game_state):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

    def get_value(self, game_state, index, depth):
        """
        Returns value as pair of [action, score] based on the different cases:
        1. Terminal state
        2. Max-agent
        3. Expectation-agent
        """


from searchAgents import mazeDistance


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"


def holeEvaluationFunction(currentGameState: GameState):
    posPacman = currentGameState.getPacmanPosition()
    posFood = currentGameState.getFood().asList()
    posGhosts = currentGameState.getGhostStates()

    if currentGameState.isLose():
        return float("-inf")

    if currentGameState.isWin():
        return float("inf")

    posJump = currentGameState.getJump()
    if not posJump == None:
        distJump = mazeDistance(posPacman, posJump, currentGameState)
    else:
        distJump = 0

    minFoodDist = min([mazeDistance(food, posPacman, currentGameState) for food in posFood])

    GhDistList = [manhattanDistance(posPacman, ghost.getPosition()) for ghost in posGhosts if
                  ghost.scaredTimer == 0]

    minGhDist = min(GhDistList) if len(GhDistList) > 0 else -1


    score = scoreEvaluationFunction(currentGameState)
    score -= 20 * distJump + 1.5 * minFoodDist + 2 * (1.0 / minGhDist ) + 100 * len(posFood)
    return score


# Abbreviation
better = betterEvaluationFunction
holeEval = holeEvaluationFunction

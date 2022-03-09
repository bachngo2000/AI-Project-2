# multiAgents.py
# --------------
# This codebase is adapted from UC Berkeley AI. Please see the following information about the license.
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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newFoodList = newFood.asList()
        newFoodSize = len(newFoodList) # number of remaining foods to eat in the proposed successor game state #
        NumFoodToEat = currentGameState.getFood().count() # number of foods in the current game state #

        if NumFoodToEat == newFoodSize:
            distance = 10000
            for pellet in newFoodList:
                # if the Manhattan distance between a single pellet and the Pacman position after moving is less than
                # the distance, update the distance with the Manhattan distance's value
                if manhattanDistance(pellet, newPos) < distance:
                    distance = manhattanDistance(pellet, newPos)
        else:
            distance = 0

        for ghost in newGhostStates:
            # Manhattan distance between the current position of a ghost and the Pacman position after moving
            mhd = manhattanDistance(ghost.getPosition(), newPos)
            distance = ((4 ** 3) / (4 ** mhd)) + distance
        return -abs(distance)
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        curr_depth = 0  # initialize it to zero for root
        max_depth = self.depth
        curr_agent = self.index
        value = self.minimax_helper(gameState, curr_agent, curr_depth, max_depth)[0]
        print("minimax value is {0}".format(value))
        action = self.minimax_helper(gameState, curr_agent, curr_depth, max_depth)[1]
        return action

    # calling a function within the same class
    def minimax_helper(self, gameState, curr_agent, curr_depth, max_depth):
        # base case:  when game is over or when the game tree reaches its max depth
        if gameState.getLegalActions(curr_agent) == [] or curr_depth == max_depth:
            val = self.evaluationFunction(gameState)
            return val, None

        # recursive case 1:
        # when it is pacman's turn the maximizer
        if curr_agent == 0:
            # pacman maximizer tries to update val to a max. possible value, so it is initialized to a very small number
            val = -100000000000.0

            # obtain the list of possible actions for the current agent to its successors
            legal_actions = gameState.getLegalActions(0)  # agent index for pacman is zero

            # finding whose turn is to make the next move in the successor state
            next_agent = curr_agent + 1

            # find the max among all the successors
            for action in legal_actions:
                successor = gameState.generateSuccessor(0, action)
                if val < self.minimax_helper(successor, next_agent, curr_depth, max_depth)[0]:
                    # update max value
                    val = self.minimax_helper(successor, next_agent, curr_depth, max_depth)[0]
                    action_to_max = action

            return val, action_to_max

        else:  # curr_agent > 0: when it is a ghost turn
            # ghost minimizer tries to update val to a min. possible value, so it is initialized to a very large number
            val = +100000000000.0

            # obtain the list of possible actions for the current agent to its successors
            legal_actions = gameState.getLegalActions(curr_agent)  # agent index for ghosts > 0

            # finding whose turn is to make the next move in the successor state
            next_agent = curr_agent + 1
            # if next agent index is beyond the last ghost index
            if next_agent == gameState.getNumAgents():  # number of agents is one more that the last ghost index
                # make the pacman the next agent
                next_agent = 0
                # increase the depth of the tree
                curr_depth += 1

            # find the min among all the successors
            for action in legal_actions:
                successor = gameState.generateSuccessor(curr_agent, action)
                if val > self.minimax_helper(successor, next_agent, curr_depth, max_depth)[0]:
                    # update min value
                    val = self.minimax_helper(successor, next_agent, curr_depth, max_depth)[0]
                    action_to_min = action

            return val, action_to_min

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        """
                Returns the minimax action using self.depth and self.evaluationFunction
                """
        "*** YOUR CODE HERE ***"
        curr_depth = 0  # initialize it to zero for root
        max_depth = self.depth
        curr_agent = self.index
        alpha = -100000000000.0
        beta = 100000000000.0
        value = self.alphabeta_helper(gameState, curr_agent, curr_depth, max_depth, alpha, beta)[0]
        print("alphabeta value is {0}".format(value))
        action = self.alphabeta_helper(gameState, curr_agent, curr_depth, max_depth, alpha, beta)[1]
        return action

        # calling a function within the same class

        # take
        # return

    def alphabeta_helper(self, gameState, curr_agent, curr_depth, max_depth, alpha, beta):
        # base case:  when game is over or when the game tree reaches its max depth
        if gameState.getLegalActions(curr_agent) == [] or curr_depth == max_depth:
            val = self.evaluationFunction(gameState)
            return val, None

        # recursive case 1:
        # when it is pacman's turn the maximizer
        if curr_agent == 0:
            # pacman maximizer tries to update val to a max. possible value, so it is initialized to a very small number
            val = -100000000000.0

            # obtain the list of possible actions for the current agent to its successors
            legal_actions = gameState.getLegalActions(0)  # agent index for pacman is zero

            # finding whose turn is to make the next move in the successor state
            next_agent = curr_agent + 1

            # find the max among all the successors
            for action in legal_actions:
                successor = gameState.generateSuccessor(0, action)
                if val < self.alphabeta_helper(successor, next_agent, curr_depth, max_depth, alpha, beta)[0]:
                    # update max value
                    val = self.alphabeta_helper(successor, next_agent, curr_depth, max_depth, alpha, beta)[0]
                    action_to_max = action

                # compare value to the stored beta i.e. the best possible outcome so far of the minimizer
                if val > beta:
                    return val, action_to_max

                # if not continue the search but update alpha
                alpha = max(alpha, val)
            return val, action_to_max

        else:  # curr_agent > 0: when it is a ghost turn
            # ghost minimizer tries to update val to a min. possible value, so it is initialized to a very large number
            val = +100000000000.0

            # obtain the list of possible actions for the current agent to its successors
            legal_actions = gameState.getLegalActions(curr_agent)  # agent index for ghosts > 0

            # finding whose turn is to make the next move in the successor state
            next_agent = curr_agent + 1
            # if next agent index is beyond the last ghost index
            if next_agent == gameState.getNumAgents():  # number of agents is one more that the last ghost index
                # make the pacman the next agent
                next_agent = 0
                # increase the depth of the tree
                curr_depth += 1

            # find the min among all the successors
            for action in legal_actions:
                successor = gameState.generateSuccessor(curr_agent, action)
                if val > self.alphabeta_helper(successor, next_agent, curr_depth, max_depth, alpha, beta)[0]:
                    # update min value
                    val = self.alphabeta_helper(successor, next_agent, curr_depth, max_depth, alpha, beta)[0]
                    action_to_min = action

                # compare value to the stored alpha i.e. the best possible outcome so far of the maximizer
                if val < alpha:
                    return val, action_to_min

                # if not continue the search but update beta of the minimizer
                beta = min(beta, val)

            return val, action_to_min

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
        curr_depth = 0  # initialize it to zero for root
        max_depth = self.depth
        curr_agent = self.index
        value = self.expectimax_helper(gameState, curr_agent, curr_depth, max_depth)[0]
        print("expectimax value is {0}".format(value))
        action = self.expectimax_helper(gameState, curr_agent, curr_depth, max_depth)[1]
        return action

        # calling a function within the same class

    def expectimax_helper(self, gameState, curr_agent, curr_depth, max_depth):
        # base case:  when game is over or when the game tree reaches its max depth
        if gameState.getLegalActions(curr_agent) == [] or curr_depth == max_depth:
            val = self.evaluationFunction(gameState)
            return val, None

        # recursive case 1:
        # when it is pacman's turn the maximizer
        if curr_agent == 0:
            # pacman maximizer tries to update val to a max. possible value, so it is initialized to a very small number
            val = -100000000000.0

            # obtain the list of possible actions for the current agent to its successors
            legal_actions = gameState.getLegalActions(0)  # agent index for pacman is zero

            # finding whose turn is to make the next move in the successor state
            next_agent = curr_agent + 1

            # find the max among all the successors
            for action in legal_actions:
                successor = gameState.generateSuccessor(0, action)
                if val < self.expectimax_helper(successor, next_agent, curr_depth, max_depth)[0]:
                    # update max value
                    val = self.expectimax_helper(successor, next_agent, curr_depth, max_depth)[0]
                    action_to_max = action

            return val, action_to_max

        else:  # curr_agent > 0: when it is a ghost turn
            # ghost is a random agent.
            # All ghosts are modeled as choosing uniformly at random from their legal moves.
            val = 0

            # obtain the list of possible actions for the current agent to its successors
            legal_actions = gameState.getLegalActions(curr_agent)  # agent index for ghosts > 0

            # finding whose turn is to make the next move in the successor state
            next_agent = curr_agent + 1
            # if next agent index is beyond the last ghost index
            if next_agent == gameState.getNumAgents():  # number of agents is one more that the last ghost index
                # make the pacman the next agent
                next_agent = 0
                # increase the depth of the tree
                curr_depth += 1

            # find average value among all the successors
            for action in legal_actions:
                successor = gameState.generateSuccessor(curr_agent, action)
                # calculate the sum of values over the successors
                val = val + self.expectimax_helper(successor, next_agent, curr_depth, max_depth)[0]
            # calculate the expectation or the average since all actions have the same probability
            val = val / len(legal_actions)

            # all legal actions have equal probability of being selected
            action_expectimax = random.choice(legal_actions)
            return val, action_expectimax

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # variable declarations #
    evalVal = 0  # Evaluation value set to 0 initially
    foodList = currentGameState.getFood().asList()  # list of all remaining foods in the current state
    foodNum = len(foodList)  # number of foods remaining in the current state
    position = currentGameState.getPacmanPosition()  # current position of Pacman in the current state
    capsules = len(currentGameState.getCapsules())  # number of capsules (big white dots)
    blueGhostsList = []  # List of blue ghosts that Pacman can eat for extra points
    activeGhostsList = []  # List of  active ghosts that can eat Pacman
    score = currentGameState.getScore()  # current score
    scoreWeight = 1.5
    foodWeight = -5
    capsulesWeight = -30

    evalVal = evalVal + (scoreWeight * score)

    # 5 points will be rewarded to Pacman every time it eats a pellet/food. Every time that happens, our evaluation
    # value gets better in the proposed successor state since the remaining food is less
    evalVal = evalVal + (foodWeight * foodNum)

    # Everytime Pacman eats a ghost, it gains 30 points. Our objective is to first eat a capsule, since by doing so, we
    # can eat a ghost. So we try to make Pacman eat capsules more frequently that pellets
    # Weight of capsules > weight of pellets (food)
    evalVal = evalVal + (capsulesWeight * capsules)

    # finding blue ghosts and active ghosts #
    for ghost in currentGameState.getGhostStates():
        if ghost.scaredTimer:  # blue ghosts
            blueGhostsList.append(ghost)
        else:  # active ghosts
            activeGhostsList.append(ghost)

    # List of distances of the Pacman position from a food/pellet
    food_pacman_Distances = []
    # List of distances of the Pacman position from an active ghost
    activeGhosts_pacman_Distances = []
    # List of distances of the Pacman position from a blue/scared ghost
    blueGhosts_pacman_Distances = []

    # Find distances #
    for pellet in foodList:
        pelletDistance = manhattanDistance(position, pellet)
        food_pacman_Distances.append(pelletDistance)

    for ghost in activeGhostsList:
        scaredGhostDistance = manhattanDistance(position, ghost.getPosition())
        blueGhosts_pacman_Distances.append(scaredGhostDistance)

    for ghost in blueGhostsList:
        blueGhostsDistance = manhattanDistance(position, ghost.getPosition())
        blueGhosts_pacman_Distances.append(blueGhostsDistance)

    # Update evaluation value based on food distances
    # weight of food very close to the Pacman: -1
    # weight of food quite close to the Pacman: -0.5
    # weight of food far away from the Pacman: -0.1
    for distance in food_pacman_Distances:
        if distance < 3:
            evalVal = evalVal + (-1 * distance)
        if distance < 6:
            evalVal = evalVal + (-0.5 * distance)
        else:
            evalVal = evalVal + (-0.1 * distance)

    # Update evaluation value based on active ghosts distances
    # weight of ghost nearby: 3
    # weight of ghost quite close by: 2
    # weight of distant ghost: 0.5
    for distance in activeGhosts_pacman_Distances:
        if distance < 3:
            evalVal = evalVal + (3 * distance)
        elif distance < 6:
            evalVal = evalVal + (2 * distance)
        else:
            evalVal = evalVal + (0.5 * distance)


    # Update evaluation based on blue ghosts distances
    # Close scared ghosts weight: -25
    # Quite close scared ghosts weight: -15
    for distance in blueGhosts_pacman_Distances:
        if distance < 3:
            evalVal = evalVal + (-25 * distance)
        else:
            evalVal = evalVal + (-15 * distance)

    return evalVal

# Abbreviation
better = betterEvaluationFunction

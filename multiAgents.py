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

    # take: current game state and a leagl saction at that state
    # return: the value of that state as a linear function of current score, distance to the closest dot,
    # the distance to the closest ghost, and whether ghost is scared or not

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
        print("newPos {}".format(newPos))
        newFood = successorGameState.getFood()
        print("newFood {}".format(newFood))
        newGhostStates = successorGameState.getGhostStates()
        print("newGhostStates {}".format(newGhostStates))
        print(newGhostStates[0].getPosition())
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        print("newScaredTimes {}".format(newScaredTimes))

        "*** YOUR CODE HERE ***"

        # Eval(s) = w1.f1(s) + w2.f2(s) + w3.f3(s) + w4.f4(s)
        # f1(s) = the current score
        f1 = successorGameState.getScore()

        # f2(s) = the distance to the closest food (Manhattan Distance)

        f2 = newPos[0]

        # f3(3) = the distance to the closest ghost

        # f4(s) = whether ghost is scared or not



        return successorGameState.getScore()

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
        curr_depth = 0   # initialize it to zero for root
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
            if next_agent == gameState.getNumAgents():   # number of agents is one more that the last ghost index
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
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

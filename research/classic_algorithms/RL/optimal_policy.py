from personalised_rewards import get_personalised_params


class Policy:
    def __init__(self, values, policy):
        self.values = values
        self.policy = policy

    def print_values(self):
        print("Optimal Values")
        for i in range(len(self.values)):
            for j in range(len(self.values[i])):
                if self.values[i][j] is not None:
                    print('{:.2f}'.format(self.values[i][j]), end='')
                else:
                    print('{:>4}'.format('.'), end='')
                print('\t', end='')
            print()

    def print_policy(self):
        arrows = ['↑', '→', '↓', '←']
        print("Optimal Policy")
        for i in range(len(self.policy)):
            for j in range(len(self.policy[i])):
                if self.policy[i][j] is None:
                    print('.', end='')
                elif type(self.policy[i][j]) is int:
                    print(arrows[self.policy[i][j]], end='')
                else:
                    print(self.policy[i][j], end='')
                print('\t', end='')
            print()


class World:
    EPSILON = 0.0001

    def __init__(self, grid, movement_reward, penalty_reward, goal_reward,
                 directions, penalty_state, reward_state, p, gamma):
        self.grid = grid
        self.movement_reward = movement_reward
        self.penalty_reward = penalty_reward
        self.goal_reward = goal_reward
        self.directions = directions
        self.penalty_state = penalty_state
        self.reward_state = reward_state
        self.p = p
        self.gamma = gamma

    @staticmethod
    def get_default(cid):
        reward_state, p, gamma = get_personalised_params(cid)
        penalty_state = 11
        grid = [
            [1, 2, 3, 4],
            [5, 6, 0, 7],
            [0, 8, 9, 10],
            [0, 0, 11, 0]
        ]
        movement_reward = -1
        penalty_reward = -100
        goal_reward = 10
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        # OVERRIDES
        #gamma = 0.4
        #p = 0.25
        #reward_state = 1

        return World(grid, movement_reward, penalty_reward, goal_reward,
                     directions, penalty_state, reward_state, p, gamma)

    def get_optimal_policy(self):
        '''
        Applies the Value Iteration Algorithm to the grid world
        and returns the optimal policy
        '''
        # init
        policy = Policy([[None for _ in range(len(self.grid[i]))]
                         for i in range(len(self.grid))],
                        [[None for _ in range(len(self.grid[i]))]
                         for i in range(len(self.grid))])

        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                if self.grid[i][j] > 0:
                    policy.values[i][j] = 0

        # find optimal values
        delta = float('-inf')
        while abs(delta) > self.EPSILON:
            delta = 0
            for i in range(len(self.grid)):
                for j in range(len(self.grid[i])):
                    if self.grid[i][j] > 0 \
                            and self.grid[i][j] != self.penalty_state \
                            and self.grid[i][j] != self.reward_state:
                        old_value = policy.values[i][j]
                        policy.values[i][j] = \
                            self.get_q(policy, (i, j),
                                       self.get_max_action(policy, (i, j)))
                        delta = max(delta,
                                    abs(old_value - policy.values[i][j]))

        # get policy from optimal values
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                if self.grid[i][j] == self.penalty_state:
                    policy.policy[i][j] = 'P'
                elif self.grid[i][j] == self.reward_state:
                    policy.policy[i][j] = 'G'
                elif self.grid[i][j] > 0:
                    policy.policy[i][j] = self.get_max_action(policy, (i, j))

        return policy

    def move(self, action, state):
        '''
        retruns new state after move
        if can't move returns same state
        '''
        new_state = tuple(map(lambda dir_coo: dir_coo[0] + dir_coo[1],
                              zip(state, self.directions[action])))
        # not within bounds
        if not (0 <= new_state[0] < len(self.grid) and
                            0 <= new_state[1] < len(self.grid[0])):
            return state
        elif self.grid[new_state[0]][new_state[1]] <= 0:
            return state
        else:
            return new_state

    def get_q(self, policy, state, action):
        '''
        returns the q value for given action and state
        '''
        val = 0
        other_p = (1 - self.p) / (len(self.directions) - 1)
        for i in range(len(self.directions)):
            prob = other_p
            if i == action:
                prob = self.p

            new_state = self.move(i, state)
            state_index = self.grid[new_state[0]][new_state[1]]
            current_reward = self.movement_reward
            if state_index == self.penalty_state:
                current_reward = self.penalty_reward
            elif state_index == self.reward_state:
                current_reward = self.goal_reward

            new_value = policy.values[new_state[0]][new_state[1]]
            val = val + prob * (current_reward + self.gamma * new_value)
        return val

    def get_max_action(self, policy, state):
        '''
        returns the action that maximises q
        '''
        action = 0
        max_val = float('-inf')
        for a in range(len(self.directions)):
            new_val = self.get_q(policy, state, a)
            if new_val > max_val:
                max_val = new_val
                action = a
        return action


if __name__ == '__main__':
    CID = [0, 1, 0, 7, 9, 9, 3, 1]
    world = World.get_default(CID)
    policy = world.get_optimal_policy()
    policy.print_values()
    print()
    policy.print_policy()
    '''
    Output: 
    Optimal Values
    1.34	7.14	0.00	7.43	
    -0.57	1.22	   .	1.41	
       .	-1.26	-11.87	-1.20	
       .	   .	0.00	   .	
    
    Optimal Policy
    →	→	G	←	
    ↑	↑	.	↑	
    .	↑	→	↑	
    .	.	P	.
    '''

from problem import Problem
import numpy as np
from random import choice, seed
import itertools


class Room:
    def __init__(self, position, clean):
        #Position is a vector with x and y coordinate
        self.position = position
        self.clean = clean


class Building:
    def __init__(self, size, dirt_rate, rooms=None, seed=None):
        self.size = size
        if seed is not None:
            np.random.seed(seed)
        if rooms is not None:
            self.rooms = rooms
        else:
            self.rooms = np.random.choice([0, 1], size=size[0]*size[1], p=[dirt_rate, 1-dirt_rate])
            self.rooms.resize(size[0], size[1])
        self.dirty_rooms = self.rooms.size-self.rooms.sum()

    def copy(self):
        building_copy = Building(self.size, 0, rooms=np.copy(self.rooms))
        return building_copy

    def clean(self, position):
        if not self.rooms[position]:
            self.dirty_rooms -= 1
            self.rooms[position] = 1

    def make_dirty(self, position):
        if self.rooms[position]:
            self.dirty_rooms += 1
            self.rooms[position] = 0

    def exist_room(self, position):
        if position[0] >= 0 and position[0] < self.rooms.shape[0] and position[1] >= 0 and position[1] < self.rooms.shape[1]:
            return True
        else:
            return False


class VC(Problem):
    def __init__(self, size, dirt_rate=0.6, seed_value=None, position=None, energy_spend=0, random_action_rate=0):
        #Position is a vector with x and y coordinate
        self.dirt_rate = dirt_rate
        self.seed_value = seed_value
        self.building = Building(size,  self.dirt_rate, seed=self.seed_value)
        if position is None:
            self.position = choice([[x, y] for x in range(self.building.size[0]) for y in range(self.building.size[1])])
        else:
            self.position = list(position)
        self.energy_spend = energy_spend
        self.state = None
        self.random_action_rate = random_action_rate
        self.actions={"right":(VC.move_right, VC.move_left),
                      "left":(VC.move_left,VC.move_right),
                      "up":(VC.move_up,VC.move_down),
                      "down":(VC.move_down,VC.move_up),
                      "clean":(VC.clean,VC.make_dirty)}

    def copy(self):
        position_copy = self.position[:]
        building_copy = self.building.copy()
        vc_copy = VC(position_copy, building_copy, self.energy_spend)
        return vc_copy

    def act(self, action):
        random_action = np.random.random()<self.random_action_rate
        if random_action:
            self.actions[action][1](self)
        else:
            self.actions[action][0](self)
        self.state = None
        self.to_state()

    def move(self, new_position):
        self.energy_spend += 1
        if self.building.exist_room(new_position):
            self.position = new_position
            return True
        else:
            return False

    def move_right(self):
        new_position = [self.position[0] + 1, self.position[1]]
        return self.move(new_position)

    def move_left(self):
        new_position = [self.position[0] - 1, self.position[1]]
        return self.move(new_position)

    def move_up(self):
        new_position = [self.position[0], self.position[1] - 1]
        return self.move(new_position)

    def move_down(self):
        new_position = [self.position[0], self.position[1] + 1]
        return self.move(new_position)

    def clean(self):
        self.energy_spend += 1
        self.building.clean(tuple(self.position))

    def make_dirty(self):
        self.energy_spend += 1
        self.building.make_dirty(tuple(self.position))

    def to_state(self):
        if self.state is None:
            state = []
            for x in np.nditer(self.building.rooms):
                state.append(int(x))
            state.append(self.position[0])
            state.append(self.position[1])
            self.state = tuple(state)
        return self.state

    def is_goal_state(self, state):
        if state.building.rooms.sum() == state.building.rooms.size:
            return True
        else:
            return False

    def get_applicable_actions(self, state):
        actions = []
        actions.append("clean")
        if state.position[0] + 1 < state.building.size[0]:
            actions.append("right")
        if state.position[0] - 1 >= 0:
            actions.append("left")
        if state.position[1] + 1 < state.building.size[1]:
            actions.append("down")
        if state.position[1] - 1 >= 0:
            actions.append("up")
        return actions

    def eval(self, state):
        #consistent and dominant
        costs = 2 * (state.building.rooms.size - state.building.rooms.sum())
        if not state.building.rooms[state.position[0], state.position[1]]:
            costs -= 1
        return costs

    def get_all_actions(self):
        actions = sorted(list(self.actions.keys()))
        return actions

    def get_all_states(self):
        states = []
        #create all room configurations with respect to dirt
        rooms = [0]*(self.building.rooms.shape[0]*self.building.rooms.shape[1])
        permutations = set()
        permutations.add(tuple(rooms))
        for i, r in enumerate(rooms):
            rooms[i] = 1
            permutation = set(itertools.permutations(rooms))
            permutations = permutations.union(permutation)
        permutations = sorted(list(permutations))
        positions = []
        #create all possible positions
        for x in range(self.building.rooms.shape[0]):
            for y in range(self.building.rooms.shape[1]):
                positions.append((x,y))

        #combine each room configuration with all possible positions
        for room_configuration in permutations:
            state = []
            state.extend(list(room_configuration))
            for position in positions:
                state_complete = state[:]
                state_complete.extend(list(position))
                states.append(tuple(state_complete))
        return states

    def get_reward(self, state):
        reward = -self.eval(state)
        return reward

    def reset(self):
        self.building = Building(self.building.size, self.dirt_rate, seed=self.seed_value)
        self.position = choice([[x, y] for x in range(self.building.size[0]) for y in range(self.building.size[1])])
        self.energy_spend = 0
        self.state = None


class Environment:

    def __init__(self, size=(3, 3), dirt_rate=0.6, width_room=274, height_room=240, seed_value=None):

        if seed_value is not None:
            seed(seed_value)
        self.size = size
        self.problem = VC(self.size, dirt_rate, seed_value=seed_value, random_action_rate=0)
        self.width_room = width_room
        self.height_room = height_room

    def reset(self):
        self.problem.reset()

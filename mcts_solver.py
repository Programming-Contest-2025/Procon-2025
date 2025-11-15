import math
import random
import time
from collections import defaultdict
from FieldClass import Field, build_mapping
from Solver import Solver
from utils import generateField

random.seed(42)

# =============================
# Node definition
# =============================
class MCTSNode:
    def __init__(self, field, parent=None, parent_action=None):
        self.field = field              # Field object
        self.parent = parent            # parent node
        self.parent_action = parent_action
        self.children = {}              # action -> child node
        self.N = 0                      # visit count
        self.W = 0.0                    # total value
        self.Q = 0.0                    # mean value
        self.untried_actions = None     # action list

    def is_fully_expanded(self):
        return self.untried_actions is not None and len(self.untried_actions) == 0

    def best_child_ucb(self, cpuct=1.0):
        """Select best child according to UCB1 formula"""
        best_action, best_score = None, -float("inf")
        for action, child in self.children.items():
            ucb = child.Q + cpuct * math.sqrt(math.log(self.N + 1) / (1 + child.N))
            if ucb > best_score:
                best_action = action
                best_score = ucb
        return best_action, self.children[best_action]


# =============================
# Action generator (domain-specific)
# =============================
def default_action_generator(field, max_children=15):
    n = field.n
    solver = Solver(init_field=field)
    _, unpaired = solver.get_unpaired(field)
    if not unpaired:
        return [(0,0,2)]  # fallback
    
    # Tìm vùng chứa nhiều unpaired nhất
    heatmap = [[0]*n for _ in range(n)]
    for coords in unpaired:
        for (y, x) in coords:
            heatmap[y][x] += 1

    # Chọn vùng có tổng heatmap cao nhất
    best_actions = []
    for size in [2,3,4]:
        for y in range(n-size+1):
            for x in range(n-size+1):
                val = sum(heatmap[i][j] for i in range(y,y+size) for j in range(x,x+size))
                best_actions.append((val, (x,y,size)))
    best_actions.sort(reverse=True)
    return [a for _, a in best_actions[:max_children]]

# =============================
# MCTS class (using heuristic_SA)
# =============================
class MCTS:
    def __init__(
        self,
        solver: Solver,
        simulations=200,
        cpuct=1.0,
        max_children=15,
        rollout_steps=30,
        rollout_policy="heuristic"  # chỉ dùng heuristic_SA
    ):
        self.solver = solver
        self.simulations = simulations
        self.cpuct = cpuct
        self.max_children = max_children
        self.rollout_steps = rollout_steps
        self.rollout_policy = rollout_policy

    def search(self, root_field, time_limit=None):
        root = MCTSNode(root_field)
        root.untried_actions = default_action_generator(root_field, self.max_children)

        start_time = time.time()
        sims = 0
        while True:
            if time_limit and (time.time() - start_time) > time_limit:
                break
            if not time_limit and sims >= self.simulations:
                break
            self._simulate_once(root)
            sims += 1

        # chọn child có visit lớn nhất
        if not root.children:
            return root_field, []
        best_action, best_child = max(root.children.items(), key=lambda kv: kv[1].N)
        # reconstruct best path
        path = []
        node = root
        while node.children:
            action, node = max(node.children.items(), key=lambda kv: kv[1].N)
            path.append(action)
        return node.field, path

    def _simulate_once(self, root):
        # 1️⃣ SELECTION
        node = root
        while node.is_fully_expanded() and node.children:
            _, node = node.best_child_ucb(cpuct=self.cpuct)

        # 2️⃣ EXPANSION
        if node.untried_actions is None:
            node.untried_actions = default_action_generator(node.field, self.max_children)

        if node.untried_actions:
            action = node.untried_actions.pop(0)
            new_field = node.field.rotate(*action)
            child = MCTSNode(new_field, parent=node, parent_action=action)
            node.children[action] = child
            node = child

        # 3️⃣ SIMULATION (evaluation)
        reward = self.evaluate(node.field)

        # 4️⃣ BACKPROPAGATION
        self._backpropagate(node, reward)

    def evaluate(self, field):
        # SA rollout thay vì heuristic thuần
        current = field
        best_score = current.score()
        T = 1.0
        alpha = 0.9
        for _ in range(20):  # chỉ 20 bước rollout
            action = self.solver.random_action(field.n)
            new_field = current.rotate(*action)
            delta = new_field.score() - current.score()
            if delta > 0 or math.exp(delta / T) > random.random():
                current = new_field
                if current.score() > best_score:
                    best_score = current.score()
            T *= alpha
        return best_score * 10.0  # scale lên cho reward có trọng số


    def _backpropagate(self, node, reward):
        while node is not None:
            node.N += 1
            node.W += reward
            node.Q = node.W / node.N
            node = node.parent


# =============================
# Example usage
# =============================
if __name__ == "__main__":
    size = 12
    entities = generateField(n=size)
    mappings = build_mapping(size)
    field = Field(size=size, entities=entities, mappings=mappings)
    solver = Solver(init_field=field)

    mcts = MCTS(solver=solver, simulations=1000, cpuct=1.2, max_children=50)
    best_field, path = mcts.search(field, time_limit=50.0)

    print(f"Initial score: {field.score()}")
    print(f"Best score after MCTS: {best_field.score()}")
    print(f"Path length: {len(path)}")

import heapq, itertools, random, time, math
import numpy as np
from FieldClass import *
from concurrent.futures import ProcessPoolExecutor
from utils import generateField
import argparse
from typing import Tuple, List, Dict, Optional

random.seed(42)

inf_num = 10**12

class Solver:
    def __init__(self, init_field, max_depth = 500):
        self.field = init_field
        self.visited = set()
        self.max_depth = max_depth
        self.best_field = None
        self.best_score = -1
        self.parents = {} #current_hash : (parent_hash, action)    
    
    def dfs(self, field = None, depth = 0, path = None):
        if field is None:
            field = self.field
        if path is None:
            path = []
        
        
        if depth > self.max_depth:
            return
        
        h = field.hash()
        if h in self.visited:
            return   # chỉ bỏ qua nhánh này
        self.visited.add(h)
        
        current_score = field.score()
        if current_score > self.best_score:
            self.best_score = current_score
            self.best_field = field
            self.best_path = path[:]
        
        if current_score == (field.n * field.n) / 2:
            raise StopIteration

        for x in range(field.n):
            for y in range(field.n):
                for size in range(2, field.n - max(y, x) + 1):
                    new_field = field.rotate(x, y, size)
                    path.append((x, y, size))
                    self.dfs(new_field, depth + 1, path)
                    path.pop()

    def get_path(self, parents, current):
        #parent: state -> (previous state, action)
        paths = []
        k = current
        while k in parents and parents[k] is not None:
            previous_state_hash, action = parents[k]
            
            paths.append(action)
            k = previous_state_hash
        paths.reverse()
        return paths
        
    def get_unpaired(self, field):
        n = field.n
        positions = {}
        unpaired = []
        
        for y in range(n):
            for x in range(n):
                v = field.entities[y][x]
                positions.setdefault(v, []).append((y, x))
        
        for coords in positions.values():
            #Cặp tọa độ của 2 giá trị trùng nhau
            (y1, x1), (y2, x2) = coords
            manhattan_distance = abs(y1 - y2) + abs(x1 - x2)
            #Cặp tọa độ không kề nhau
            if manhattan_distance > 1: 
                unpaired.append(coords)
        return positions, unpaired
    
    def heuristic(self, field):
        n = self.field.n 
        entities = self.field.entities
        
        #Lấy tọa độ các cặp giá trị
        pairs, _ = self.get_unpaired(field=field)
        
        #Đếm số lượng giá trị có thể cải thiện trong vòng 1 bước xoay
        improve_count = 0 
        for v, coords in pairs.items():
            if len(coords) == 2:
                (y1, x1), (y2, x2) = coords
                dist = abs(x1 - x2) + abs(y1 - y2)
                if dist == 1:
                    continue #2 điểm này đã kề nhau
                if dist > 3:
                    continue #Không thể kề nhau trong 1 bước xoay
                
                #Thử xoay để kiểm tra có phải điểm tiềm năng
                found = False
                for size in [2, 3]:
                    for x in range(x1, x2):
                        for y in range(y1, y2):
                            #Chưa tối ưu cách kiểm tra điều kiện
                            if size + x - 1 >= n or size + y - 1 >= n:
                                continue
                            new_field = field.rotate(x, y, size)
                            new_pairs = []
                            for i in range(n):
                                for j in range(n):
                                    if new_field.entities[i][j] == v:
                                        new_pairs.append((i, j))
                            (nx1, ny1), (nx2, ny2) = new_pairs
                            if abs(nx1 - nx2) + abs(ny1 - ny2) == 1:
                                improve_count += 1
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
        improve_count = max(1, improve_count)
        #Can't admissible heuristic 
        #Lấy số ô còn lại cần cải thiện / số ô có thể cải thiện trong vòng 1 bước
        return ((n*n) // 2 - field.score()) / improve_count
    
    def heuristic_SA(self, field, en_area = None):
        n = self.field.n 
        
        score = 0
        #Xác định vùng ảnh hưởng
        if en_area is None:
            x_, y_, size_ = 0, 0, n
        else:
            x_, y_, size_ = en_area
        
        #Cộng điểm cho các ô kề nhau
        for i in range(y_, y_ + size_):
            for j in range(x_, x_ + size_):
                v = field.entities[i][j]
                if i + 1 < n and field.entities[i+1][j] == v:
                    score += 100
                if j + 1 < n and field.entities[i][j+1] == v:
                    score += 100
        
        #Bonus cho các block đúng nhiều
        #------

        #Trừ khoảng cách manhattan cho các cặp chưa kề
        position = {}
        for i in range(n):
            for j in range(n):
                v = field.entities[i][j]
                position.setdefault(v, []).append((i, j))
        for coords in position.values():
            (y1, x1), (y2, x2) = coords
            manhattan_distance = abs(y1 - y2) + abs(x1 - x2)
            if manhattan_distance > 1:
                score -= manhattan_distance 
        
        return score
        
    def a_star(self):
        start_field = self.field
        start_hash = start_field.incremental_hash()
        n = start_field.n
        target_score = (n * n) // 2
        
        #priority queue (f, g, counter, field)
        pq = []
        
        #Thêm counter để nó không so sánh field (object) -> tránh lỗi
        counter = itertools.count()  # bộ đếm duy nhất
        heapq.heappush(pq, (self.heuristic(start_field), 0, next(counter), start_field))
        self.parents[start_hash] = None
        
        g_score_for_states = {start_hash : 0}
        
        step = 0
        
        while pq:
            f, g, _, field = heapq.heappop(pq)
            
            #Kiểm tra trạng thái đã đi qua chưa
            current_hash_score = field.incremental_hash()
            if g > g_score_for_states.get(current_hash_score, float("inf")):
                continue
            
            #Có thể tối ưu Score bằng cách tính trên "En" bị ảnh hưởng
            field_score = field.score()
            if field_score > self.best_score:
                self.best_score = field_score
                self.best_field = field
            
            if field_score == target_score:
                print("Đã tìm thấy trạng thái tối ưu")
                print(f"Cần loop {step} lần")
                return self.best_field ,self.get_path(self.parents, current_hash_score)

            #Nếu nhánh hiện tại đủ sâu thì bỏ qua, không đi xuống nữa
            if g >= self.max_depth:
                continue
            
            #Mở rộng nhánh
            for x in range(field.n):
                for y in range(field.n):
                    for size in range(2, field.n - max(x, y) + 1):
                        new_field = field.rotate(x, y, size)
                        
                        new_hash_score = new_field.incremental_hash()
                        g2 = g + 1
                        
                        #Cập nhật
                        if g2 < g_score_for_states.get(new_hash_score, float("inf")):
                            g_score_for_states[new_hash_score] = g2
                            h2 = self.heuristic(new_field)
                            f2 = g2 + h2 
                            heapq.heappush(pq, (f2, g2, next(counter), new_field))
                            self.parents[new_hash_score] = (current_hash_score, (x, y, size))
            
            #Kiểm tra xem duyệt bao nhiêu bước
            step += 1
        
        print("Không tìm thấy nghiệm")
        return None

    def beamSearch(self, field, max_iters=1000, beam_width=3):
        """
        Improved greedy pairing using beam search idea.
        - beam_width: số lượng move tốt nhất giữ lại ở mỗi vòng
        """
        steps = []
        best_score = self.heuristic(field)

        for _ in range(max_iters):
            _, unpaired_coords = self.get_unpaired(field)
            if not unpaired_coords:
                break

            candidate_moves = []

            for coords in unpaired_coords:
                (y1, x1), (y2, x2) = coords
                min_x, max_x = min(x1, x2), max(x1, x2)
                min_y, max_y = min(y1, y2), max(y1, y2)

                for size in [2, 3, 4]:
                    for i in range(max(0, max_y - size + 1), 
                                min(min_y + 1, field.n - size + 1)):
                        for j in range(max(0, max_x - size + 1), 
                                    min(min_x + 1, field.n - size + 1)):
                            new_field = field.rotate(j, i, size)
                            new_score = self.heuristic(new_field)
                            delta = new_score - best_score
                            candidate_moves.append((delta, (j, i, size), new_field))

            if not candidate_moves:
                break

            # Sắp xếp move theo delta giảm dần
            candidate_moves.sort(reverse=True, key=lambda x: x[0])

            # Chọn 1 trong top beam_width move (nếu tất cả delta <= 0 thì chọn move ít xấu nhất)
            chosen = random.choice(candidate_moves[:beam_width])
            delta, move, new_field = chosen

            steps.append(move)
            field = new_field
            best_score += delta

        return field, steps

    def random_action(self, n):
        #Cần tối ưu chọn size nhỏ, thoát local optimal mới chọn size lớn
        size = random.randint(2, n)
        x = random.randint(0, n - size)
        y = random.randint(0, n - size)
        return (x, y, size)
    
    def simulated_annealing(self, max_neiborghs = 100, T_start = 1.0, T_min = 0.01, alpha = 0.995):
        current_field = self.field
        n = current_field.n 
        current_heuristic = self.heuristic_SA(current_field)
        best_field = current_field
        best_heuristic = current_heuristic
        steps = []
        
        T = T_start
        while T > T_min:
            iters = 1
            while iters <= max_neiborghs:
                #Mở rộng nhánh
                action = self.random_action(n)
                new_field = current_field.rotate(*action)
                new_heuristic = self.heuristic_SA(new_field)
                
                delta = new_heuristic - current_heuristic
                
                if delta > 0:
                    current_field = new_field
                    current_heuristic = new_heuristic
                    steps.append(action)
                    
                    if new_heuristic > best_heuristic:
                        best_field = new_field
                        best_heuristic = new_heuristic
                
                else:
                    #Xác suất chấp nhận cái mới
                    appreciate_probability = math.exp(delta / T)
                    if random.random() < appreciate_probability:
                        current_field = new_field
                        current_heuristic = new_heuristic
                        steps.append(action)
                iters += 1
            T = T * alpha
        
        return best_field, steps

    import math, random

    def simulated_annealing_record(
        self, 
        max_neiborghs=100, 
        T_start=1.0, 
        T_min=0.01, 
        alpha=0.995,
    ):
        
        current_field = self.field
        n = current_field.n
        current_heuristic = self.heuristic_SA(current_field)
        best_heuristic = current_heuristic
        best_field = current_field

        T = T_start
        step_counter = 0

        # dùng yield để trả dần state ra ngoài
        yield current_field.entities, best_field.score()  # ghi state ban đầu

        while T > T_min:
            for _ in range(max_neiborghs):
                step_counter += 1
                action = self.random_action(n)
                new_field = current_field.rotate(*action)
                new_heuristic = self.heuristic_SA(new_field)

                delta = new_heuristic - current_heuristic

                if delta > 0:
                    current_field = new_field
                    current_heuristic = new_heuristic
                    yield current_field.entities, current_field.score()
                    
                    # cập nhật best
                    if new_heuristic > best_heuristic:
                        best_field = new_field
                        best_heuristic = new_heuristic

                else:
                    #Xác suất chấp nhận cái không tốt
                    appreciate_probability = math.exp(delta / T)
                    if random.random() < appreciate_probability:
                        current_field = new_field
                        current_heuristic = new_heuristic
                        yield current_field.entities, current_field.score()

            T *= alpha
    
    def _convert_field_to_tuple(self, field) -> Tuple[int, ...]:
        """Convert Field entities to immutable tuple"""
        return tuple(field.entities[i][j] 
                    for i in range(field.n) 
                    for j in range(field.n))
    
    def _create_zobrist_table(self, n: int) -> np.ndarray:
        """Tạo Zobrist hashing table"""
        max_value = (n * n) // 2
        return np.random.randint(0, 2**63, size=(n * n, max_value), dtype=np.int64)
    
    def _compute_zobrist_hash(self, board_tuple: Tuple[int, ...], zobrist_table: np.ndarray) -> int:
        """Tính Zobrist hash cho board state"""
        h = 0
        for pos, value in enumerate(board_tuple):
            h ^= zobrist_table[pos][value]
        return int(h)
    
    def _tuple_to_field(self, board_tuple: Tuple[int, ...], n: int) -> 'Field':
        """Convert tuple back to Field object"""
        entities = []
        for i in range(n):
            entities.append(list(board_tuple[i * n:(i + 1) * n]))
        return Field(size=n, entities=entities, mappings=self.field.mappings)
    
    def heuristic_optimized(self, field) -> float:
        """
        Optimized heuristic: Manhattan distance estimation
        """
        positions, unpaired = self.get_unpaired(field)
        
        if not unpaired:
            return 0.0
        
        total_distance = 0
        for coords in unpaired:
            (y1, x1), (y2, x2) = coords
            total_distance += abs(y1 - y2) + abs(x1 - x2)
        
        # Ước lượng số move cần thiết
        estimated_moves = total_distance / 2.5
        return estimated_moves
    
    def OptimizedAStar(self, weight: float = 1.5, time_limit: int = 300,
                      stuck_threshold: int = 5000, sa_iterations: int = 50,
                      max_depth: int = 100) -> Tuple:
        """
        Hybrid Optimized A* với Simulated Annealing escape
        
        Features:
        1. Tuple-based state (immutable, hashable)
        2. Zobrist Hashing (incremental hash O(k²))
        3. Min-Heap priority queue
        4. Dictionary-based visited set với g_score tracking
        5. SA escape khi stuck
        
        Args:
            weight: Trọng số cho weighted A* (>1.0 để tăng tốc)
            time_limit: Giới hạn thời gian (giây)
            stuck_threshold: Số nodes không cải thiện -> trigger SA
            sa_iterations: Số iterations SA khi escape
            max_depth: Giới hạn độ sâu tìm kiếm
        
        Returns:
            (best_field, solution_path)
        """
        print(f"\n{'='*70}")
        print(f"🚀 OPTIMIZED A* WITH HYBRID SA ESCAPE")
        print(f"{'='*70}")
        print(f"Weight: {weight}, Time limit: {time_limit}s")
        print(f"Stuck threshold: {stuck_threshold}, SA iterations: {sa_iterations}")
        print(f"Max depth: {max_depth}")
        print()
        
        start_time = time.time()
        n = self.field.n
        target_score = (n * n) // 2
        
        # Khởi tạo Zobrist table
        zobrist_table = self._create_zobrist_table(n)
        
        # Initialize tracking
        counter = itertools.count()
        current_start_field = self.field
        global_path = []
        restart_count = 0
        
        # Statistics
        total_nodes_explored = 0
        total_nodes_generated = 0
        
        while time.time() - start_time < time_limit:
            # === A* PHASE ===
            print(f"\n🎯 A* Phase (Restart #{restart_count})")
            
            # Convert current field to tuple state
            start_tuple = self._convert_field_to_tuple(current_start_field)
            start_hash = self._compute_zobrist_hash(start_tuple, zobrist_table)
            
            # Priority queue: (f_score, counter, g_score, field, field_tuple, zobrist_hash)
            open_set = []
            heapq.heappush(open_set, (0, next(counter), 0, current_start_field, start_tuple, start_hash))
            
            # visited: hash -> g_score
            # parent: hash -> (parent_hash, action)
            visited = {start_hash: 0}
            parent = {start_hash: None}
            
            best_score = current_start_field.score()
            best_field = current_start_field
            nodes_since_improvement = 0
            
            nodes_explored = 0
            nodes_generated = 0
            astar_stuck = False
            
            last_print_time = time.time()
            
            while open_set and time.time() - start_time < time_limit:
                # Progress update
                current_time = time.time()
                if current_time - last_print_time >= 5.0:
                    print(f"  [{current_time - start_time:.1f}s] Explored: {nodes_explored:,}, "
                          f"Queue: {len(open_set):,}, Best: {best_score}/{target_score}")
                    last_print_time = current_time
                
                f, _, g, current_field, current_tuple, current_hash = heapq.heappop(open_set)
                nodes_explored += 1
                
                # Skip if outdated
                if g > visited.get(current_hash, float('inf')):
                    continue
                
                current_score = current_field.score()
                
                # Goal check
                if current_score == target_score:
                    print(f"\n✅ Perfect solution found!")
                    path = self._reconstruct_path_from_parent(current_hash, parent)
                    final_path = global_path + path
                    
                    print(f"\n{'='*70}")
                    print(f"🎉 SUCCESS - Perfect solution!")
                    print(f"{'='*70}")
                    print(f"Final score: {current_score}/{target_score}")
                    print(f"Total restarts: {restart_count}")
                    print(f"Solution steps: {len(final_path)}")
                    print(f"Nodes explored: {total_nodes_explored + nodes_explored:,}")
                    print(f"Time: {time.time() - start_time:.2f}s")
                    print(f"{'='*70}\n")
                    
                    return current_field, final_path
                
                # Update best
                if current_score > best_score:
                    best_score = current_score
                    best_field = current_field
                    nodes_since_improvement = 0
                    print(f"  📈 Improved: {best_score}/{target_score} at depth {g}")
                else:
                    nodes_since_improvement += 1
                
                # Check stuck
                if nodes_since_improvement >= stuck_threshold:
                    astar_stuck = True
                    print(f"  ⚠️  Stuck detected! ({nodes_since_improvement} nodes)")
                    break
                
                # Depth limit
                if g >= max_depth:
                    continue
                
                # Expand neighbors
                for x in range(n):
                    for y in range(n):
                        for size in range(2, min(5, n - max(x, y) + 1)):
                            new_field = current_field.rotate(x, y, size)
                            new_tuple = self._convert_field_to_tuple(new_field)
                            new_hash = self._compute_zobrist_hash(new_tuple, zobrist_table)
                            new_g = g + 1
                            nodes_generated += 1
                            
                            # Check if better path
                            if new_g < visited.get(new_hash, float('inf')):
                                visited[new_hash] = new_g
                                h = self.heuristic_optimized(new_field)
                                new_f = new_g + weight * h
                                
                                heapq.heappush(open_set, 
                                             (new_f, next(counter), new_g, new_field, new_tuple, new_hash))
                                parent[new_hash] = (current_hash, (x, y, size))
            
            total_nodes_explored += nodes_explored
            total_nodes_generated += nodes_generated
            
            # If not stuck or timeout, return best
            if not astar_stuck:
                path = self._reconstruct_path_from_parent(self._compute_zobrist_hash(
                    self._convert_field_to_tuple(best_field), zobrist_table), parent)
                final_path = global_path + path
                
                print(f"\n{'='*70}")
                print(f"⏱️  Search exhausted or timeout")
                print(f"{'='*70}")
                print(f"Best score: {best_score}/{target_score} ({best_score/target_score*100:.1f}%)")
                print(f"Total restarts: {restart_count}")
                print(f"Solution steps: {len(final_path)}")
                print(f"Nodes explored: {total_nodes_explored:,}")
                print(f"Time: {time.time() - start_time:.2f}s")
                print(f"{'='*70}\n")
                
                return best_field, final_path
            
            # === SA ESCAPE PHASE ===
            print(f"\n🌡️  SA Escape Phase")
            
            # Update global path
            path_to_best = self._reconstruct_path_from_parent(
                self._compute_zobrist_hash(self._convert_field_to_tuple(best_field), zobrist_table), 
                parent)
            global_path.extend(path_to_best)
            
            # Simulated Annealing with forced diversification
            T = 2.0
            T_min = 0.001
            alpha = 0.85
            
            current_sa_field = best_field
            current_sa_score = best_score
            sa_best_field = best_field
            sa_best_score = best_score
            sa_steps = []
            
            forced_exploration = sa_iterations // 3
            iteration = 0
            
            while T > T_min and iteration < sa_iterations:
                # Random move with size preference
                if iteration < forced_exploration:
                    size = random.randint(3, min(5, n))
                else:
                    size = random.randint(2, min(5, n))
                
                x = random.randint(0, n - size)
                y = random.randint(0, n - size)
                
                new_sa_field = current_sa_field.rotate(x, y, size)
                new_sa_score = new_sa_field.score()
                
                delta = new_sa_score - current_sa_score
                
                # Accept move
                if iteration < forced_exploration:
                    current_sa_field = new_sa_field
                    current_sa_score = new_sa_score
                    sa_steps.append((x, y, size))
                elif delta > 0 or random.random() < math.exp(delta / T):
                    current_sa_field = new_sa_field
                    current_sa_score = new_sa_score
                    sa_steps.append((x, y, size))
                
                # Track best
                if new_sa_score > sa_best_score:
                    sa_best_field = new_sa_field
                    sa_best_score = new_sa_score
                
                T *= alpha
                iteration += 1
            
            print(f"  🔄 SA: {iteration} iterations")
            print(f"  📊 Score: {best_score} → {sa_best_score}")
            print(f"  🚀 Escaped with {len(sa_steps)} steps")
            
            # Update for next A* iteration
            current_start_field = sa_best_field
            global_path.extend(sa_steps)
            restart_count += 1
            
            # Check if SA found solution
            if sa_best_score == target_score:
                print(f"\n{'='*70}")
                print(f"🎉 SUCCESS - SA found solution!")
                print(f"{'='*70}")
                print(f"Total restarts: {restart_count}")
                print(f"Solution steps: {len(global_path)}")
                print(f"Time: {time.time() - start_time:.2f}s")
                print(f"{'='*70}\n")
                return sa_best_field, global_path
        
        # Timeout
        print(f"\n{'='*70}")
        print(f"⏱️  Time limit reached")
        print(f"{'='*70}")
        print(f"Best score: {current_start_field.score()}/{target_score}")
        print(f"Total restarts: {restart_count}")
        print(f"Solution steps: {len(global_path)}")
        print(f"Nodes explored: {total_nodes_explored:,}")
        print(f"Time: {time.time() - start_time:.2f}s")
        print(f"{'='*70}\n")
        
        return current_start_field, global_path
    
    def _reconstruct_path_from_parent(self, state_hash: int, parent: Dict) -> List:
        """Reconstruct path từ parent dictionary"""
        path = []
        current = state_hash
        
        while current in parent and parent[current] is not None:
            parent_hash, action = parent[current]
            path.append(action)
            current = parent_hash
        
        path.reverse()
        return path

if __name__ == "__main__":
    initial_time = time.time()
    
    sizeOfEntities = 8  # Bắt đầu với size nhỏ để test
    entities = generateField(n = sizeOfEntities)
    mappings = build_mapping(n = sizeOfEntities)
    
    field = Field(size=sizeOfEntities, entities=entities, mappings=mappings)    
    print(f"Trạng thái ban đầu:")
    print(field)
    print(f"\nScore ban đầu: {field.score()}/{(sizeOfEntities * sizeOfEntities) // 2}")
    
    solverAlgorithm = Solver(init_field=field, max_depth=100)
    
    # Test Optimized A* with Hybrid SA
    print("\n" + "="*70)
    print("TESTING OPTIMIZED A* WITH HYBRID SA")
    print("="*70)
    
    best_field, solution = solverAlgorithm.OptimizedAStar(
        weight=1.5,           # Weighted A* để tăng tốc
        time_limit=120,       # 2 phút
        stuck_threshold=3000, # Trigger SA sau 3000 nodes không cải thiện
        sa_iterations=50,     # 50 iterations SA mỗi lần escape
        max_depth=100         # Giới hạn độ sâu
    )
    
    print(f"\nKết quả cuối cùng:")
    print(best_field)
    print(f"\nFinal Score: {best_field.score()}/{(sizeOfEntities * sizeOfEntities) // 2}")
    print(f"Number of actions: {len(solution)}")
    print(f"Total time: {time.time() - initial_time:.2f}s")

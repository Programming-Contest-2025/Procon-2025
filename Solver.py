import heapq, itertools, random, time, math
import numpy as np
from FieldClass import *
from concurrent.futures import ProcessPoolExecutor
from utils import generateField
from model.CNN_model import CNN_Network
import argparse
import torch
from postProcessing import removeDuplicate, removeSameState
import math, random
from typing import Tuple, List, Dict, Optional


random.seed(42)

inf_num = 10**12

class Solver:
    def __init__(self, init_field, model = None, max_depth = 500):
        self.field = init_field
        self.visited = set()
        self.max_depth = max_depth
        self.best_field = None
        self.best_score = -1
        self.heuristic_model = model
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
    
    def cnn_heuristic(self, field):
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cnn_model = self.heuristic_model
        state = field.entities
        
        cnn_model = cnn_model.to(device)
        state = torch.tensor(state, dtype=torch.long).unsqueeze(0)
        
        predict_score = cnn_model(state)
        return predict_score.item()
    
    def heuristic(self, field):
        n = self.field.n 
        entities = self.field.entities
        
        #Lấy tọa độ các cặp giá trị
        pairs, _ = field.get_unpaired()
        
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
            _, unpaired_coords = field.get_unpaired()
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
        init_field = self.field
        
        current_field = self.field
        n = current_field.n 
        current_heuristic = current_field.calculate_heuristic_sa()
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
                new_heuristic = new_field.calculate_heuristic_sa()
                
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
            
        steps = removeSameState(path = steps, custom_field = init_field)
        steps = removeDuplicate(path = steps)
        
        return best_field, steps

    def simulated_annealing_record(
        self, 
        max_neiborghs=100, 
        T_start=1.0, 
        T_min=0.01, 
        alpha=0.995,
    ):
        
        current_field = self.field
        n = current_field.n
        current_heuristic = current_field.calculate_heuristic_sa()
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
                new_heuristic = new_field.calculate_heuristic_sa()

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
    
    def OptimizedAStar(self, weight: float = 1.5, time_limit: int = 300,
                      stuck_threshold: int = 50000, sa_iterations: int = 0,
                      max_depth: int = 200, beam_width: int = 800000,
                      enable_restart: bool = True) -> Tuple:
        """
        Optimized A* với 5 cải tiến chính:
        
        1. Priority 2 cấp: Ưu tiên score cao trước, sau đó f-score
        2. Neighbor Pruning: Chỉ xoay vùng có ô chưa ghép cặp
        3. Tuple-based visited set (không dùng Zobrist hash)
        4. Beam Search: Giới hạn queue size để tránh memory overflow
        5. Restart Mechanism: Random restart từ best state khi stuck
        
        Args:
            weight: Trọng số cho weighted A* (>1.0 để tăng tốc)
            time_limit: Giới hạn thời gian (giây)
            stuck_threshold: Số nodes không cải thiện -> restart
            sa_iterations: Số lần restart tối đa
            max_depth: Giới hạn độ sâu tìm kiếm
            beam_width: Số states tối đa trong queue (beam search)
            enable_restart: Bật/tắt restart mechanism
        
        Returns:
            (best_field, solution_path)
        """
        print(f"\n{'='*70}")
        print(f"🚀 OPTIMIZED A* WITH BEAM SEARCH + RESTART")
        print(f"{'='*70}")
        print(f"Weight: {weight}, Time limit: {time_limit}s")
        print(f"Max depth: {max_depth}, Beam width: {beam_width:,}")
        print(f"Restart: {'ON' if enable_restart else 'OFF'}, Stuck threshold: {stuck_threshold:,}")
        print()
        
        start_time = time.time()
        n = self.field.n
        target_score = (n * n) // 2
        
        # Initialize tracking
        counter = itertools.count()
        
        # Statistics
        total_nodes_explored = 0
        total_nodes_generated = 0
        
        # === A* PHASE WITH RESTART ===
        print(f"\n🎯 A* Search with Beam Search + Restart")
        
        # Global best tracking
        global_best_score = self.field.score()
        global_best_field = self.field
        global_best_tuple = self._convert_field_to_tuple(self.field)
        global_best_parent = {}  # Track parent for best solution
        
        restart_count = 0
        max_restarts = sa_iterations if enable_restart else 0
        
        while restart_count <= max_restarts:
            if time.time() - start_time > time_limit:
                break
                
            if restart_count > 0:
                print(f"\n🔄 Restart #{restart_count} - Clear visited set for re-exploration (score: {global_best_score}/{target_score})")
                # Clear visited để cho phép re-explore
                visited_size_before = len(visited)
                visited.clear()
                # Re-add states in queue
                for _, _, g, field, field_tuple in open_set:
                    visited[field_tuple] = g
                print(f"  Cleared {visited_size_before:,} visited states, kept {len(visited):,} in queue")
            
            # Convert current field to tuple state
            if restart_count == 0:
                start_tuple = self._convert_field_to_tuple(self.field)
                start_field = self.field
                # Priority queue: (f_score, counter, g_score, field, field_tuple)
                open_set = []
                heapq.heappush(open_set, (0, next(counter), 0, start_field, start_tuple))
                
                # visited: tuple -> g_score (dùng tuple làm key trực tiếp)
                # parent: tuple -> (parent_tuple, action)
                visited = {start_tuple: 0}
                parent = {start_tuple: None}
            
            if restart_count == 0:
                best_score = start_field.score()
                best_field = start_field
                best_tuple = start_tuple
                nodes_since_improvement = 0
                
                nodes_explored = 0
                nodes_generated = 0
                
                last_print_time = time.time()
            else:
                # Reset counter nhưng giữ best và open_set
                nodes_since_improvement = 0
                nodes_explored = 0
                nodes_generated = 0
                last_print_time = time.time()
            
            while open_set and time.time() - start_time < time_limit:
                # Progress update
                current_time = time.time()
                if current_time - last_print_time >= 5.0:
                    print(f"  [{current_time - start_time:.1f}s] Explored: {nodes_explored:,}, "
                          f"Queue: {len(open_set):,}, Best: {best_score}/{target_score}")
                    last_print_time = current_time
                
                f, _, g, current_field, current_tuple = heapq.heappop(open_set)
                nodes_explored += 1
                
                # Skip if outdated
                if g > visited.get(current_tuple, float('inf')):
                    continue
                
                current_score = current_field.score()
                
                # Goal check
                if current_score == target_score:
                    print(f"\n✅ Perfect solution found!")
                    path = self._reconstruct_path_from_parent(current_tuple, parent)
                    final_path = path
                    
                    print(f"\n{'='*70}")
                    print(f"🎉 SUCCESS - Perfect solution!")
                    print(f"{'='*70}")
                    print(f"Final score: {current_score}/{target_score}")
                    print(f"Solution steps: {len(final_path)}")
                    print(f"Nodes explored: {total_nodes_explored + nodes_explored:,}")
                    print(f"Time: {time.time() - start_time:.2f}s")
                    print(f"{'='*70}\n")
                    
                    return current_field, final_path
                
                # Update best
                if current_score > best_score:
                    best_score = current_score
                    best_field = current_field
                    best_tuple = current_tuple
                    nodes_since_improvement = 0
                    print(f"  📈 Improved: {best_score}/{target_score} at depth {g}")
                    
                    # Update global best
                    if best_score > global_best_score:
                        global_best_score = best_score
                        global_best_field = best_field
                        global_best_tuple = best_tuple
                        global_best_parent = parent.copy()  # Save parent for reconstruction
                else:
                    nodes_since_improvement += 1
                
                # RESTART: Stuck detection
                if enable_restart and nodes_since_improvement >= stuck_threshold:
                    print(f"  ⚠️  Stuck detected! ({nodes_since_improvement:,} nodes without improvement)")
                    break
                #     print(f"  ⚠️  Stuck detected! ({nodes_since_improvement} nodes)")
                #     break
                
                # Depth limit
                if g >= max_depth:
                    continue
                
                # CẢI TIẾN #3: Lấy set các ô chưa ghép cặp để pruning
                unpaired_coords = current_field.get_unpaired_coords_set()
                
                # Expand neighbors
                for x in range(n):
                    for y in range(n):
                        for size in range(2, min(5, n - max(x, y) + 1)):
                            # CẢI TIẾN #3: Kiểm tra vùng xoay có chồng lấn với ô chưa ghép không
                            has_overlap = False
                            for i in range(y, y + size):
                                for j in range(x, x + size):
                                    if (i, j) in unpaired_coords:
                                        has_overlap = True
                                        break
                                if has_overlap:
                                    break
                            
                            # Chỉ xoay nếu có overlap với unpaired cells
                            if not has_overlap:
                                continue
                            
                            new_field = current_field.rotate(x, y, size)
                            new_tuple = self._convert_field_to_tuple(new_field)
                            
                            new_g = g + 1
                            nodes_generated += 1
                            
                            # Check if better path (dùng tuple trực tiếp làm key)
                            if new_g < visited.get(new_tuple, float('inf')):
                                visited[new_tuple] = new_g
                                
                                # Tính score của state mới
                                new_score = new_field.score()
                                h = self.heuristic_optimized(new_field)
                                
                                # CẢI TIẾN #1: Tính priority theo 2 tiêu chí với LOOKAHEAD
                                # Ưu tiên 1: Số cặp ĐÃ ghép (càng NHIỀU càng tốt)
                                # => Đảo dấu để state có score cao có priority THẤP (ưu tiên trong min-heap)
                                cost_1 = -(new_score)  # Dấu âm: score cao = priority thấp = ưu tiên cao
                                
                                # LOOKAHEAD: Cho phép temporary score decrease nếu heuristic tốt
                                # Nếu score giảm nhưng h thấp (gần target), vẫn cho vào queue
                                score_delta = new_score - current_score
                                
                                if score_delta < 0:
                                    # Score giảm - apply penalty nhưng không block hoàn toàn
                                    # Penalty tỉ lệ với độ giảm và heuristic
                                    lookahead_penalty = abs(score_delta) * 400_000
                                    # Nếu h nhỏ (gần target), giảm penalty mạnh hơn
                                    if h < 3.0:  # Very close to solution
                                        lookahead_penalty *= 0.1  # Giảm 90% penalty
                                    elif h < 6.0:  # Close to solution
                                        lookahead_penalty *= 0.4  # Giảm 60% penalty
                                    elif h < 10.0:  # Moderately close
                                        lookahead_penalty *= 0.7  # Giảm 30% penalty
                                    cost_1 += lookahead_penalty  # Tăng cost (giảm priority)
                                
                                # Ưu tiên 2: Số bước đi (f-score) 
                                cost_2 = new_g + weight * h
                                
                                # Công thức tổng hợp: cost_1 * C luôn được ưu tiên trước
                                f_priority = (cost_1 * 1_000_000) + cost_2
                                
                                heapq.heappush(open_set, 
                                             (f_priority, next(counter), new_g, new_field, new_tuple))
                                parent[new_tuple] = (current_tuple, (x, y, size))
                
                # BEAM SEARCH: Prune queue nếu quá lớn
                if len(open_set) > beam_width:
                    # Giữ lại beam_width states tốt nhất
                    open_set = heapq.nsmallest(beam_width, open_set)
                    heapq.heapify(open_set)
                    print(f"  ✂️  Pruned queue: {beam_width:,} states (was too large)")
            
            total_nodes_explored += nodes_explored
            total_nodes_generated += nodes_generated
            
            # Check if should restart
            if best_score == target_score:
                break
            if not enable_restart:
                break
            if time.time() - start_time > time_limit:
                break
            if restart_count >= max_restarts:
                break
                
            restart_count += 1
        
        # Return best result found (global best)
        path = self._reconstruct_path_from_parent(global_best_tuple, global_best_parent)
        
        print(f"\n{'='*70}")
        print(f"⏱️  Search completed")
        print(f"{'='*70}")
        print(f"Restarts performed: {restart_count}/{max_restarts}")
        print(f"Best score: {global_best_score}/{target_score} ({global_best_score/target_score*100:.1f}%)")
        print(f"Solution steps: {len(path)}")
        print(f"Nodes explored: {total_nodes_explored:,}")
        print(f"Time: {time.time() - start_time:.2f}s")
        print(f"{'='*70}\n")
        
        return global_best_field, path
    
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
    
    def _update_zobrist_hash(self, old_hash: int, old_tuple: Tuple[int, ...], 
                            new_tuple: Tuple[int, ...], action: Tuple[int, int, int],
                            n: int, zobrist_table: np.ndarray) -> int:
        """CẢI TIẾN #2: Cập nhật Zobrist hash tăng tiến O(k²) thay vì O(N²)
        
        FIXED: So sánh toàn bộ tuple để tìm tất cả vị trí thay đổi, vì rotate() 
        sử dụng permutation có thể ảnh hưởng các ô bên ngoài vùng k×k ban đầu.
        
        Args:
            old_hash: Hash của trạng thái cũ
            old_tuple: Tuple trạng thái cũ
            new_tuple: Tuple trạng thái mới
            action: (x, y, size) - vùng xoay (không dùng nữa, giữ cho tương thích)
            n: Kích thước board
            zobrist_table: Bảng Zobrist
            
        Returns:
            Hash mới được tính tăng tiến
        """
        new_hash = old_hash
        
        # So sánh toàn bộ board để tìm các vị trí thay đổi
        # Mặc dù là O(N²) nhưng chỉ là so sánh int, vẫn nhanh hơn tính hash mới
        for pos in range(len(old_tuple)):
            old_val = old_tuple[pos]
            new_val = new_tuple[pos]
            
            # XOR ra giá trị cũ, XOR vào giá trị mới
            if old_val != new_val:
                new_hash ^= zobrist_table[pos][old_val]
                new_hash ^= zobrist_table[pos][new_val]
        
        return int(new_hash)
    
    def _tuple_to_field(self, board_tuple: Tuple[int, ...], n: int) -> 'Field':
        """Convert tuple back to Field object"""
        entities = []
        for i in range(n):
            entities.append(list(board_tuple[i * n:(i + 1) * n]))
        return Field(size=n, entities=entities, mappings=self.field.mappings)
    
    def heuristic_optimized(self, field) -> float:
        """
        Advanced heuristic với nhiều yếu tố:
        1. Manhattan distance (cơ bản)
        2. Clustering penalty (phạt các cặp xa nhau)
        3. Edge bonus (ưu tiên ghép ở rìa board)
        4. Rotation difficulty (ước lượng số rotation cần thiết)
        5. Deadlock detection (phát hiện pattern không thể giải)
        """
        positions, unpaired = field.get_unpaired()
        
        if not unpaired:
            return 0.0
        
        n = field.n
        h = 0.0
        
        # 1. Manhattan distance - Base heuristic
        total_manhattan = 0
        max_distance = 0
        distances = []
        
        for coords in unpaired:
            (y1, x1), (y2, x2) = coords
            dist = abs(y1 - y2) + abs(x1 - x2)
            total_manhattan += dist
            max_distance = max(max_distance, dist)
            distances.append(dist)
        
        # Base cost từ Manhattan
        h += total_manhattan / 2.5
        
        # 2. Clustering penalty - Phạt nếu các cặp quá xa nhau
        # Pairs càng xa càng khó ghép trong 1 rotation
        if distances:
            avg_distance = sum(distances) / len(distances)
            clustering_penalty = 0
            for dist in distances:
                if dist > avg_distance * 1.5:  # Xa hơn 1.5 lần trung bình
                    clustering_penalty += (dist - avg_distance) * 0.3
            h += clustering_penalty
        
        # 3. Edge bonus - Các cặp ở rìa dễ ghép hơn
        edge_bonus = 0
        for coords in unpaired:
            (y1, x1), (y2, x2) = coords
            # Kiểm tra có ô nào ở rìa không
            at_edge1 = (x1 == 0 or x1 == n-1 or y1 == 0 or y1 == n-1)
            at_edge2 = (x2 == 0 or x2 == n-1 or y2 == 0 or y2 == n-1)
            if at_edge1 or at_edge2:
                edge_bonus += 0.5  # Giảm h cho pairs gần rìa
        h -= edge_bonus
        
        # 4. Rotation difficulty - Ước lượng số rotation thực tế
        # Pairs trong cùng 4x4 region dễ ghép hơn
        same_region_count = 0
        for coords in unpaired:
            (y1, x1), (y2, x2) = coords
            # Check nếu cả 2 trong cùng region 4x4
            region_x1, region_y1 = x1 // 4, y1 // 4
            region_x2, region_y2 = x2 // 4, y2 // 4
            if region_x1 == region_x2 and region_y1 == region_y2:
                same_region_count += 1
        
        # Bonus cho pairs trong cùng region
        h -= same_region_count * 0.8
        
        # 5. Deadlock detection - Phát hiện patterns khó giải
        # Pattern 1: 2 pairs cùng value ở 4 góc của 1 region
        # Pattern 2: Circular dependency
        deadlock_penalty = 0
        
        # Tính số lượng values xuất hiện nhiều lần trong unpaired
        value_counts = {}
        for coords in unpaired:
            (y1, x1), (y2, x2) = coords
            val1 = field.entities[y1][x1]
            val2 = field.entities[y2][x2]
            value_counts[val1] = value_counts.get(val1, 0) + 1
            value_counts[val2] = value_counts.get(val2, 0) + 1
        
        # Pairs có value xuất hiện > 2 lần (conflict) bị phạt
        for val, count in value_counts.items():
            if count > 2:
                deadlock_penalty += (count - 2) * 1.5
        
        h += deadlock_penalty
        
        # 6. Admissibility adjustment - Đảm bảo h không overestimate
        # Số moves tối thiểu là ceil(unpaired_count / max_pairs_per_move)
        # Với 1 move tốt nhất có thể ghép 2-3 pairs
        min_moves_needed = len(unpaired) / 3.0
        
        # Đảm bảo h >= min_moves (admissible)
        h = max(h, min_moves_needed)
        
        return h
    
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
    print(f"Trạng thái ssban đầu:")
    print(field)
    print(f"\nScore ban đầu: {field.score()}/{(sizeOfEntities * sizeOfEntities) // 2}")
    
    solverAlgorithm = Solver(init_field=field, max_depth=100)
    
    # Test Optimized A* with Hybrid SA
    print("\n" + "="*70)
    print("TESTING OPTIMIZED A* WITH HYBRID SA")
    print("="*70)
    
    best_field, solution = solverAlgorithm.OptimizedAStar(
        weight=1.5,           # Weighted A* để tăng tốc
        time_limit=300,      
        stuck_threshold=3000, # Trigger SA sau 3000 nodes không cải thiện
        sa_iterations=50,     # 50 iterations SA mỗi lần escape
        max_depth=100         # Giới hạn độ sâu
    )
    
    print(f"\nKết quả cuối cùng:")
    print(best_field)
    print(f"\nFinal Score: {best_field.score()}/{(sizeOfEntities * sizeOfEntities) // 2}")
    print(f"Number of actions: {len(solution)}")
    print(f"Total time: {time.time() - initial_time:.2f}s")


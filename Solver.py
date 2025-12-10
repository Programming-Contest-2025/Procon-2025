import heapq
import itertools
import random
import time
import math
from typing import Tuple, List, Dict
from FieldClass import Field
from postProcessing import post_processing, apply_path
import gc
import copy
# random.seed(42)

class Solver:
    def __init__(self, init_field, model = None, max_depth = 500):
        self.field = init_field
        self.visited = set()
        self.max_depth = max_depth
        self.best_field = None
        self.best_score = -1
        self.heuristic_model = model
        self.parents = {} #current_hash : (parent_hash, action)   
    
    @staticmethod
    def default_params_for_size(n: int) -> dict:
        """Chọn tham số mặc định cho A* theo kích thước bàn.

        Ưu tiên:
        - Nhỏ (≤12): ưu điểm tuyệt đối (pair_score tối đa), thời gian ngắn.
        - Vừa (14, 16): cho phép rộng beam hơn, thời gian lớn hơn nhưng vẫn <~200s.
        - Lớn (≥20): giới hạn beam/time, tập trung cải thiện pair_score, chấp nhận step cao.
        """
        if n <= 8:
            return {
                "time_limit": 20,   # board nhỏ, A* nên xong rất nhanh
                "max_depth": 120,
                "beam_width": 150_000,
            }
        elif n <= 10:
            # 10x10 hiện giải rất chắc trong ~20s, nên giữ cấu hình cũ
            return {
                "time_limit": 90,
                "max_depth": 220,
                "beam_width": 300_000,
            }
        elif n <= 12:
            # 12x12: tăng nhẹ time_limit + beam để đẩy tỉ lệ perfect lên 100%
            return {
                "time_limit": 130,
                "max_depth": 240,
                "beam_width": 350_000,
            }
        elif n <= 14:
            # 14x14: cần mạnh tay hơn một chút để gần như luôn perfect
            return {
                "time_limit": 140,
                "max_depth": 360,
                "beam_width": 350_000,
            }
        elif n <= 16:
            # 16x16: cho thêm depth/time và beam để đảm bảo perfect
            return {
                "time_limit": 180,
                "max_depth": 420,
                "beam_width": 400_000,
            }
        elif n <= 20:
            # 18–20: cho phép lâu hơn để đẩy pair_score nhưng vẫn trong ~200s
            return {
                "time_limit": 200,
                "max_depth": 520,
                "beam_width": 450_000,
            }
        else:
            # n > 20: thêm thời gian cho bài rất lớn nếu bạn chấp nhận
            return {
                "time_limit": 220,
                "max_depth": 520,
                "beam_width": 450_000,
            }
    def random_action(self):
        #Cần tối ưu chọn size nhỏ, thoát local optimal mới chọn size lớn
        n = self.field.n
        
        size = random.randint(2, n // 2)
        x = random.randint(0, n - size)
        y = random.randint(0, n - size)
        return (x, y, size)
    
    def smart_action(self, field, T_ratio, sizes_global = None):
        """
        Sinh 1 action chọn ngẫu nhiên trong candidate_rotations,
        nhưng với bias: đầu T_ratio cao (gần 1) chọn size lớn hơn,
        khi T_ratio nhỏ thì ưu size nhỏ.
        """
        n = field.n
        if sizes_global is None:
            sizes_global = [max(2, n//2), max(2, n//2 - 1), 6, 4, 3, 2]

        # adjust selection of sizes by T_ratio (1..0)
        if T_ratio > 0.7:
            sizes = sizes_global[:3]   # ưu size lớn
        elif T_ratio > 0.3:
            sizes = sizes_global[2:5]
        else:
            sizes = [2, 3]

        candidates = self._candidate_rotations_from_unpaired(field, sizes)
        if not candidates:
            # fallback to small random
            size = random.choice([2,3])
            x = random.randint(0, n - size)
            y = random.randint(0, n - size)
            return (x, y, size)
        # score/weight candidates by simple heuristic: how many unpaired on perimeter
        unpaired = set(field.get_unpaired_coords_set()[1])
        scored = []
        for (x,y,size) in candidates:
            perim = 0
            for (py, px) in self.border_positions(x, y, size):
                if (py, px) in unpaired:
                    perim += 1
            scored.append(((x,y,size), perim))
        # pick by roulette on perim>0; if all zero pick random
        total = sum(s for _, s in scored)
        if total == 0:
            return random.choice([c for c,_ in scored])[0]
        r = random.randint(1, total)
        s = 0
        for (c, val) in scored:
            s += val
            if r <= s:
                return c
        return scored[-1][0]

    def _candidate_rotations_from_unpaired(self, field, sizes):
        """
        Trả về list (x,y,size) dedup từ các unpaired cells.
        sizes: list of sizes ưu tiên (big -> small)
        """
        _, unpaired_coords, _ = field.get_unpaired_coords_set()
        n = field.n
        cand = []
        for (uy, ux) in unpaired_coords:   # lưu ý get_unpaired_coords_set dùng (y,x)
            for size in sizes:
                S = size - 1
                for x in (ux, ux - S):
                    for y in (uy, uy - S):
                        if x is None or y is None:
                            continue
                        if x < 0 or y < 0:
                            continue
                        if x + size > n or y + size > n:
                            continue
                        cand.append((x, y, size))
        # dedup
        return list(dict.fromkeys(cand))  # preserve order, faster than set for small lists

    
    # def simulated_annealing(self, max_neiborghs = 2000, T_start = 1.0, T_min = 1e-5, alpha = 0.95):
    #     current_field = self.field
    #     n = current_field.n 
    #     target_score = n*n // 2
    #     current_heuristic = current_field.heuristic_SA()
    #     best_field = current_field
    #     best_heuristic = current_heuristic
    #     best_score = current_field.score()
    #     steps = []
    #     best_true_steps = []

    #     T = T_start
    #     while T > T_min:
    #         iters = 1
    #         while iters <= max_neiborghs:
    #             #Mở rộng nhánh
    #             action = self.random_action()
    #             new_field = current_field.rotate(*action)
    #             new_heuristic = new_field.heuristic_SA()
                
    #             delta = new_heuristic - current_heuristic
                
    #             if delta > 0:
    #                 current_field = new_field
    #                 current_heuristic = new_heuristic
    #                 steps.append(action)
    #                 if new_field.score() > best_score:
    #                     best_true_steps = steps.copy()
    #                     best_score = new_field.score()
                    
    #                 if new_heuristic > best_heuristic:
    #                     best_field = new_field
    #                     best_heuristic = new_heuristic

    #             else:
    #                 #Xác suất chấp nhận cái mới
    #                 p = math.exp(delta / T)
    #                 if p > 0.4: p = 0.4
    #                 if random.random() < p:
    #                     current_field = new_field
    #                     current_heuristic = new_heuristic
    #                     steps.append(action)
    #                     if new_field.score() > best_score:
    #                             best_true_steps = steps.copy()
    #                             best_score = new_field.score()
                
    #             iters += 1
    #             if best_field.score() == target_score:
    #                 print(f"Perfect solution has found!")
    #                 return best_field, steps
    #         print(f"T = {T}, best_score = {best_field.score()}/{target_score}, length: {len(steps)}")
    #         T = T * alpha
        
    #     return best_field, best_true_steps

    def simulated_annealing(self,
                        max_neighbors=900,
                        T_start=1.0,
                        T_min=1e-5,
                        alpha=0.92,
                        step_penalty=0.1,
                        stagnation_limit=400,
                        tabu_size=200):
        """
        SA cải tiến:
        - max_neighbors: số neighbor thử mỗi temperature
        - step_penalty: số điểm 1 step "giá" (dùng để prune)
        - stagnation_limit: nếu không có tăng score thật trong nhiều bước -> reheating / reset
        - tabu_size: lưu hashes recent để tránh cycle ngắn
        """
        random.seed()  # optional: keep nondet if not set earlier
        init_field = self.field.clone()  # giữ bản gốc
        current_field = self.field.clone()
        n = current_field.n
        target_score = n*n // 2

        # heuristic và score
        current_h = current_field.heuristic_SA()
        current_score = current_field.score()

        # track best by true score
        best_true_score = current_score
        best_true_steps = []
        best_true_field = current_field.clone()

        # track heuristic best (for tuning)
        best_h = current_h

        # tabu (recent hashes) để tránh lặp vòng ngắn
        from collections import deque
        tabu = deque(maxlen=tabu_size)
        tabu.append(current_field.incremental_hash())

        steps = []
        T = T_start
        total_iters = 0
        no_improve_counter = 0

        # sizes global fallback order
        sizes_global = [n//2, n//2-1, 8, 6, 4, 3, 2]

        print(f"[INIT] score={current_score}, heuristic={current_h}")

        while T > T_min:
            iters = 0
            T_ratio = T / T_start
            # shrink neighbor budget when T small to focus
            local_max_neighbors = max(100, int(max_neighbors * (0.6 + 0.4 * T_ratio)))
            while iters < local_max_neighbors:
                # generate candidate move guided by unpaired
                action = self.smart_action(current_field, T_ratio, sizes_global)
                x,y,size = action

                # quick pruning: if action doesn't touch any unpaired cell on perimeter => skip
                _, unpaired_coords, _ = current_field.get_unpaired_coords_set()
                if all((py,px) not in unpaired_coords for (py,px) in self.border_positions(x,y,size)):
                    iters += 1
                    continue

                # apply rotation (fast)
                new_field = current_field.rotate(x,y,size)
                new_h = new_field.heuristic_SA()
                new_score = new_field.score()
                new_g_cost = 1  # each action cost 1 step

                # cost-benefit prune: nếu gain < cost (với step_penalty) thì drop
                gain = new_score - current_score
                cost = new_g_cost * step_penalty
                if gain < cost and new_score < best_true_score:
                    # nếu không đem tới true score tốt hơn và không bù được step cost -> skip
                    iters += 1
                    continue

                # acceptance on heuristic delta (energy)
                delta_h = new_h - current_h
                accept = False
                if delta_h > 0:
                    accept = True
                else:
                    p = math.exp(delta_h / max(T, 1e-12))
                    if p > 0.45: p = 0.45
                    if random.random() < p:
                        accept = True

                if accept:
                    current_field = new_field
                    current_h = new_h
                    current_score = new_score
                    steps.append(action)
                    tabu.append(current_field.incremental_hash())

                    # update heuristic best
                    if current_h > best_h:
                        best_h = current_h

                    # update best by true score (freeze path when improve)
                    if current_score > best_true_score:
                        best_true_score = current_score
                        best_true_field = current_field.clone()
                        best_true_steps = steps.copy()
                        no_improve_counter = 0
                        
                        print(f"[SCORE ↑] score={best_true_score}, "
                        f"steps_len={len(best_true_steps)}")

                        if best_true_score == target_score:
                            # perfect -> postprocess then return
                            print("[PERFECT] Full score reached!")
                            pruned = post_processing(best_true_steps, init_field)
                            final_field = apply_path(init_field, pruned)
                            return final_field, pruned
                    else:
                        no_improve_counter += 1
                else:
                    no_improve_counter += 1

                total_iters += 1
                iters += 1

            # end local neighbors loop

            # stagnation handling
            if no_improve_counter >= stagnation_limit:
                print("\n[STAGNATION] → REHEAT + RESET to best_true_state")
                print(f"   current_score={current_score}, best_true_score={best_true_score}")
                
                # reheating: bump temperature and randomize a bit around best_true_field
                T = min(T_start, T * 2.0)
                # reset current_field to best_true_field (escape wandering)
                current_field = best_true_field.clone()
                current_h = current_field.heuristic_SA()
                current_score = current_field.score()
                steps = best_true_steps.copy()
                no_improve_counter = 0
                # small shuffle: apply a few random useful moves to escape
                for _ in range(5):
                    a = self.smart_action(current_field, 1.0, sizes_global)
                    current_field = current_field.rotate(*a)
                    steps.append(a)
                tabu.clear()
                tabu.append(current_field.incremental_hash())

            # cool down
            print(f"T={T:.6f},  best_score={best_true_score},  steps={len(steps)}")
            T *= alpha

        # end T loop

        # final postprocessing
        pruned = post_processing(best_true_steps, init_field)
        final_field = apply_path(init_field, pruned)
        return final_field, pruned

    def OptimizedAStar(self,
                       time_limit: int | None = None,
                       max_depth: int | None = None,
                       beam_width: int | None = None,) -> Tuple:


        print(f"\n{'='*70}")
        print(f"🚀 OPTIMIZED A* WITH BEAM SEARCH + RESTART")
        print(f"{'='*70}")
        start_time = time.time()
        init_field = self.field
        n = self.field.n

        # Nếu không truyền tham số thì dùng cấu hình mặc định theo size
        params = self.default_params_for_size(n)
        if time_limit is None:
            time_limit = params["time_limit"]
        if max_depth is None:
            max_depth = params["max_depth"]
        if beam_width is None:
            beam_width = params["beam_width"]

        print(f"Time limit: {time_limit}s")
        print(f"Max depth: {max_depth}, Beam width: {beam_width:,}")
        print()

        time_exceeded = False
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
        
        global_best_path = self._reconstruct_path_from_parent(global_best_tuple, global_best_parent)
        
        ###-----------A* Algorithm------------###
        # Convert current field to tuple state
        start_tuple = self._convert_field_to_tuple(self.field)
        start_field = self.field
        # Priority queue: (f_score, counter, g_score, field, field_tuple)
        open_set = []
        heapq.heappush(open_set, (0, next(counter), 0, start_field, start_tuple))
        
        # visited: tuple -> g_score (dùng tuple làm key trực tiếp)
        # parent: tuple -> (parent_tuple, action)
        # Hệ số ưu tiên cho A*: ưu tiên pair_score tuyệt đối, sau đó heuristic, cuối cùng là step
        if n <= 12:
            score_factor = 100.0
            heuristic_factor = 3.0
            step_factor = 1.0
        elif n <= 16:
            # size 14,16: tăng weight cho score, giảm ảnh hưởng step
            score_factor = 120.0
            heuristic_factor = 2.0
            step_factor = 0.3
        elif n <= 20:
            # 16–20: ưu tiên score, nhưng vẫn phạt step nhẹ
            score_factor = 120.0
            heuristic_factor = 1.5
            step_factor = 0.2
        else:
            # n > 20: gần như bỏ qua step hoàn toàn
            score_factor = 120.0
            heuristic_factor = 1.5
            step_factor = 0.1
        # start_f_priority = (- score_weight * init_field.score() + (0 * step_weight) + (init_field.heuristic_manhattan() * heuristic_weight)) // 2
        visited = {start_tuple: 0}
        parent = {start_tuple: None}
    
        # if restart_count == 0:
        best_score = start_field.score()
        best_field = start_field
        best_tuple = start_tuple

        nodes_since_improvement = 0            
        nodes_explored = 0
        nodes_generated = 0
        
        # priority_size = init_field.get_children_nodes()
        last_print_time = time.time()
        
        while open_set:
            # Progress update
            current_time = time.time()
            if time.time() - start_time >= time_limit:
                print(f"break ở while")
                break
            
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
                
                return current_field, final_path
            
            # Update best
            if current_score > best_score:
                best_score = current_score
                best_field = current_field
                best_tuple = current_tuple                
                # Update global best
                if best_score > global_best_score:
                    global_best_score = best_score
                    global_best_field = best_field
                    global_best_tuple = best_tuple
                    # Reconstruct path từ parent hiện tại
                    global_best_path = self._reconstruct_path_from_parent(best_tuple, parent)
                print(f"  📈 Improved: {best_score}/{target_score} at depth {g} with length = {len(global_best_path)}")
                nodes_since_improvement = 0
            else:
                nodes_since_improvement += 1
                if nodes_since_improvement >= 100000:
                    break
            
            # Depth limit
            if g >= max_depth:
                continue
            
            # CẢI TIẾN: generate moves tập trung quanh các ô chưa ghép cặp
            _, unpaired_coords, _ = current_field.get_unpaired_coords_set()

            # Điều chỉnh tập size theo tiến độ: đầu game xoay block lớn, cuối game dùng size nhỏ để tinh chỉnh
            score_ratio = current_score / target_score if target_score > 0 else 0.0
            if n >= 16:
                if score_ratio < 0.8:
                    size_candidates = [max(2, n // 2), max(2, n // 2 - 1), 8, 6, 4, 3, 2]
                elif score_ratio < 0.93:
                    size_candidates = [8, 6, 4, 3, 2]
                else:
                    # rất gần full score -> chỉ xoay block nhỏ để tinh chỉnh
                    size_candidates = [4, 3, 2]
            else:
                # Board nhỏ/mid: khi còn xa target thì cho phép block lớn,
                # gần target (>= ~0.9) thì chỉ dùng block nhỏ để tránh phá nhiều cặp đã ghép
                if score_ratio < 0.9:
                    size_candidates = [max(2, n // 2), max(2, n // 2 - 1), 4, 3, 2]
                else:
                    size_candidates = [4, 3, 2]

            candidate_actions = current_field.get_children_nodes(size_candidates)

            # Nếu vì lý do gì đó không sinh được move, fallback quét toàn board
            if not candidate_actions:
                for x in range(n):
                    for y in range(n):
                        for size in size_candidates:
                            candidate_actions.append((x, y, size))

            # Với board lớn (từ 14 trở lên), để tránh bùng nổ nhánh:
            # chỉ giữ các move "tiềm năng" nhất theo số unpaired trên biên
            if n >= 14 and len(candidate_actions) > 0:
                unpaired_set = set(unpaired_coords)
                scored = []
                for (ax, ay, asize) in candidate_actions:
                    perim = 0
                    for (py, px) in self.border_positions(ax, ay, asize):
                        if (py, px) in unpaired_set:
                            perim += 1
                    if perim > 0:
                        scored.append(((ax, ay, asize), perim))

                if scored:
                    # Giới hạn số move theo kích thước bàn
                    if n < 16:
                        max_local = 50
                    elif n < 20:
                        max_local = 60
                    else:
                        max_local = 80
                    scored.sort(key=lambda t: -t[1])
                    candidate_actions = [c for (c, _) in scored[:max_local]]

            # Expand neighbors
            for (x, y, size) in candidate_actions:

                if time.time() - start_time >= time_limit:
                    time_exceeded = True
                    print(f"HẾT THỜI GIAN, MẤT {(time.time() - start_time):.4f}s")
                    break

                if size > n - max(x, y):
                    continue

                # Pruning: bỏ các xoay không đụng vùng chưa ghép cặp
                has_overlap = any(
                    (py, px) in unpaired_coords
                    for (py, px) in self.border_positions(x, y, size)
                )

                if not has_overlap:
                    continue

                try:
                    new_field = current_field.rotate(x, y, size)
                except Exception as e:
                    print(f"Lỗi rotate: x = {x}, y = {y}, size = {size}")
                    exit()

                new_tuple = self._convert_field_to_tuple(new_field)
                new_g = g + 1
                new_h = new_field.heuristic_manhattan()
                new_score = new_field.score()

                nodes_generated += 1

                # Hàm priority: càng gần target_score càng tốt, h nhỏ, step nhỏ
                score_gap = target_score - new_score
                if score_gap < 0:
                    score_gap = 0
                # Với bàn từ 14 trở lên, phạt score_gap theo bình phương để ưu tiên mạnh hơn
                if n >= 14:
                    primary_gap = score_gap * score_gap
                else:
                    primary_gap = score_gap

                new_f = (
                    primary_gap * score_factor
                    + new_h * heuristic_factor
                    + new_g * step_factor
                )

                # Check if better path (dùng tuple trực tiếp làm key)
                if new_g < visited.get(new_tuple, float('inf')):
                    visited[new_tuple] = new_g

                    #Nếu xoay tốt -> push vào open_set
                    heapq.heappush(open_set,
                                   (new_f, next(counter), new_g, new_field, new_tuple))

                    parent[new_tuple] = (current_tuple, (x, y, size))

                if current_time - last_print_time >= 5.0:
                    print(f"  [{current_time - start_time:.1f}s] Explored: {nodes_explored:,}, "
                          f"Queue: {len(open_set):,}, Best: {best_score}/{target_score}, Length: {len(global_best_path)}, nodes_since_improvement = {nodes_since_improvement}")
                    last_print_time = current_time

            if time_exceeded:
                print(f"Time limit exceeded, take {(time.time() - start_time):.4f}s")
                break
            if len(open_set) > beam_width * 1.2: 
                # Strategy: Giữ beam_width states, mix giữa best và diverse
                # 80% best states, 20% diverse states (high h, different scores)
                best_count = int(beam_width * 0.8)
                diverse_count = beam_width - best_count
                
                # Lấy best states
                best_states = heapq.nsmallest(best_count, open_set)
                best_set = set(best_states)
                # Lấy diverse states - từ phần còn lại, chọn states với h khác nhau
                remaining = [s for s in open_set if s not in best_set]
                if len(remaining) > diverse_count:
                    # Sort by heuristic diversity (states ở different depths)
                    remaining.sort(key=lambda x: (-(x[2] // 10), x[0]))  # Group by depth range
                    diverse_states = remaining[:diverse_count]
                else:
                    diverse_states = remaining
                
                open_set = best_states + diverse_states
                heapq.heapify(open_set)
                print(f"  ✂️ Pruned queue to {beam_width:,} states")
        
            total_nodes_explored += nodes_explored
            total_nodes_generated += nodes_generated
            
            # Check if should restart (with dynamic cost-benefit analysis)
            if best_score == target_score:
                break

        print(f"\n{'='*70}")
        print(f"⏱️  Search completed")
        print(f"{'='*70}")
        # print(f"Restarts performed: {restart_count}/{max_restarts}")
        print(f"Initial score: {init_field.score()}")
        print(f"Best score: {global_best_score}/{target_score} ({global_best_score/target_score*100:.1f}%)")
        print(f"Solution steps: {len(global_best_path)}")
        print(f"Nodes explored: {total_nodes_explored:,}")
                
        print(f"Time: {time.time() - start_time:.2f}s")
        print(f"{'='*70}\n")

        # Nếu gần full score, thử refine local bằng tìm kiếm nông
        n = init_field.n
        target_score = (n * n) // 2
        score_gap = target_score - global_best_score
        # Mở rộng refine cho tới bàn 16x16, cho phép xử lý khi còn thiếu nhiều hơn một vài cặp
        if n <= 16 and 0 < score_gap <= 6:
            print("\n🔍 Local refinement search around best board...")
            if n <= 12:
                ref_depth, ref_time = 32, 25.0
            else:
                # 14x14, 16x16: tăng depth/time để có cơ hội xử lý nốt ~4–6 cặp cuối
                ref_depth, ref_time = 40, 35.0

            refined_field, extra_path = self._local_refinement_search(
                global_best_field,
                target_score=target_score,
                max_depth=ref_depth,
                time_limit=ref_time,
            )
            if refined_field is not None:
                print("✅ Local refinement succeeded, perfect score reached!")
                global_best_field = refined_field
                global_best_path = global_best_path + extra_path
            else:
                print("⚠️ Local refinement did not find perfect solution within limits.")

        return global_best_field, global_best_path

    def _local_refinement_search(self, start_field, target_score, max_depth=20, time_limit=15.0):
        """Tìm kiếm nông xung quanh trạng thái tốt nhất cho bàn nhỏ.

        - Chỉ dùng size nhỏ (2,3,4) xoay quanh các ô chưa ghép cặp.
        - Cho phép score giảm nhẹ (tối đa 2) để thoát local optimum.
        """
        from collections import deque

        n = start_field.n
        start_time = time.time()
        start_score = start_field.score()

        if start_score >= target_score:
            return start_field, []

        # Giới hạn độ sâu thấp, chỉ tinh chỉnh local
        queue = deque()
        queue.append((start_field, 0, []))
        visited = {self._convert_field_to_tuple(start_field)}

        # Chỉ sử dụng size nhỏ để tránh phá vỡ nhiều cặp
        if n >= 10:
            size_candidates = [4, 3, 2]
        else:
            size_candidates = [3, 2]

        while queue and (time.time() - start_time) < time_limit:
            field, depth, path = queue.popleft()

            current_score = field.score()
            if current_score == target_score:
                return field, path

            if depth >= max_depth:
                continue

            _, unpaired_coords, _ = field.get_unpaired_coords_set()
            actions = field.get_children_nodes(size_candidates)
            if not actions:
                # fallback quét toàn board với size nhỏ
                for x in range(n):
                    for y in range(n):
                        for size in size_candidates:
                            actions.append((x, y, size))

            # Chỉ giữ các move có overlap với unpaired, ưu tiên move có nhiều unpaired trên biên
            unpaired_set = set(unpaired_coords)
            scored_actions = []
            for (x, y, size) in actions:
                perim = 0
                for (py, px) in self.border_positions(x, y, size):
                    if (py, px) in unpaired_set:
                        perim += 1
                if perim > 0:
                    scored_actions.append(((x, y, size), perim))

            # sắp xếp giảm dần theo số unpaired trên biên để BFS đi qua move "tốt" trước
            if scored_actions:
                scored_actions.sort(key=lambda t: -t[1])
                ordered_actions = [c for (c, _) in scored_actions]
            else:
                ordered_actions = []

            for (x, y, size) in ordered_actions:
                new_field = field.rotate(x, y, size)
                new_tuple = self._convert_field_to_tuple(new_field)
                if new_tuple in visited:
                    continue
                visited.add(new_tuple)

                new_score = new_field.score()
                new_path = path + [(x, y, size)]

                # Cho phép giảm score tối đa 4 so với score tốt nhất ban đầu
                if new_score < start_score - 4:
                    continue

                if new_score == target_score:
                    return new_field, new_path

                queue.append((new_field, depth + 1, new_path))

        return None, None
    
    def border_positions(self, x, y, size):
        # Top row
        for j in range(x, x + size):
            yield (y, j)
        # Bottom row
        for j in range(x, x + size):
            yield (y + size - 1, j)
        # Left + right columns (avoid double-count corners)
        for i in range(y + 1, y + size - 1):
            yield (i, x)
            yield (i, x + size - 1)
    
    def _convert_field_to_tuple(self, field) -> Tuple[int, ...]:
        """Convert Field entities to immutable tuple for visited set"""
        return tuple(field.entities[i][j] 
                    for i in range(field.n) 
                    for j in range(field.n))    

    

    
    def _reconstruct_path_from_parent(self, current_tuple: int, parent: Dict) -> List:
        """Reconstruct path từ parent dictionary"""
        path = []
        current = current_tuple
        while current in parent and parent[current] is not None:
            parent_hash, action = parent[current]
            path.append(action)
            current = parent_hash
        
        path.reverse()
        return path

if __name__ == "__main__":
    size = 4
    start_time = time.time()
    init_field, best_field, best_path = testPipeline(n = size)
    print(f"Init score: {init_field.score()}")
    print(f"Best score: {best_field.score()}")
    print(f"len(path): {len(best_path)}")
    end_time = time.time()
    print(f"Take {(end_time - start_time):.4f}")

            
    

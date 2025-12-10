import numpy as np
import math

class Field:
    def __init__(self, size, entities, mappings):
        self.n = size
        self.entities = entities
        self.mappings = mappings
        # #hash theo từng dòng để tối ưu -> chỉ hash lại các row ảnh hưởng
        self.row_hashes = [hash(tuple(r)) for r in self.entities] 
    
    def clone(self):
        import copy
        return Field(self.n, copy.deepcopy(self.entities), self.mappings)
    
    #Có thể tối ưu bằng cách: xây dựng sẵn mapping cho tất cả các action. 
    def rotate(self, x, y, size):
        # Lấy perm từ mapping đã build trước
        local_map = self.mappings[(x, y, size)]
        
        
        # Flatten board (list is faster than numpy)
        flat = [v for row in self.entities for v in row]
        
        # Copy để tạo board mới
        new_flat = flat[:] 
        
        for dst, src in local_map.items():
            new_flat[dst] = flat[src]
        
        # Chuyển về 2D
        n = self.n
        new_entities = [new_flat[i*n : (i+1)*n] for i in range(n)]
        
        # Tạo Field mới
        new_field = Field(n, new_entities, self.mappings)
        
        # Cập nhật lại Hash
        for row_idx in range(y, y + size):
            new_field.row_hashes[row_idx] = hash(tuple(new_entities[row_idx]))
        
        return new_field
    
    #Đếm số cặp kề nhau
    def score(self):
        n, score = self.n, 0
        for i in range(n):
            for j in range(n):
                v = self.entities[i][j]
                if i + 1 < n and self.entities[i + 1][j] == v: score += 1
                if j + 1 < n and self.entities[i][j + 1] == v: score += 1
        return score
    
    def get_unpaired_coords_set(self):
        """CẢI TIẾN #3: Trả về set tọa độ của các ô chưa được ghép cặp"""
        n = self.n
        positions = {}
        average_manhattan = 0
        
        for y in range(n):
            for x in range(n):
                v = self.entities[y][x]
                positions.setdefault(v, []).append((y, x))
        
        unpaired_coords = set()
        for coords in positions.values():
            (y1, x1), (y2, x2) = coords
            manhattan_distance = abs(y1 - y2) + abs(x1 - x2)
            average_manhattan += manhattan_distance
            
            if manhattan_distance > 1:
                unpaired_coords.add((y1, x1))
                unpaired_coords.add((y2, x2))
        average_manhattan /= (n * n // 2) 
        
        return positions, unpaired_coords, average_manhattan
    
    def get_unpaired(self):
        n = self.n
        positions = {}
        unpaired = []
        
        for i in range(n):
            for j in range(n):
                v = self.entities[i][j]
                positions.setdefault(v, []).append((i, j))
        
        for coords in positions.values():
            #Cặp tọa độ của 2 giá trị trùng nhau
            (y1, x1), (y2, x2) = coords
            manhattan_distance = abs(y1 - y2) + abs(x1 - x2)
            #Cặp tọa độ không kề nhau
            if manhattan_distance > 1: 
                unpaired.append(coords)
        return unpaired
    
    def get_children_nodes(self, sizes):
        '''
        Dựa vào trạng thái ban đầu, 
        kích thước của bài toán sẽ có những size khác nhau
        ''' 
        candidate_actions = []
        n = self.n
        _, unpaired_coords, _ = self.get_unpaired_coords_set()
        # unpaired_coords lưu theo dạng (y, x)
        for uy, ux in unpaired_coords:
            for size in sizes:
                S = size - 1
                
                possible_x = [ux, ux - S]
                possible_y = [uy, uy - S]
                
                for x in possible_x:
                    for y in possible_y:
                        if x < 0 or y < 0:
                            continue
                        if x + size > n or y + size > n:
                            continue
                        candidate_actions.append((x, y, size))
        #remove duplicate
        candidate_actions = list(set(candidate_actions))
        return candidate_actions
    
    def heuristic_manhattan(self) -> int:
        """
        Lower heuristic = closer to goal.
        """
        n = self.n
        entities = self.entities

        # Build position map: value -> list of positions
        pos_map = {}
        for r in range(n):
            for c in range(n):
                v = entities[r][c]
                pos_map.setdefault(v, []).append((r, c))

        h = 0
        for v, positions in pos_map.items():
            if len(positions) != 2:
                continue  # defensive
            (r1, c1), (r2, c2) = positions
            # Manhattan distance between the pair
            manhattan = abs(r1 - r2) + abs(c1 - c2)
            h += manhattan

        return h    
    def heuristic_SA(self, en_area = None):
        n = self.n 
        score = 0
        #Xác định vùng ảnh hưởng
        if en_area is None:
            x_, y_, size_ = 0, 0, n
        else:
            x_, y_, size_ = en_area
        
        #Cộng điểm cho các ô kề nhau
        for i in range(y_, y_ + size_):
            for j in range(x_, x_ + size_):
                v = self.entities[i][j]
                if i + 1 < n and self.entities[i+1][j] == v:
                    score += 100
                if j + 1 < n and self.entities[i][j+1] == v:
                    score += 100
        
        #Bonus cho các block đúng nhiều
        #------

        #Trừ khoảng cách manhattan cho các cặp chưa kề
        position = {}
        for i in range(n):
            for j in range(n):
                v = self.entities[i][j]
                position.setdefault(v, []).append((i, j))
        for coords in position.values():
            (y1, x1), (y2, x2) = coords
            manhattan_distance = abs(y1 - y2) + abs(x1 - x2)
            if manhattan_distance > 1:
                score -= manhattan_distance 
        
        return score

    def incremental_hash(self):
        return hash(tuple(self.row_hashes))
    
    def __str__(self):
        return "\n".join(" ".join(map(str, row)) for row in self.entities)
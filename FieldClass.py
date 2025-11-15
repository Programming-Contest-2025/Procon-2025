import numpy as np


#Chưa tối ưu, permute của một action (x, y, size) chỉ nên có size^2 giá trị, nhưng 
#hiện tại lại lưu n^2 giá trị 

#Không thể sử dụng mappings được tạo bởi một 24x24 cho các n khác -> sai index
#khi flatten
def build_mapping(n):
    """
    Tạo mapping cho phép xoay 90° theo chiều kim đồng hồ
    Công thức: giá trị tại (i,j) sẽ đi tới vị trí (j, size-1-i)
    """
    mappings = {}  # dict: (x, y, size) -> perm
    for size in range(2, n+1):
        for y in range(n - size + 1):
            for x in range(n - size + 1):
                perm = list(range(n*n))  # mặc định: giữ nguyên
                for i in range(size):
                    for j in range(size):
                        # Vị trí nguồn trong ma trận lớn
                        src_pos = (y+i) * n + (x+j)
                        # Xoay 90° clockwise: (i,j) -> (j, size-1-i)
                        new_i = j
                        new_j = size - 1 - i
                        dst_pos = (y+new_i) * n + (x+new_j)
                        # perm[i] = j nghĩa là: new_flat[i] = flat[j]
                        # Nên: new_flat[dst_pos] = flat[src_pos] => perm[dst_pos] = src_pos
                        perm[dst_pos] = src_pos
                mappings[(x, y, size)] = perm
    return mappings



class Field:
    def __init__(self, size, entities, mappings):
        self.n = size
        self.entities = entities
        self.mappings = mappings
        #hash theo từng dòng để tối ưu -> chỉ hash lại các row ảnh hưởng
        self.row_hashes = [hash(tuple(r)) for r in self.entities] 
    
    def clone(self):
        import copy
        return Field(self.n, copy.deepcopy(self.entities))
    
    #Có thể tối ưu bằng cách: xây dựng sẵn mapping cho tất cả các action. 
    def rotate(self, x, y, size):
        # Lấy perm từ mapping đã build trước
        perm = self.mappings[(x, y, size)][:self.n * self.n]
        
        # Flatten field
        flat = np.array(self.entities).reshape(-1)
        
        # Apply perm: new_flat[i] = flat[perm[i]]
        new_flat = flat[perm]  # FIXED: dùng fancy indexing
        
        # Reshape lại thành ma trận
        new_entities = new_flat.reshape(self.n, self.n).tolist()
        
        # Tạo field mới
        new_field = Field(self.n, new_entities, self.mappings)
        
        # Update row_hashes
        for i in range(y, y + size):
            new_field.row_hashes[i] = hash(tuple(new_field.entities[i]))
        
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
    
    def get_unpaired(self):
        """
        Lấy danh sách các cặp giá trị và các cặp chưa kề nhau
        Returns:
            positions (dict): {value: [(y1, x1), (y2, x2)]}
            unpaired (list): [((y1, x1), (y2, x2)), ...] - các cặp chưa kề nhau
        """
        n = self.n
        positions = {}
        unpaired = []
        
        for y in range(n):
            for x in range(n):
                v = self.entities[y][x]
                positions.setdefault(v, []).append((y, x))
        
        for coords in positions.values():
            #Cặp tọa độ của 2 giá trị trùng nhau
            (y1, x1), (y2, x2) = coords
            manhattan_distance = abs(y1 - y2) + abs(x1 - x2)
            #Cặp tọa độ không kề nhau
            if manhattan_distance > 1: 
                unpaired.append(coords)
        return positions, unpaired
    
    def get_unpaired_coords_set(self):
        """CẢI TIẾN #3: Trả về set tọa độ của các ô chưa được ghép cặp"""
        n = self.n
        positions = {}
        
        for y in range(n):
            for x in range(n):
                v = self.entities[y][x]
                positions.setdefault(v, []).append((y, x))
        
        unpaired_coords = set()
        for coords in positions.values():
            (y1, x1), (y2, x2) = coords
            manhattan_distance = abs(y1 - y2) + abs(x1 - x2)
            if manhattan_distance > 1:
                unpaired_coords.add((y1, x1))
                unpaired_coords.add((y2, x2))
        
        return unpaired_coords
    
    def get_manhattan_distance_sum(self):
        """
        Tính tổng khoảng cách Manhattan của tất cả các cặp chưa kề nhau
        """
        _, unpaired = self.get_unpaired()
        total = 0
        for coords in unpaired:
            (y1, x1), (y2, x2) = coords
            total += abs(y1 - y2) + abs(x1 - x2)
        return total
    
    def calculate_heuristic_sa(self, en_area=None):
        """
        Heuristic cho Simulated Annealing
        Tính điểm dựa trên các ô kề nhau và khoảng cách Manhattan
        """
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




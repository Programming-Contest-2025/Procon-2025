import numpy as np


#Chưa tối ưu, permute của một action (x, y, size) chỉ nên có size^2 giá trị, nhưng 
#hiện tại lại lưu n^2 giá trị 

#Không thể sử dụng mappings được tạo bởi một 24x24 cho các n khác -> sai index
#khi flatten
def build_mapping(n):
    mappings = {}  # dict: (x, y, size) -> perm
    for size in range(2, n+1):
        for y in range(n - size + 1):
            for x in range(n - size + 1):
                perm = list(range(n*n))  # mặc định: giữ nguyên
                for i in range(size):
                    for j in range(size):
                        src = (y+i) * n + (x+j)
                        dst = (y+j) * n + (x+size-1-i)
                        perm[src] = dst
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
        
        # Apply perm (src -> dst)
        new_flat = np.empty_like(flat)
        new_flat[perm] = flat  # quan trọng!
        
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
    
    def incremental_hash(self):
        return hash(tuple(self.row_hashes))
    
    def get_unpaired(self):
        """Trả về dictionary các giá trị chưa ghép cặp và list tọa độ các cặp"""
        n = self.n
        positions = {}  # value -> list of (y, x) positions
        
        # Tìm tất cả positions của mỗi value
        for i in range(n):
            for j in range(n):
                v = self.entities[i][j]
                if v not in positions:
                    positions[v] = []
                positions[v].append((i, j))
        
        # Lọc ra các values chưa ghép cặp (không có 2 ô kề nhau)
        unpaired = {}
        unpaired_coords = []
        
        for v, coords in positions.items():
            if len(coords) != 2:
                continue
            
            (y1, x1), (y2, x2) = coords
            # Check nếu 2 ô kề nhau
            is_paired = (abs(y1 - y2) + abs(x1 - x2) == 1)
            
            if not is_paired:
                unpaired[v] = coords
                unpaired_coords.append(coords)
        
        return unpaired, unpaired_coords
    
    def get_unpaired_coords_set(self):
        """Trả về set các tọa độ (i, j) của các ô chưa ghép cặp"""
        _, unpaired_coords = self.get_unpaired()
        coords_set = set()
        
        for coords in unpaired_coords:
            (y1, x1), (y2, x2) = coords
            coords_set.add((y1, x1))
            coords_set.add((y2, x2))
        
        return coords_set
    
    def __str__(self):
        return "\n".join(" ".join(map(str, row)) for row in self.entities)




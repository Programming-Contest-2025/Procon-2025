import random
from FieldClass import Field
from Solver import Solver
def build_mapping(n):
    """
    Tạo mapping MINIMAL:
    Chỉ lưu các chỉ số bị ảnh hưởng bởi xoay trong block size×size.
    """
    mappings = {}

    for size in range(2, n+1):
        for y in range(n - size + 1):
            for x in range(n - size + 1):

                local_map = {}  # chỉ chứa các vị trí thay đổi

                for i in range(size):
                    for j in range(size):
                        src_pos = (y+i) * n + (x+j)

                        # xoay 90 độ
                        new_i = j
                        new_j = size - 1 - i
                        dst_pos = (y+new_i) * n + (x+new_j)

                        local_map[dst_pos] = src_pos
                mappings[(x, y, size)] = local_map
                
    return mappings

def generateField(n):
    values = list(range(n * n // 2)) * 2
        
    random.shuffle(values)
    
    #Chia thành ma trận NxN
    matrix = []
    for i in range(n):
        matrix.append(values[i * n : (i + 1) * n])
    
    return matrix

def testPipeline(n):
    entities = generateField(n=n)
    mappings = build_mapping(n=n)
    field = Field(size=n, entities=entities, mappings=mappings)
    solver = Solver(init_field=field)
    params = solver.default_params_for_size(n=n)

    # Chạy SA để tìm trạng thái xuất phát tốt cho A*
    best_field, best_path = solver.simulated_annealing(
        max_neighbors=900,
        T_start=1.0,
        T_min=1e-6,
        alpha=0.9,
    )

    # Chạy A* tối ưu trên trạng thái tốt nhất từ SA
    solver2 = Solver(init_field=best_field)
    best_field2, best_path2 = solver2.OptimizedAStar(
        time_limit=params["time_limit"],
        max_depth=params["max_depth"],
        beam_width=params["beam_width"],
    )

    final_best_path = best_path + best_path2
    return field, best_field2, final_best_path


if __name__ == "__main__":
    n = 4
    field = generateField(n)
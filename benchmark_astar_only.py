import time
import random

from FieldClass import Field
from Solver import Solver
from utils import build_mapping, generateField


def run_single_astar(
    size: int,
    seed: int | None = None,
    use_sa_start: bool = True,
) -> tuple[bool, float, float, int, int]:
    """Chạy RIÊNG A* trên board size×size.

    - Nếu use_sa_start=True: dùng SA để tạo trạng thái xuất phát tốt, nhưng
      chỉ tính thời gian cho pha A* khi benchmark.
    - Nếu use_sa_start=False: A* chạy trực tiếp từ field random ban đầu.

    Trả về:
        (is_perfect, astar_time, total_time, init_score, final_score)
    """
    if seed is not None:
        random.seed(seed)

    entities = generateField(size)
    mappings = build_mapping(size)
    field = Field(size=size, entities=entities, mappings=mappings)

    solver = Solver(init_field=field)
    params = solver.default_params_for_size(size)

    # Chuẩn bị trạng thái xuất phát cho A*
    sa_time = 0.0
    if use_sa_start:
        sa_start = time.time()
        sa_field, sa_path = solver.simulated_annealing()
        sa_time = time.time() - sa_start
        start_field = sa_field
        init_score = sa_field.score()
    else:
        start_field = field
        init_score = field.score()
        sa_path = []

    # Benchmark chỉ A*
    astar_solver = Solver(init_field=start_field)
    astar_start = time.time()
    best_field, best_path = astar_solver.OptimizedAStar(
        time_limit=params["time_limit"],
        max_depth=params["max_depth"],
        beam_width=params["beam_width"],
    )
    astar_time = time.time() - astar_start
    total_time = sa_time + astar_time

    final_score = best_field.score()
    target = size * size // 2
    is_perfect = final_score == target

    mode = "SA-start" if use_sa_start else "raw-start"
    print(
        f"[A* size={size} {mode}] init={init_score}, final={final_score}/{target}, "
        f"A* time={astar_time:.2f}s, total={total_time:.2f}s, "
        f"astar_steps={len(best_path)}"
    )

    return is_perfect, astar_time, total_time, init_score, final_score


def main():
    """Benchmark riêng cho A*.

    Mặc định chạy cho các size nhỏ/mid để tune trước:
        sizes = [6, 8, 10, 12, 14]
    Bạn có thể chỉnh list này tuỳ ý.
    """
    sizes = [10]
    runs_per_size = 3
    use_sa_start = False  # muốn test A* từ trạng thái đã qua SA

    for size in sizes:
        print("\n" + "=" * 50)
        print(f"A* ONLY – TEST SIZE {size}x{size}")
        print("=" * 50)

        success = 0
        total_astar_time = 0.0
        total_total_time = 0.0

        for r in range(runs_per_size):
            ok, t_astar, t_total, _, _ = run_single_astar(
                size=size, seed=r, use_sa_start=use_sa_start
            )
            total_astar_time += t_astar
            total_total_time += t_total
            if ok:
                success += 1

        print(
            f"Summary A* size={size}: perfect={success}/{runs_per_size}, "
            f"avg_astar_time={total_astar_time / runs_per_size:.2f}s, "
            f"avg_total_time={total_total_time / runs_per_size:.2f}s"
        )


if __name__ == "__main__":
    main()

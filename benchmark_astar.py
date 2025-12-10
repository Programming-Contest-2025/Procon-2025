import time
import random

from FieldClass import Field
from Solver import Solver
from utils import build_mapping, generateField


def run_single(size: int, seed: int | None = None) -> tuple[bool, float, int, int]:
    """Chạy 1 lần SA + A* trên board size×size.

    Trả về (is_perfect, elapsed_time, init_score, final_score).
    """
    if seed is not None:
        random.seed(seed)

    entities = generateField(size)
    mappings = build_mapping(size)
    field = Field(size=size, entities=entities, mappings=mappings)

    solver = Solver(init_field=field)
    params = solver.default_params_for_size(size)

    start = time.time()

    # SA khởi tạo
    sa_field, sa_path = solver.simulated_annealing()

    # A* tinh chỉnh
    astar_solver = Solver(init_field=sa_field)
    best_field, best_path = astar_solver.OptimizedAStar(
        time_limit=params["time_limit"],
        max_depth=params["max_depth"],
        beam_width=params["beam_width"],
    )

    elapsed = time.time() - start
    init_score = field.score()
    final_score = best_field.score()
    target = size * size // 2
    is_perfect = final_score == target

    print(
        f"[size={size}] init={init_score}, final={final_score}/{target}, "
        f"time={elapsed:.2f}s, steps={len(sa_path) + len(best_path)}"
    )

    return is_perfect, elapsed, init_score, final_score


def main():
    # Kiểm tra độ tin cậy cho size nhỏ và trung bình
    sizes = [6, 8, 10, 12, 14, 16, 20]
    runs_per_size = 3

    for size in sizes:
        print("\n" + "=" * 50)
        print(f"TEST SIZE {size}x{size}")
        print("=" * 50)

        success = 0
        total_time = 0.0

        for r in range(runs_per_size):
            ok, t, _, _ = run_single(size, seed=r)
            total_time += t
            if ok:
                success += 1

        print(
            f"Summary size={size}: perfect={success}/{runs_per_size}, "
            f"avg_time={total_time / runs_per_size:.2f}s"
        )

if __name__ == "__main__":
    main()

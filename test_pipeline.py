from utils import testPipeline
import time, random

random.seed(42)

def main():
    size = 24
    start_time = time.time()
    init_field, best_field, best_path = testPipeline(n = size)
    print(f"Init score: {init_field.score()}")
    print(f"Best score: {best_field.score()}")
    print(f"Length path: {len(best_path)}")

if __name__ == "__main__":
    main()
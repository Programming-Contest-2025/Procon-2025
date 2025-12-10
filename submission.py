from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import requests
import json
import time
from FieldClass import Field
from Solver import Solver
from utils import *
from postProcessing import post_processing
from fastapi.middleware.cors import CORSMiddleware
from config import TOKEN, URL
import random
random.seed(42)

def getAnswer(url, headers, question_id):
    """
    Get the question data and solve it.
    """
    try:
        s_time = time.time()
        
        response = requests.get(f"{url}/question/{question_id}", headers=headers, verify=False).json()
        e_time = time.time()
        # Extract size and entities
        question_data = json.loads(response["question_data"])
        field = question_data["field"]
        size, entities = field["size"], field["entities"]

        #Khởi tạo mapping để xoay sẵn
        mapping_result = build_mapping(n=size)

        #Khởi tạo đối tượng field
        custom_field = Field(size=size, entities=entities, mappings=mapping_result)
        e_time2 = time.time()
        #Khởi tạo đối tượng solver chạy thuật toán
        SA_solver = Solver(init_field=custom_field, max_depth=500)
        

        # Select optimal parameters based on size
        params = SA_solver.default_params_for_size(size)

        print(f"\n📊 Problem size: {size}x{size}")
        print(f"Original score: {custom_field.score()}")
        best_field, best_path = SA_solver.simulated_annealing()
        
        e_time3 = time.time()
        print(f"Time to get answer: {(e_time3 - s_time):.4f}")
        print("*" * 50)
        print(f"Final result:")
        print(f"Init score: {custom_field.score()}")
        print(f"Best score: {best_field.score()}")
        print(f"SA length: {len(best_path)}")
        return best_field, best_path
    except Exception as e:
        print("REQUEST KHÔNG THÀNH CÔNG!")
        print(f"Lỗi: {e}")
        raise e

def submitAnswer(url, headers, question_id):
    
    try:
        best_field, best_path = getAnswer(url=url, headers=headers, question_id=question_id)
        
        #----ĐỊNH DẠNG KẾT QUẢ NỘP BÀI----#
        submission_result = {
            "ops": []
        }
        for x, y, size in best_path:
            sol = {"x": x, "y": y, "n": size}
            submission_result['ops'].append(sol)

        #----GỬI KẾT QUẢ ĐẾN SERVER----#
        payload = {"question_id": question_id, "answer_data": submission_result}
        resultSubmitted = requests.post(f"{url}/answer", json=payload, headers=headers, verify=False).json()
        print("*" * 70)    
        print(f"{'NỘP BÀI THÀNH CÔNG':^70}")
        print("*" * 70)
        
        #Lấy id câu trả lời
        answer_id = resultSubmitted["id"]

        
        #----LẤY ĐIỂM SỐ CỦA KẾT QUẢ----#
        print("\n⏳ Đang chờ server chấm điểm...", end="", flush=True)
        answer = requests.get(f"{url}/answer/{answer_id}", headers=headers, verify=False).json()
        score_data = json.loads(answer["score_data"])

        print(" ✅ Đã chấm xong!\n")

        # Khi chấm xong mới đọc điểm
        final_score = score_data.get("final_score", "Không có điểm")
        target_score = best_field.n * best_field.n // 2
        
        print(f"{'='*70}")
        print(f"🎯 KẾT QUẢ CUỐI CÙNG")
        print(f"Server Score:      {final_score}")
        print(f"Local Match Count: {best_field.score()}/{target_score}")
        
    except Exception as e:
        print(f"Error: {e}")
        print(f"KHÔNG THỂ LẤY CÂU TRẢ LỜI !")

def main():
    start_time = time.time()
    url = URL
    headers = {"Authorization": TOKEN}
    question_id = 133
    submitAnswer(url=url, headers=headers, question_id=question_id)
    
    end_time = time.time()
    print(f"TỐN {(end_time - start_time):.4f}s để giải")

if __name__ == '__main__':
    main()

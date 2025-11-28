# swagger doc: https://proconvn.duckdns.org/docs

import requests
import json
from FieldClass import Field
from Solver import build_mapping, Solver
from postProcessing import removeDuplicate, removeSameState
import time
import urllib3

# Tắt warning SSL
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
def getAnswer(url, headers, question_id):
    try:
        response = requests.get(f"{url}/question/{question_id}", headers=headers, verify=False).json()
        #Lấy size và entities
        question_data = json.loads(response['question_data'])
        field = question_data['field']
        print(f"field = {field}")
        size, entities = field['size'], field['entities']
        print(f"size: {size} \n entities = {entities}")
        #Lấy sẵn mapping rotate cho size tương ứng
        mapping_result = build_mapping(n = size)
        #Tạo object FIELD
        custom_field = Field(size = size, entities = entities, mappings=mapping_result)
        
        #Tạo object để giải bài toán với max_depth phù hợp
        solver = Solver(init_field = custom_field, max_depth=500)
        
        # Chọn parameters tối ưu dựa trên size — dùng helper trong Solver
        # (moved to Solver to centralize tuning)
        params = Solver.default_params_for_size(size)
        
        print(f"\n📊 Problem size: {size}x{size}")
        print(f"🎯 TARGET: {(size*size)//2} pairs (Priority #1)")
        print(f"📋 Parameters selected:")
        for key, value in params.items():
            print(f"   - {key}: {value}")
        print(f"\n💡 Strategy: Maximize pairs first, then minimize moves")
        print()
        
        #Sử dụng OptimizedAStar với parameters tối ưu
        best_field, path = solver.OptimizedAStar(**params)

        return best_field, custom_field, path
    except Exception as e:
        print("REQUEST KHÔNG THÀNH CÔNG!")
        print(f"Lỗi: {e}")

def submitAnswer(url, headers, question_id):
    
    try:
        best_field, custom_field, best_path = getAnswer(url=url, headers=headers, question_id=question_id)
        print(f"\n{'='*70}")
        print(f"📋 VALIDATING SOLUTION")
        print(f"{'='*70}")
        print(f"BEFORE processing - len(best_path): {len(best_path)}")
        
        # # Remove duplicates
        # best_path = removeDuplicate(best_path)
        # print(f"AFTER removeDuplicate - len(best_path): {len(best_path)}")
        
        # # Remove same state
        # best_path = removeSameState(path=best_path, custom_field=custom_field)
        # print(f"AFTER removeSameState - len(best_path): {len(best_path)}")
        
        #Định dạng kết quả nộp bài giống yêu cầu
        submission_result = {
            "ops": []
        }
        for x, y, size in best_path:
            sol = {"x": x, "y": y, "n": size}
            submission_result['ops'].append(sol)

        #Gửi kết quả đến server
        payload = {"question_id": question_id, "answer_data": submission_result}
        resultSubmitted = requests.post(f"{url}/answer", json=payload, headers=headers, verify=False).json()
        
        print("*" * 70)    
        print(f"{'NỘP BÀI THÀNH CÔNG':^70}")
        print("*" * 70)
        
        #Lấy id câu trả lời
        answer_id = resultSubmitted["id"]

        
        # Chờ server chấm điểm xong
        print("\n⏳ Đang chờ server chấm điểm...", end="", flush=True)
        time.sleep(7)
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
        
        if best_field.score() == target_score:
            print(f"PERFECT - All pairs matched!")
        else:
            print(f"INCOMPLETE - {target_score - best_field.score()} pairs missing")
        
        print(f"{'='*70}\n")
        # print("Chi tiết:", score_data)
    except:
        print(f"KHÔNG THỂ LẤY CÂU TRẢ LỜI !")
    
def testLocal(url, headers, question_id):
    """Test solution locally without submitting"""
    
    try:
        best_field, custom_field, best_path = getAnswer(url=url, headers=headers, question_id=question_id)
        
        print(f"\n{'='*70}")
        print(f"LOCAL TESTING (NO SUBMISSION)")
        print(f"{'='*70}")
        print(f"BEFORE processing - len(best_path): {len(best_path)}")
        
        # Remove duplicates
        best_path = removeDuplicate(best_path)
        print(f"AFTER removeDuplicate - len(best_path): {len(best_path)}")
        
        # Remove same state
        best_path = removeSameState(path=best_path, custom_field=custom_field)
        print(f"AFTER removeSameState - len(best_path): {len(best_path)}")
        
        # ===== VALIDATION: Apply path to initial field =====
        print(f"\n🔍 Validating solution by applying moves...")
        validation_field = custom_field
        initial_score = custom_field.score()
        target_score = (custom_field.n * custom_field.n) // 2
        
        print(f"\nInitial field state:")
        print(validation_field)
        print(f"Initial score: {initial_score}/{target_score}")
        
        for step_idx, (x, y, size) in enumerate(best_path):
            prev_score = validation_field.score()
            validation_field = validation_field.rotate(x, y, size)
            step_score = validation_field.score()
            
            # Print all steps for debugging
            score_change = step_score - prev_score
            indicator = "📈" if score_change > 0 else "📉" if score_change < 0 else "➡️"
            print(f"  {indicator} Step {step_idx + 1}: rotate(x={x}, y={y}, size={size}) -> score {prev_score} → {step_score} ({score_change:+d})")
        
        final_validation_score = validation_field.score()
        
        print(f"\n📍 Final field state:")
        print(validation_field)
        
        print(f"\n{'='*70}")
        print(f"📊 LOCAL TEST RESULTS")
        print(f"Initial score:     {initial_score}/{target_score} ({initial_score/target_score*100:.1f}%)")
        print(f"Expected score:    {best_field.score()}/{target_score} ({best_field.score()/target_score*100:.1f}%)")
        print(f"Validated score:   {final_validation_score}/{target_score} ({final_validation_score/target_score*100:.1f}%)")
        print(f"Number of moves:   {len(best_path)}")
        print(f"Score improvement: {final_validation_score - initial_score:+d}")
        
        if final_validation_score == best_field.score():
            print(f"Solution is VALID!")
        else:
            print(f"WARNING: Solution mismatch!")
            print(f"Expected: {best_field.score()}, Got: {final_validation_score}")
        
        print(f"{'='*70}\n")
        
        return validation_field, best_path
    except:
        print("TEST KHÔNG THÀNH CÔNG!!!")

def main():
    TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MiwibmFtZSI6IkhDTVVURS5Qcm9jb24iLCJpc19hZG1pbiI6ZmFsc2UsImlhdCI6MTc2NDIzNzc1OSwiZXhwIjoxNzY0NDEwNTU5fQ.MOJEZsZ6hFEpA0Ke1G08EWPcIIuD2oq_xrZ3n-pSzYc"
    url = "http://112.137.129.202:8000/"
    headers = {"Authorization": TOKEN}


    #ID câu hỏi
    question_id = 6
    # MODE: 'test' để test local, 'submit' để nộp bài
    MODE = 'submit'
    
    if MODE == 'test':
        print("\n🧪 RUNNING IN TEST MODE (No submission)")
        testLocal(url=url, headers=headers, question_id=question_id)
    else:
        print("\n📤 RUNNING IN SUBMIT MODE")
        submitAnswer(url=url, headers=headers, question_id=question_id)

if __name__ == '__main__':
    main()
    
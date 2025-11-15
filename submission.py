# swagger doc: https://proconvn.duckdns.org/docs

import requests
import json
from FieldClass import Field
from Solver import build_mapping, Solver
from postProcessing import removeDuplicate, removeSameState
import time
import urllib3

# Tắt warning SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
def getAnswer(url, headers, question_id):
    question = requests.get(f"{url}/question/{question_id}", headers=headers, verify=False).json()
    
    #Lúc lấy về là string json, nên phải load để thành dict
    field = json.loads(question['question_data'])['field']
    
    #Lấy size và entities
    size, entities = field['size'], field['entities']

    #Lấy sẵn mapping rotate cho size tương ứng
    mapping_result = build_mapping(n = size)
    #Tạo object FIELD
    custom_field = Field(size = size, entities = entities, mappings=mapping_result)
    
    #Tạo object để giải bài toán với max_depth phù hợp
    solver = Solver(init_field = custom_field, max_depth=500)
    
    # Chọn parameters tối ưu dựa trên size
    if size <= 8:
        # Size nhỏ: Tìm kiếm sâu để đạt perfect score
        params = {
            'weight': 1.2,           
            'time_limit': 300,      
            'stuck_threshold': 3000,  # Restart sớm hơn
            'sa_iterations': 5,       # 5 lần restart
            'max_depth': 120,
            'beam_width': 600000,     # Giới hạn queue
            'enable_restart': True    # Bật restart
        }
    elif size <= 12:
        # Size trung bình: Với heuristic thông minh hơn + lookahead
        params = {
            'weight': 1.0,            # Heuristic tốt, không cần weight cao
            'time_limit': 300,     
            'stuck_threshold': 5000,  # Search sâu hơn nữa trước restart
            'sa_iterations': 8,       # Ít restart, focus vào quality
            'max_depth': 180,         # Depth cao để cho lookahead hoạt động
            'beam_width': 1000000,    # Beam rất lớn để keep diverse paths
            'enable_restart': True
        }
    elif size <= 16:
        # Size lớn: Tăng SA để thoát local optima nhưng vẫn ưu tiên pairs
        params = {
            'weight': 1.5,           
            'time_limit': 300,       
            'stuck_threshold': 5000, 
            'sa_iterations': 10,     
            'max_depth': 150,
            'beam_width': 1000000,
            'enable_restart': True
        }
    elif size <= 20:
        # Size rất lớn: Aggressive nhưng vẫn cố gắng đạt maximum pairs
        params = {
            'weight': 1.8,           
            'time_limit': 300,     
            'stuck_threshold': 6000, 
            'sa_iterations': 12,     
            'max_depth': 180,
            'beam_width': 1200000,
            'enable_restart': True
        }
    else:
        # Size khổng lồ (>20): Rất aggressive
        params = {
            'weight': 2.0,           
            'time_limit': 300,       
            'stuck_threshold': 8000, 
            'sa_iterations': 15,     
            'max_depth': 200,
            'beam_width': 1500000,
            'enable_restart': True
        }
    
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

def submitAnswer(url, headers, question_id):
    
    best_field, custom_field, best_path = getAnswer(url=url, headers=headers, question_id=question_id)
    print(f"\n{'='*70}")
    print(f"📋 VALIDATING SOLUTION")
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
    
    for step_idx, (x, y, size) in enumerate(best_path):
        validation_field = validation_field.rotate(x, y, size)
        step_score = validation_field.score()
        if step_idx < 5 or step_idx >= len(best_path) - 2:  # Print first 5 and last 2 steps
            print(f"  Step {step_idx + 1}: rotate({x}, {y}, {size}) -> score = {step_score}")
    
    final_validation_score = validation_field.score()
    target_score = (custom_field.n * custom_field.n) // 2
    
    print(f"\n{'='*70}")
    print(f"📊 VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"Initial score:     {initial_score}/{target_score}")
    print(f"Expected score:    {best_field.score()}/{target_score}")
    print(f"Validated score:   {final_validation_score}/{target_score}")
    
    if final_validation_score == best_field.score():
        print(f"✅ Solution is VALID!")
    else:
        print(f"❌ WARNING: Solution mismatch!")
        print(f"   Expected: {best_field.score()}, Got: {final_validation_score}")
        print(f"\n🔧 Trying to fix solution...")
        
        # Re-validate step by step to find the issue
        test_field = custom_field
        valid_path = []
        
        for idx, (x, y, size) in enumerate(best_path):
            prev_score = test_field.score()
            test_field = test_field.rotate(x, y, size)
            new_score = test_field.score()
            
            # Only keep moves that don't decrease score significantly
            if new_score >= prev_score or (new_score >= prev_score - 1):
                valid_path.append((x, y, size))
            else:
                print(f"   Skipping step {idx + 1}: rotate({x}, {y}, {size}) - score dropped from {prev_score} to {new_score}")
        
        if len(valid_path) < len(best_path):
            print(f"   Removed {len(best_path) - len(valid_path)} problematic moves")
            best_path = valid_path
            
            # Re-validate
            test_field = custom_field
            for x, y, size in best_path:
                test_field = test_field.rotate(x, y, size)
            
            print(f"   New validated score: {test_field.score()}/{target_score}")
    
    print(f"\n📤 Final path length: {len(best_path)}")
    print(f"{'='*70}\n")
    
    # ===== CHECK IF WE NEED TO CONTINUE SEARCHING =====
    if final_validation_score < target_score:
        print(f"⚠️  WARNING: Not yet optimal solution!")
        print(f"   Current: {final_validation_score}/{target_score} pairs")
        print(f"   Missing: {target_score - final_validation_score} pairs")
        print(f"   💡 Consider increasing time_limit or adjusting parameters")
        print(f"{'='*70}\n")

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
    server_pairs = score_data.get("match_count", final_validation_score)
    
    print(f"{'='*70}")
    print(f"🎯 KẾT QUẢ CUỐI CÙNG")
    print(f"{'='*70}")
    print(f"📊 SCORING BREAKDOWN:")
    print(f"   Priority 1 - Pairs:  {server_pairs}/{target_score} pairs")
    print(f"   Priority 2 - Moves:  {len(best_path)} moves")
    print(f"   Priority 3 - Time:   (submitted)")
    print()
    print(f"Server Score:      {final_score}")
    print(f"Local Match Count: {validation_field.score()}/{target_score}")
    
    if final_validation_score == target_score:
        print(f"✅ PERFECT - All pairs matched!")
    else:
        print(f"⚠️  INCOMPLETE - {target_score - final_validation_score} pairs missing")
    
    print(f"{'='*70}\n")
    # print("Chi tiết:", score_data)
    


def testLocal(url, headers, question_id):
    """Test solution locally without submitting"""
    
    best_field, custom_field, best_path = getAnswer(url=url, headers=headers, question_id=question_id)
    
    print(f"\n{'='*70}")
    print(f"📋 LOCAL TESTING (NO SUBMISSION)")
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
    
    print(f"\n📍 Initial field state:")
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
    print(f"{'='*70}")
    print(f"🎯 SCORING CRITERIA (Priority Order):")
    print(f"   1. Number of pairs (higher is better)")
    print(f"   2. Number of moves (lower is better)")
    print(f"   3. Submission time (earlier is better)")
    print()
    print(f"Initial score:     {initial_score}/{target_score} ({initial_score/target_score*100:.1f}%)")
    print(f"Expected score:    {best_field.score()}/{target_score} ({best_field.score()/target_score*100:.1f}%)")
    print(f"Validated score:   {final_validation_score}/{target_score} ({final_validation_score/target_score*100:.1f}%)")
    print(f"Number of moves:   {len(best_path)}")
    print(f"Score improvement: {final_validation_score - initial_score:+d}")
    
    if final_validation_score == best_field.score():
        print(f"✅ Solution is VALID!")
    else:
        print(f"❌ WARNING: Solution mismatch!")
        print(f"   Expected: {best_field.score()}, Got: {final_validation_score}")
    
    print()
    if final_validation_score == target_score:
        print(f"🎉 PERFECT SOLUTION! Maximum pairs achieved!")
        print(f"   ⭐ Priority 1: ✅ {final_validation_score}/{target_score} pairs")
        print(f"   ⭐ Priority 2: {len(best_path)} moves (optimize if possible)")
    elif final_validation_score >= target_score * 0.9:
        print(f"👍 GOOD SOLUTION (>90%)")
        print(f"   ⚠️  Priority 1: {final_validation_score}/{target_score} pairs - Missing {target_score - final_validation_score} pairs")
        print(f"   💡 Try to improve pairs count first!")
    elif final_validation_score >= target_score * 0.7:
        print(f"😐 ACCEPTABLE SOLUTION (>70%)")
        print(f"   ⚠️  Priority 1: {final_validation_score}/{target_score} pairs - Missing {target_score - final_validation_score} pairs")
        print(f"   💡 Need more improvements on pairs count!")
    else:
        print(f"😞 POOR SOLUTION (<70%)")
        print(f"   ❌ Priority 1: {final_validation_score}/{target_score} pairs - Missing {target_score - final_validation_score} pairs")
        print(f"   💡 Focus on increasing pairs count!")
    
    print(f"{'='*70}\n")
    
    return validation_field, best_path

def main():
    TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MiwibmFtZSI6IkhDTVVURS5Qcm9jb24iLCJpc19hZG1pbiI6ZmFsc2UsImlhdCI6MTc2MzIwMjcxNiwiZXhwIjoxNzYzMzc1NTE2fQ.hnWBlQDQGBdRy3uy1REDymKDWYAt_P6TwAiO2t6Z4_0"
    url = "https://112.137.129.202"
    headers = {"Authorization": TOKEN}

    #ID câu hỏi
    question_id = 1
    
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
    
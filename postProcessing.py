#Nhận vào một chuỗi các action nhưng đang chưa tối ưu
#Thực hiện rút gọn để giảm số bước
from FieldClass import Field

def removeDuplicate(path : list) -> list:
    """ 
    Nếu có 4 action liên tiếp trùng nhau thì thực hiện xóa cả 4 action đó
    Vì nó quay lại trạng thái cũ
    """
    
    result = []
    run = 1
    for i in range(1, len(path)):
        if path[i] == path[i - 1]:
            run += 1
        else:
            result.extend([path[i - 1]] * (run % 4))
            run = 1
    return result

def removeSameState(path : list, custom_field : Field) -> list:
    """ 
    Xóa các hành động làm cho trạng thái quay lại trạng thái 
    đã gặp trong quá khứ
    """
    field = custom_field
    seen = {field.incremental_hash() : 0}
    result = []
    
    for (x, y, size) in path:
        field = field.rotate(x=x, y=y, size=size)
        h = field.incremental_hash()
        
        #Quay lại trạng thái đã đi
        #Tất cả các hành động từ trạng thái trước đó đến trạng thái hiện tại
        #là vô nghĩa
        if h in seen:
            #Xóa toàn bộ giữa hai mốc
            rollback_idx = seen[h]
            result = result[:rollback_idx]
            
            #Khôi phục field bằng cách mô phỏng lại từ đầu
            field = custom_field
            for (xx, yy, ss) in result:
                field = field.rotate(xx, yy, ss)
            
            #Làm mới map seen
            seen = {custom_field.incremental_hash() : 0}
            f_tmp = custom_field
            for i, (xx, yy, ss) in enumerate(result, 1):
                f_tmp = f_tmp.rotate(xx, yy, ss)
                seen[f_tmp.incremental_hash()] = i
        
        else:
            result.append((x, y, size))
            seen[h] = len(result)
    
    return result
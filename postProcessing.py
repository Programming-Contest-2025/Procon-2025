from FieldClass import Field

def removeDuplicate(path: list) -> list:
    """
    Xóa các chuỗi action lặp lại 4 lần (vì quay 4 lần trả về trạng thái cũ).
    Giữ lại run % 4 lần mỗi nhóm run của same action.
    """
    if not path:
        return []

    result = []
    prev = path[0]
    run = 1
    for action in path[1:]:
        if action == prev:
            run += 1
        else:
            keep = run % 4
            if keep:
                result.extend([prev] * keep)
            # reset
            prev = action
            run = 1

    # xử lý run cuối cùng
    keep = run % 4
    if keep:
        result.extend([prev] * keep)

    return result


def removeSameState(path: list, init_field: Field) -> list:
    """
    Duyệt path từ init_field, nếu gặp state đã thấy trước đó (hash), 
    thì rollback all actions between hai lần gặp đó (vì tạo cycle),
    và tiếp tục.
    Trả về path đã rút gọn (không có các đoạn tạo lại state cũ).
    """
    field = init_field
    seen = {field.incremental_hash(): 0}  # map hash -> length(result) at that time
    result = []

    for action in path:
        x, y, size = action
        field = field.rotate(x, y, size)
        h = field.incremental_hash()

        if h in seen:
            # rollback to the earlier occurrence index
            rollback_idx = seen[h]
            # keep prefix of result up to rollback_idx
            result = result[:rollback_idx]

            # rebuild field and seen from init_field using current result
            field = init_field
            seen = {field.incremental_hash(): 0}
            for i, (xx, yy, ss) in enumerate(result, 1):
                field = field.rotate(xx, yy, ss)
                seen[field.incremental_hash()] = i
            # continue (we already applied this action and it matched seen,
            # so we effectively "discard" the looped segment)
        else:
            result.append(action)
            seen[h] = len(result)

    return result


def post_processing(path: list, init_field: Field):
    """
    Chạy removeDuplicate rồi removeSameState, trả về path rút gọn.
    Gợi ý: sau khi có path rút gọn, nếu cần bạn có thể rebuild field từ init_field.
    """
    path = removeDuplicate(path=path)
    path = removeSameState(path=path, init_field=init_field)
    return path

def apply_path(init_field, path):
    f = init_field
    for (x,y,size) in path:
        f = f.rotate(x,y,size)
    return f

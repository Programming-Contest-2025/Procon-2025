import random

def generateField(n):
    values = list(range(n * n // 2)) * 2
        
    random.shuffle(values)
    
    #Chia thành ma trận NxN
    matrix = []
    for i in range(n):
        matrix.append(values[i * n : (i + 1) * n])
    
    return matrix


if __name__ == "__main__":
    n = 4
    field = generateField(n)
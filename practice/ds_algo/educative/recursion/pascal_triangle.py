from typing import List


def printPascal(n: int) -> List[int]:
    # Base case
    if n == 0:
        return [1]
    else:
        line: List[int] = [1]
        # recursive case
        previous_line: List[int] = printPascal(n - 1)
        for i in range(len(previous_line) - 1):
            line.append(previous_line[i] + previous_line[i + 1])
        line += [1]
    return line


def iterative_printPascal(numRows: int) -> List[int]:
    """
    Iterative pascal triangle
    """
    out: List[int] = [[1]]

    for i in range(numRows - 1):
        temp: List[int] = [0] + out[-1] + [0]
        row: list = []
        for j in range(len(out[-1]) + 1):
            row.append(temp[j] + temp[j + 1])
        out.append(row)
    return out[-1]


assert printPascal(5) == [1, 5, 10, 10, 5, 1]
assert printPascal(1) == [1, 1]
assert printPascal(2) == [1, 2, 1]

# need to add one to get last
assert iterative_printPascal(5 + 1) == [1, 5, 10, 10, 5, 1]
assert iterative_printPascal(1 + 1) == [1, 1]
assert iterative_printPascal(2 + 1) == [1, 2, 1]

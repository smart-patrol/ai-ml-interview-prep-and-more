# Min Heap
class Heap:
    def __init__(self):
        self.heap = [0]

    def push(self, val):
        self.heap.append(val)
        i = len(self.heap) - 1

        # Percolate up
        while i > 1 and self.heap[i] < self.heap[i // 2]:
            tmp = self.heap[i]
            self.heap[i] = self.heap[i // 2]
            self.heap[i // 2] = tmp
            i = i // 2

    def pop(self):
        if len(self.heap) == 1:
            return None
        if len(self.heap) == 2:
            return self.heap.pop()

        # save the smallest element
        res = self.heap[1]
        # Move last value to root
        self.heap[1] = self.heap.pop()
        i = 1
        # Percolate down - root elements to correct position in heap
        # loop continues as long as left child exists and is smaller than parent
        while 2 * i < len(self.heap):
            # check if right child exists and is smaller left child
            if (
                2 * i + 1 < len(self.heap)
                and self.heap[2 * i + 1] < self.heap[2 * i]
                and self.heap[i] > self.heap[2 * i + 1]
            ):
                # Swap right child
                tmp = self.heap[i]
                self.heap[i] = self.heap[2 * i + 1]
                self.heap[2 * i + 1] = tmp
                i = 2 * i + 1
            # no right child, left child is smaller
            elif self.heap[i] > self.heap[2 * i]:
                # Swap left child
                tmp = self.heap[i]
                self.heap[i] = self.heap[2 * i]
                self.heap[2 * i] = tmp
                i = 2 * i
            else:
                break
        return res  # smallest element saved in the heap

    def top(self):
        if len(self.heap) > 1:
            return self.heap[1]
        return None

    def heapify(self, arr):
        # 0-th position is moved to the end
        # this ensures that the root node is smaller than its children to keep the heap property
        arr.append(arr[0])

        self.heap = arr
        # set index of last parent node
        cur = (len(self.heap) - 1) // 2
        # rebuild heap from the bottom up
        while cur > 0:
            # Percolate down to maintain heap property
            i = cur
            while 2 * i < len(self.heap):
                # right child exists and is smaller left child
                if (
                    2 * i + 1 < len(self.heap)
                    and self.heap[2 * i + 1] < self.heap[2 * i]
                    and self.heap[i] > self.heap[2 * i + 1]
                ):
                    # Swap right child
                    tmp = self.heap[i]
                    self.heap[i] = self.heap[2 * i + 1]
                    self.heap[2 * i + 1] = tmp
                    i = 2 * i + 1
                # no right child, left child is smaller
                elif self.heap[i] > self.heap[2 * i]:
                    # Swap left child
                    tmp = self.heap[i]
                    self.heap[i] = self.heap[2 * i]
                    self.heap[2 * i] = tmp
                    i = 2 * i
                else:
                    break
            # move up towards the root
            # all parent nodes heapify before chil

            cur -= 1

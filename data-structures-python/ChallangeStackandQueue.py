from Queue import Queue
from Stack import Stack


# 1.Push first k elements in queue in a stack.
# 2.Pop Stack elements and enqueue them at the end of queue
# 3.Dequeue queue elements till "k" and append them at the end of queue

def reverseK(queue: Queue, k: int) -> Queue:

    if queue.is_empty() is True or k > queeu.size() or k < 0:
        return None


    stack = Stack()
    for i in range(k):
        stack.push(queue.dequeue())
    while not stack.is_empty():
        queue.enqueue(stack.pop())
    size = queue.size()
    for i in range(size - k):
        queue.enqueue(queue.dequeue())

    return queue

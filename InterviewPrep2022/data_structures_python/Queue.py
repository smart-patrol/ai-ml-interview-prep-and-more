from DoublyLinkedList import DoublyLinkedList


class Queue:
    def __init__(self) -> None:
        self.items = DoublyLinkedList()

    def is_empty(self) -> bool:
        return self.items.length == 0

    def front(self):
        if self.is_empty():
            return None
        return self.items.get_head()

    def rear(self):
        if self.is_empty():
            return None
        return self.items.tail_node()

    def size(self):
        return self.items.length

    def enqueue(self, value):
        self.items.insert_tail(value)

    def dequeue(self):
        return self.items.remove_head()


if __name__ == "__main__":
    queue_obj = Queue()
    print("queue.enqueue(2);")
    queue_obj.enqueue(2)
    print("queue.enqueue(4);")
    queue_obj.enqueue(4)
    print("queue.enqueue(6);")
    queue_obj.enqueue(6)
    print("queue.enqueue(8);")
    queue_obj.enqueue(8)
    print("queue.enqueue(10);")
    queue_obj.enqueue(10)

    queue_obj.print_list()

    print("is_empty(): " + str(queue_obj.is_empty()))
    print("front(): " + str(queue_obj.front()))
    print("rear(): " + str(queue_obj.rear()))
    print("size(): " + str(queue_obj.size()))
    print("Dequeue(): " + str(queue_obj.dequeue()))
    print("Dequeue(): " + str(queue_obj.dequeue()))
    print("queue.enqueue(12);")
    queue_obj.enqueue(12)
    print("queue.enqueue(14);")
    queue_obj.enqueue(14)

while queue_obj.is_empty() is False:
    print("Dequeue(): " + str(queue_obj.dequeue()))

print("is_empty(): " + str(queue_obj.is_empty()))

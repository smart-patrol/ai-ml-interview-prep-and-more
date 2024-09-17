"""
This is a doubly linked list implementation of a deque, allowing O(1) operations at both ends.
Dummy head and tail nodes simplify edge cases and prevent null pointer exceptions.
The isEmpty() method checks if the deque is empty in O(1) time.
append() and appendleft() add elements to the right and left ends, respectively.
pop() and popleft() remove and return elements from the right and left ends, respectively.
All operations (isEmpty, append, appendleft, pop, popleft) have O(1) time complexity.
The implementation handles empty deque cases by returning -1 for pop operations.
"""


class ListNode:
    def __init__(self, val, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next


# Deque implementation using a doubly linked list
class Deque:

    def __init__(self):
        # Initialize with dummy head and tail nodes for easier edge case handling
        self.head = ListNode(-1)
        self.tail = ListNode(-1)
        self.head.next = self.tail
        self.tail.prev = self.head

    def isEmpty(self) -> bool:
        # Deque is empty if head and tail are adjacent
        return self.head.next == self.tail

    def append(self, value: int) -> None:
        # Add new node to the right end (before tail)
        new_node = ListNode(value)
        last_node = self.tail.prev

        # Update pointers to insert new node
        last_node.next = new_node
        new_node.prev = last_node
        new_node.next = self.tail
        self.tail.prev = new_node

    def appendleft(self, value: int) -> None:
        # Add new node to the left end (after head)
        new_node = ListNode(value)
        first_node = self.head.next

        # Update pointers to insert new node
        self.head.next = new_node
        new_node.prev = self.head
        new_node.next = first_node
        first_node.prev = new_node

    def pop(self) -> int:
        # Remove and return the rightmost element
        if self.isEmpty():
            return -1  # Return -1 if deque is empty
        last_node = self.tail.prev
        value = last_node.val
        prev_node = last_node.prev

        # Update pointers to remove last node
        prev_node.next = self.tail
        self.tail.prev = prev_node

        return value

    def popleft(self) -> int:
        # Remove and return the leftmost element
        if self.isEmpty():
            return -1  # Return -1 if deque is empty
        first_node = self.head.next
        value = first_node.val
        next_node = first_node.next

        # Update pointers to remove first node
        self.head.next = next_node
        next_node.prev = self.head

        return value

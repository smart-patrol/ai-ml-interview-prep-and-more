class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class LinkedList:

    def __init__(self):
        # Init the list with a 'dummy' node which makes
        # removing a node from the beginning of list easier.
        self.head = ListNode(-1)
        self.tail = self.head

    def get(self, index: int) -> int:
        # will return the value of the ith node (0-indexed). If the index is out of bounds, return -1.
        curr = self.head.next
        ln = 0
        while curr:
            if ln == index:
                return curr.val
            curr = curr.next
            ln += 1
        return -1

    def insertHead(self, val: int) -> None:
        # will insert a node with val at the tail of the list.
        new_node = ListNode(val)
        new_node.next = self.head.next
        self.head.next = new_node
        if self.tail == self.head:  # list empty before insertion
            self.tail = new_node

    def insertTail(self, val: int) -> None:
        # will insert a node with val at the tail of the list.
        new_node = ListNode(val)
        self.tail.next = new_node
        self.tail = new_node

    def remove(self, index: int) -> bool:
        i = 0
        curr = self.head
        while i < index and curr:
            i += 1
            curr = curr.next

        # Remove the node ahead of curr
        if curr and curr.next:
            if curr.next == self.tail:
                self.tail = curr
            curr.next = curr.next.next
            return True
        return False

    def getValues(self) -> List[int]:
        # return an array of all the values in the linked list, ordered from head to tail.
        curr = self.head.next
        vals = []
        while curr:
            vals.append(curr.val)
            curr = curr.next
        return vals

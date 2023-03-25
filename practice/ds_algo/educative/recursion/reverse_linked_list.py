class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, new_data):
        new_node = Node(new_data)

        if self.head is None:  # If head node is null
            self.head = new_node
            return

        last = self.head
        while last.next:
            last = last.next
        last.next = new_node  # add new node to end of list

    def printList(self):
        temp = self.head
        while temp:
            print(temp.data)
            temp = temp.next


def helperFunction(myLinkedList, current, previous):
    # base case
    if current.next is None:
        myLinkedList.head = current
        current.next = previous
        return
    nxt = current.next
    current.next = previous

    # recursive case
    helperFunction(myLinkedList, nxt, current)


def reverse(myLinkedList):
    # check if head node of the linked list is null or not
    if myLinkedList.head is None:
        return
    helperFunction(myLinkedList, myLinkedList.head, None)


# Driver Code
myLinkedList = LinkedList()
myLinkedList.append(3)
myLinkedList.append(4)
myLinkedList.append(7)
myLinkedList.append(11)

print("Original Linked List")
myLinkedList.printList()

reverse(myLinkedList)
print("\nReversed Linked List")
myLinkedList.printList()

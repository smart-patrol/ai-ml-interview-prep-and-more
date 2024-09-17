from HashEntry import HashEntry


class HashTable:
    def __init__(self):
        # size of the hash table
        self.slots = 10
        # number entries in table, counter
        self.size = 0
        # list of entries, default None
        self.bucket = [None] * self.slots
        self.threshold = 0.6

    # Helper functions
    def get_size(self):
        return self.size

    def is_empty(self):
        return self.size == 0

    # Hash function
    def get_index(self, key):
        hash_code = hash(key)
        index = hash_code % self.slots
        return index

    def resize(self):
        new_slots = self.slots
        new_bucket = [None] * new_slots
        # rehash all items into new slots
        for item in self.bucket:
            head = item
            while head is not None:
                new_index = hash(head.key) % new_slots
                if new_bucket[new_index] is None:
                    new_bucket[new_index] = HashEntry(head.key, head.value)
                else:
                    node = new_bucket[new_index]
                    while node is not None:
                        if node.key == head.key:
                            node.value = head.value
                            node = None
                        elif node.next is None:
                            node.next = HashEntry(head.key, head.value)
                            node = None
                        else:
                            node = node.next

                head = head.next
        self.bucket = new_bucket
        self.slots = new_slots

    def insert(self, key, value):
        # Find the node with the given key
        b_index = self.get_index(key)
        if self.bucket[b_index] is None:
            self.bucket[b_index] = HashEntry(key, value)
            print(key, "-", value, "inserted at index:", b_index)
            self.size += 1
        else:
            head = self.bucket[b_index]
            while head is not None:
                if head.key == key:
                    head.value = value
                    break
                elif head.next is None:
                    head.next = HashEntry(key, value)
                    print(key, "-", value, "inserted at index:", b_index)
                    self.size += 1
                    break
                head = head.next

        load_factor = float(self.size) / float(self.slots)
        # Checks if 60% of the entries in table are filled, threshold = 0.6
        if load_factor >= self.threshold:
            self.resize()
        # return a value for a given key

    def search(self, key):
        # find the node with the given key
        b_index = self.get_index(key)
        head = self.bucket[b_index]
        # search key in the slots
        while head is not None:
            if head.key == key:
                return head.value
            head = head.next
        return None

    def delete(self, key):
        # Find index
        b_index = self.get_index(key)
        head = self.bucket[b_index]
        # If key exists at first slot
        if head.key == key:
            self.bucket[b_index] = head.next
            print(key, "-", head.value, "deleted")
            # Decrease the size by one
            self.size -= 1
            return self
        # Find the key in slots
        prev = None
        while head is not None:
            # If key exists
            if head.key == key:
                prev.next = head.next
                print(key, "-", head.value, "deleted")
                # Decrease the size by one
                self.size -= 1
                return
            # Else keep moving in chain
            prev = head
            head = head.next

        # If key does not exist
        print("Key not found")
        return

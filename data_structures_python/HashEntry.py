class HashEntry:
    def __init__(self, key, data):
        self.key = key
        self.value = data
        self.next = None

    def __str__(self):
        return str(self.key) + " " + str(self.value)

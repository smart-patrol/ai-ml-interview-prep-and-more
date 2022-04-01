from typing import List
from collections import defaultdict, OrderedDict
from LinkedList import LinkedList
from Node import Node

def union(list1, list2):
    if list1.is_empty():
        return list2
    if list2.is_empty():
        return list1
    seen = set()
    result = LinkedList()

    start = list1.get_head()

    while start:
        seen.add(start.data)
        start = start.next_element
    
    start = list2.get_head()

    while start:
        seen.add(start.data)
        start = start.next_element

    for x in seen:
        result.insert_at_head(x)
    return result
       

def intersection(list1, list2):

    result = LinkedList()
    visited_nodes = set()  # Keep track of all the visited nodes
    current_node = list1.get_head()

    # Traversing list1 and adding all unique nodes into the hash set
    while current_node is not None:
        value = current_node.data
        if value not in visited_nodes:
            visited_nodes.add(value)  # Visiting current_node for first time
        current_node = current_node.next_element

    start = list2.get_head()

    # Traversing list 2
    # Nodes which are already present in visited_nodes are added to result
    while start is not None:
        value = start.data
        if value in visited_nodes:
            result.insert_at_head(start.data)
        start = start.next_element
    result.remove_duplicates()
    return result


def remove_duplicates2(lst:LinkedList) -> LinkedList:
    seen = set()
    curr = lst.get_head()
    prev = None
    while curr:
        if curr.data in seen:
            prev.next_element = curr.next_element
        else:
            seen.add(curr.data)
            prev = curr
        curr = curr.next_element
    return lst


def remove_duplicates(lst):
    """
    You will now be implementing the remove_duplicates() function. When a linked list is passed to this function, it removes any node which is a duplicate of another existing node.
    """
    current_node = lst.get_head()
    prev_node = lst.get_head()
    # To store values of nodes which we already visited
    visited_nodes = set()
    # If List is not empty and there is more than 1 element in List
    if not lst.is_empty() and current_node.next_element:
        while current_node:
            value = current_node.data
            if value in visited_nodes:
                # current_node is already in the HashSet
                # connect prev_node with current_node's next element
                # to remove it
                prev_node.next_element = current_node.next_element
                current_node = current_node.next_element
                continue
            # Visiting currentNode for first time
            visited_nodes.add(current_node.data)
            prev_node = current_node
            current_node = current_node.next_element
        
    return lst


def detect_loop(lst):
    """
    By definition, a loop is formed when a node in your linked list points to a previously traversed node.

    You must implement the detect_loop() function which will take a linked list as input and deduce whether or not a loop is present.
    """
    # Used to store nodes which we already visited
    visited_nodes = set()
    current_node = lst.get_head()

    # Traverse the set and put each node in the visitedNodes set
    # and if a node appears twice in the map
    # then it means there is a loop in the set
    while current_node:
        if current_node in visited_nodes:
            return True
        visited_nodes.add(current_node)  # Insert node in visitedNodes set
        current_node = current_node.next_element
    return False

def findFirstUnique2(lst:List[int]) -> int:
    """
    Implement a function, findFirstUnique(lst) that returns the first unique integer in the list. Unique means the number does not repeat and appears only once in the whole list.
    """
    order_counts = OrderedDict()
    order_counts = order_counts.fromkeys(lst, 0)
    for ele in lst:
        order_counts[ele] = order_counts[ele]+1
    for ele in order_counts:
        if order_counts[ele] == 1:
            return ele
    return None


def findFirstUnique(lst:List[int]) -> int:
    counts = {}  # Creating a dictionary
    # Initializing dictionary with pairs like (lst[i],count)
    counts = counts.fromkeys(lst, 0)
    for ele in lst:
        # counts[ele] += 1  # Incrementing for every repitition
        counts[ele] = counts[ele]+1
    answer_key = None
    # filter first non-repeating 
    for ele in lst:
        if (counts[ele] is 1):
            answer_key = ele
            break
    return answer_key


assert findFirstUnique([1, 1, 1, 2]) == 2



def findSum(arr:List[int], k:int) -> List[int]:
    """
    In this problem, you have to implement the findSum(lst,k) function which will take a number k as input and return two numbers that add up to k.

    You have already seen this challenge previously in chapter 2 of this course. Here you would use HashTables for a more efficient solution.
    """
    lkp = set()
    for a in arr:
        if k-a in lkp:
            return [k-a, a]
        lkp.add(a)
    return []

assert findSum([1,2,3,4],6) == [3,4], "Test 1 Failed"

def findSum2(arr:List[int], k:int) -> List[int]:
    lkp = dict()
    for a in arr:
        for a*-1 in lkp:
                return [a, lkp[a*-1]]
        lkp[a-k] = a
    return []


def is_formation_possible(alist:List[int], word:List[str]) -> bool:
    if len(word) < 2 and len(alist) < 2:
        return False
    
    hashmap = dict()
    for letter in word:
        if letter in hashmap:
            hashmap[letter] += 1
        else:
            hashmap[letter] = 1
    
    for i in range(1, len(word)):
        first = word[0:i]
        second = word[i:len(word)]
        check1 = False
        check2 = False

        if first in hashmap:
            check1 = True
        if second in hashmap:
            check2 = True
        
        if check1 and check2:
            return True

    return False


keys = ["the", "hello", "there", "answer",
        "any", "educative", "world", "their", "abc"]

assert is_formation_possible(keys, "helloworld") == True, "Function failed to run"


def is_formation_possible2(alist: List[int], word:List[str]) -> bool:
    """
    You have to implement the is_formation_possible() function which will find whether a given word can be formed by combining two words from a dictionary. We assume that all words are in lower case.
    """
    counter = 0
    window_start = 0
    for window_end in range(1,len(word)+1):
        sub = word[window_start:window_end]

        if sub in alist:
            counter +=1
            window_start = window_end

    return counter >= 2

def find_sub_zero(my_list: List[int]) -> List[int]:
    """
    """
    # Use hash table to store the cumulative sum as a key and
    # the element as the value till which the sum has been calculated
    # Traverse the list and return true if either
    # elem == 0 or sum == 0 or hash table already contains the sum
    # If you completely traverse the list and haven't found any 
    # of the above three conditions, then simply return false
    ht = dict()
    total_sum = 0
    # Traverse through the given list
    for elem in my_list:
        total_sum += elem
        if elem is 0 or total_sum is 0 or ht.get(total_sum) is not None:
            return True
        ht[total_sum] = elem
    return False

my_list = [6, 4, -7, 3, 12, 9]

assert find_sub_zero(my_list) == True


def find_pair(alist:List[int]) -> List[List[int]]:
    """
    In this problem, you have to implement the find_pair() function which will find two pairs, [a, b] and [c, d], in a list such that :

    a+b = c+da+b=c+d

    You only have to find the first two pairs in the list which satisfies the above condition.
    """
    d = defaultdict(list)
    for i in range(len(alist)):
        for j in range(i+1, len(alist)):
            added = alist[i] + alist[j]
            if added not in d:
                d[added] = [alist[i], alist[j]]
            else:
                # added alrdy present in the dictionary
                perv_pair = d[added]
                second_pair = [alist[i], alist[j]]
                return [perv_pair, second_pair]

    return []




def trace_path(adict:dict) -> List[str]:
    """
    You have to implement the trace_path() function which will take in a list of source-destination pairs and return the correct sequence of the whole journey from the first city to the last.
    """
    # Create a reverse dict of the given dict i.e if the given dict has (N,C)
    # then reverse dict will have (C,N) as key-value pair
    # Traverse original dict and see if it's key exists in reverse dict
    # If it doesn't exist then we found our starting point.
    # After the starting point is found, simply trace the complete path
    # from the original dict.
    result = []
    reverse_dict = dict()
    keys = adict.keys()
    for key in keys:
        reverse_dict[adict.get(key)] = key
    # find the starting point
    from_loc = None
    for key in keys:
        if key not in reverse_dict:
            from_loc = key
            break
    # trace the path
    to = adict.get(from_loc)
    while to is not None:
        result.append([from_loc, to])
        from_loc = to
        to = adict.get(to)
    return result


def find_symmetric(alist: List[List[int]]) -> List[List[int]]:
    """
    By definition, (a, b) and (c, d) are symmetric pairs iff, a = d and b = c. In this problem, you have to implement the find_symmetric(list) function which will find all the symmetric pairs in a given list.
    """
    pair_set = set()
    result = []

    for pair in alist:
        # make a tuplle and reverse tuple of the pair
        pair_tuple = tuple(pair)
        pair.reverse()
        reverse_tup = tuple(pair)

        if reverse_tup in pair_set:
            # symmetric pair found
            result.append(list(pair_tuple))
            result.append(list(reverse_tup))
        else:
            # insert teh current tuple into the set
            pair_set.add(pair_tuple)
    
    return result

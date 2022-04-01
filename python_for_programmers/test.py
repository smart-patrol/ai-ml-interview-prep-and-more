from typing import List
from collections import deque
import math
import numpy as np
from collections import Counter, defaultdict
from itertools import groupby



def minRemoveToMakeValid(s: str) -> str:
    """
    Given a string s of '(' , ')' and lowercase English characters.

    Your task is to remove the minimum number of parentheses ( '(' or ')', in any positions ) so that the resulting parentheses string is valid and return any valid string.

    Formally, a parentheses string is valid if and only if:

    It is the empty string, contains only lowercase characters, or
    It can be written as AB (A concatenated with B), where A and B are valid strings, or
    It can be written as (A), where A is a valid string.
    """
    indexes_to_remove = set()
    stack = []
    for i,c in enumerate(s):
        if c not in "()":
            continue
        if c == "(":
            stack.append(i)
        elif not stack:
            indexes_to_remove.add(i)
        else:
            stack.pop()
    indexes_to_remove.update(stack)
    string_builder = []
    for i,c in enumerate(s):
        if i not in indexes_to_remove:
            string_builder.append(c)
    return "".join(string_builder)
    


def isSameTree(p: 'TreeNode', q: 'TreeNode') -> bool:
    """
    Given the roots of two binary trees p and q, write a function to check if they are the same or not.

    Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.
    """
    queue1 = deque([p])
    queue2 = deque([q])
    while queue1 and queue2:
        node1 = queue1.popleft()
        node2 = queue2.popleft()
        if not node1 and not node2:
            continue
        if not node1 or not node2:
            return False
        if node1.val != node2.val:
            return False
        queue1.append(node1.left)
        queue1.append(node1.right)
        queue2.append(node2.left)
        queue2.append(node2.right)
    
    return True
        
    

def intersection(list1:'ListNode', list2:'ListNode') -> 'ListNode':
    """
    The intersection function will return all the elements that are common between two linked lists.
    """
    curr = list2.get_head()
    hashmap = {}
    while curr:
        hashmap.add(curr.data)
        curr = curr.next_element

    curr = list1.gethead()
    prev = None
    while curr:
        if curr.data in hashmap:
            if prev is None:
                list1.head = curr
                prev = curr
            else:
                prev.next_element = curr
                prev = curr
        curr = curr.next_element
    return list1


def reorderList( head: 'ListNode') -> None:
    if not head:
        return 
    
    # find the middle node
    slow = head
    fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    # revese the second half
    prev = None
    curr = slow
    while curr:
        next = curr.next
        curr.next = prev
        prev = curr
        curr = next
    
    # merge the two halves
    first = head
    second = prev
    while second.next:
        first.next, first = second, first.next
        second.next, second = first.next, second.next

    return




def deleteAndEarn(nums: List[int]) -> int:
    """
    You are given an integer array nums. You want to maximize the number of points you get by performing the following operation any number of times:

    Pick any nums[i] and delete it to earn nums[i] points. Afterwards, you must delete every element equal to nums[i] - 1 and every element equal to nums[i] + 1.
    Return the maximum number of points you can earn by applying the above operation some number of times.
    """
    points = defaultdict(int)
    max_number =0

    for num in nums:
        points[num] += num
        max_number = max(max_number, num)

    dp = [0] * (max_number+1)
    dp[1] = points[1]

    for num in range(2, len(dp)):
        dp[num] = max(dp[num-1], points[num] + dp[num-2])
    
    return dp[max_number]


def isValid(s: str) -> bool:
    lookup =  {")": "(", "}": "{", "]": "["}
    stack = []

    for char in s:
        if char in lookup:
            if len(stack) == 0 or lookup[char] != stack.pop():
                return False
            else:
                # valid parantheses
                continue
        else:
            stack.append(char)
    return len(stack) == 0



def eraseOverlapIntervals(intervals: List[List[int]]) -> int:
    """
    Given an array of intervals intervals where intervals[i] = [starti, endi], return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.
    """
    intervals.sort(key=lambda x: x[1])
    count = 0
    for i in range(0, len(intervals)):
        if i == 0:
            continue
        if intervals[i][0] < intervals[i-1][1]:
            count += 1
    return count
    



def subarraysDivByK(nums: List[int], k: int) -> int:
    """Brute force approach"""
        total_sum, cnt = 0, 0
        d = defaultdict(int)
        
        d[total_sum] += 1
        for i in range(0, len(nums)):
            total_sum += nums[i]
            total_sum %= k
            d[total_sum] += 1
        return d[0]
        




def dutch_flag_sort(arr:List[int]):
    low, high = 0, len(arr)-1
    i = 0
    while i <= high:
        if arr[i] == 0:
            arr[i],arr[low] = arr[low], arr[i]
            i+=1
            low+=1
        elif arr[i] == 1:
            i+=1
        else:
            arr[i], arr[high] = arr[high], arr[i]
            high-=1


arr = [1, 0, 2, 1, 0]
assert dutch_flag_sort(arr) == [0, 0, 1, 1, 2]
arr = [2, 2, 0, 1, 2, 0]
assert dutch_flag_sort(arr) == [0, 0, 1, 1, 2]



def sort_list_by_frequency(arr: List[int]) -> List[int]:
    """
    Sort a list by the frequency of elements.
    """
    counter = Counter(arr)
    return sorted(arr, key=lambda x: counter[x], reverse=True)
    


def setZeroes(matrix: List[List[int]]) -> None:
    """
    Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's.
    """
    if len(matrix) == 0:
        return

    mat = np.array(matrix)
    zero_rows = np.where(mat == 0)[0]
    zero_cols = np.where(mat == 0)[1]

    for i in zero_rows:
        mat[i,:] = 0
    for j in zero_cols:
        mat[:,j] = 0
    return mat.tolist()
    


class Node(object):
    def __init__(self, val, neighbors):
        self.val = val
        self.neighbors = neighbors

def cloneGraph(node):
    """
    Given a reference of a node in a connected undirected graph.
    Return a deep copy (clone) of the graph.
    Each node in the graph contains a value (int) and a list (List[Node]) of its neighbors.
    """
    if node is None:
        return None

    visited = set()
    queue = deque()
    queue.append(node)
    visited.add(node)

    while len(queue) > 0:
        curr_node = queue.popleft()
        for neighbor in curr_node.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return node

    
    



def titleToNumber(s: str) -> int:
    """
    Given a string columnTitle that represents the column title as appear in an Excel sheet, return its corresponding column number.
    """
    res = 0
    for i in range(0, len(s)):
        res += (ord(s[i]) - 64) * 26 ** (len(s) - i - 1)
    return res



def maxSubarraySum(nums: List[int]) -> int:
    """
    Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.
    A subarray is a contiguous part of an array.
    """
    dp = [0] * len(nums)
    dp[0] = nums[0]
    max_sum = dp[0]
    for i in range(1, len(nums)):
        dp[i] = max(dp[i-1] + nums[i], nums[i])

        if dp[i] > max_sum:
            max_sum = dp[i]
    return max_sum
    


def removeCoveredIntervals(intervals: List[List[int]]) -> int:
    """
    Given an array intervals where intervals[i] = [li, ri] represent the interval [li, ri), remove all intervals that are covered by another interval in the list.
    The interval [a, b) is covered by the interval [c, d) if and only if c <= a and b <= d.
    Return the number of remaining intervals.
    """
    intervals.sort(key=lambda x: x[0])

    count = 0
    for i in range(0, len(intervals)):
        if intervals[i][0] > intervals[i][1]:
            continue
        if i == 0:
            count += 1
            continue
        if intervals[i][0] > intervals[i-1][1]:
            count += 1
    return count
    



def calcualte_tfidf(docs:List[str]) -> float:
    """You are provided with four documents, numbered 1 to 4, each with a single sentence of text. 
    Determine the identifier of the document  which is the most similar to the first document, 
    as computed according to the TF-IDF scores."""

    def tf(word:str, doc:str) -> float:
        return doc.count(word) / len(doc)
    
    def n_containing(word:str, docs:List[str]) -> int:
        return sum(1 for doc in docs if word in doc)
    
    def idf(word:str, docs:List[str]) -> float:
        return math.log(len(docs) / (1 + n_containing(word, docs)))

    def tfidf(word:str, doc:str, docs:List[str]) -> float:
        return tf(word, doc) * idf(word, docs)

    scores = []
    for doc in docs:
        score = 0
        for word in doc.split():
            score += tfidf(word, doc, docs)
        scores.append(score)
    return scores.index(max(scores)) + 1
    






def binary_search(arr:List[int], left:int, right:int, target:int) -> int:

    if right > left:
        mid = int(left + (right-1)/2)

        if mid == 0 or target > arr[mid-1] and (arr[mid] == target):
            return mid
        
        if arr[mid] < target:
            return binary_search(arr, mid+1, right, target)

        else:
            return binary_search(arr, left, mid-1, target)

    return -1


def is_majority_element(arr: List[int], number:int) -> bool:
    if len(arr) == 0:
        return False
    
    if len(arr) == 1:
        if arr[0] == number:
            return True
        else:
            return False

    bs_result = binary_search(arr, 0, len(arr)-1, number)
    if bs_result<0:
        return False
    
    majority_threshold = int(len(arr)/2)
    if bs_result +  majority_threshold <= len(arr):
        return True
    else:
        return False



def is_number_exists(arr, target):
    return binary_search(arr, 0, len(arr)-1, target)

data,number = [2,5,8,12,45],13
print(is_number_exists(data,number))




def mostCompetitive(nums: List[int], k: int) -> List[int]:
    """
    Given an integer array nums and a positive integer k, return the most competitive subsequence of nums of size k.

    An array's subsequence is a resulting sequence obtained by erasing some (possibly zero) elements from the array.

    We define that a subsequence a is more competitive than a subsequence b (of the same length) if in the first position where a and b differ, subsequence a has a number less than the corresponding number in b. For example, [1,3,4] is more competitive than [1,3,5] because the first position they differ is at the final number, and 4 is less than 5. 
    """
    q = deque()
    count = len(nums) - k

    for i in range(0, len(nums)):
        while len(q) > 0 and nums[q[-1]] > nums[i]:
            q.pop()
        q.append(i)
        if i >= count:
            q.popleft()
    return [nums[i] for i in q]
    

def print_nge(numbers: List[int]) -> None:
    """
    You have been given n non-negative numbers. You have to find the first next greater element for each of these n numbers. For the last element, the next greater element is -1.
    """
    if len(numbers) == 0:
        return
    
    stack = []
    stack.append(numbers[0])

    for i in range(1, len(numbers)):
        while len(stack) >0 and stack[-1] < numbers[i]:
            print(stack[-1], numbers[i])
            stack.pop()
        stack.append(numbers[i])
    
    while len(stack) >0:
        print(stack[-1], -1)
        stack.pop()


print_nge([5,34,4,46,2])
print_nge([9,8,5,7,1])



"""
You have N people standing in a line. Each person has a rank. They are standing in random order. We have to find the captains in the line. A person is a captain if their rank is higher than all the others standing to the right of them in the line. For example, if we are given the ranks as [3,7,5,2,4] then the captains are 7,5, and 4. How can you find all the captains in the line?
"""

def print_captains(ranks:List[int]) -> List[int]:
    """Return captains"""
    if len(ranks) == 0:
        print("No captains")
    max_value = -1
    for i in range(0,len(ranks)):
        j = len(ranks) - i - 1
        if ranks[j] > max_value:
            max_value = ranks[j]
            print(max_value)


arr = [3,7,5,2,4]
print(print_captains(arr))


# def combinationSum(candidates: List[int], target: int) -> List[List[int]]:
#     """
#     Given an array of distinct integers candidates and a target integer target, 
#     return a list of all unique combinations of candidates where the chosen numbers sum to target. 
#     You may return the combinations in any order.
#     The same number may be chosen from candidates an unlimited number of times.
#     Two combinations are unique if the frequency of at least one of the chosen numbers is different.
#     It is guaranteed that the number of unique combinations that sum up to target is less than 150 combinations for the given input.
#     """
#     # use backtracking to sovle the problem
#     # time complexity: O(2^n)
#     # space complexity: O(n)

#     def backtrack(start: int, curr_sum: int, curr_list: List[int]) -> None:
#         if curr_sum == target:
#             res.append(curr_list)
#             return
#         if curr_sum > target:
#             return
#         for i in range(start, len(candidates)):
#             backtrack(i, curr_sum + candidates[i], curr_list + [candidates[i]])

#     res = []
#     backtrack(0, 0, [])
#     return res

# candidates = [2, 3, 6, 7]
# target = 7
# print(combinationSum(candidates, target))



# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# def binaryTreePath(root: 'TreeNode') -> List[str]:
#     """
#     Given the root of a binary tree, return all root-to-leaf paths in any order.
#     A leaf is a node with no children.
#     """
#         def construct_paths(root, path):
#             if root:
#                 path += str(root.val)
#                 if not root.left and not root.right: # if reach a leaf
#                     paths.append(path) # update paths
#                 else:
#                     path += "->" # extend the current path
#                     construct_paths(root.left, path)
#                     construct_paths(root.right, path)
                    
#     paths = []
#     construct_paths(root, '')
#     return paths


# #Input: root = [1,2,3,null,5]
# #Output: ["1->2->5","1->3"]
# #assert binaryTreePath(root) == ["1->2->5", "1->3"]


# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
#     if not root:
#         return []
        
#     paths = []
   
#     stack = [(root, [root.val])]
#     while stack:
#         node, path = stack.pop()
        
#         if not node.left and not node.right:
#             if sum(path) == targetSum:
#                 paths.append(path)
#         if node.left:
#                 stack.append((node.left, path +  [node.left.val]))
#         if node.right:
#                 stack.append((node.right, path + [node.right.val]))
        
#     return paths

# # Input: root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
# # Output: [[5,4,11,2],[5,8,4,5]]


# class Soltuion
#     def pathSum(self, root: Optional[TreeNode], target: int):
            
#             self.numOfPaths = 0
#             self.dfs(root, target)
#             return self.numOfPaths
        
#         def dfs(self,node, target):
#             if node is None:
#                 return
#             # preorder
#             self.test(node, target)
#             self.dfs(node.left, target)
#             self.dfs(node.right, target)
            
#         def test(self, node, target):
#             if node is None:
#                 return
#             if node.val == target:
#                 self.numOfPaths += 1
            
#             self.test(node.left, target-node.val)
#             self.test(node.right, target-node.val)

# # Input: root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8
# # Output: 3
# # Explanation: The paths that sum to 8 are shown.


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def create_linked_list(arr: List[int]) -> List[int]:
    """
    Create a linked list from a list of integers
    """

    if len(arr) == 0:
        return None

    head = ListNode(arr[0])
    curr = head

    for i in range(1, len(arr)):
        curr.next = ListNode(arr[i])
        curr = curr.next
    return head
    

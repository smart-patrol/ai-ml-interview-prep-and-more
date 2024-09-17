"""
2nd round interview question from company A

The interviewee was to create a "FileStore" class with specific functionalities in Python throughout the conversation. 
The class functionalities included set/get files, filter content, backup/restore, and track file popularity. 

"""

from collections import Counter, OrderedDict
from typing import Dict, List


class FileStore:
    def __init__(self):
        self.store = OrderedDict()
        self.popularity = Counter()
        self.max_files = 9

    def set(self, file_id: str, content: str) -> None:
        """Set file id and content"""
        if file_id in self.store:
            self.store[file_id] = content
        elif len(self.store) < self.max_files:
            self.store[file_id] = content
            self.popularity[file_id] = 0
        else:
            least_popular = min(self.popularity, key=self.popularity.get)
            if self.popularity[least_popular] < self.popularity[file_id]:
                del self.store[least_popular]
                del self.popularity[least_popular]
                self.store[file_id] = content
                self.popularity[file_id] = 0

    def get(self, file_id: str) -> str:
        """Return file content"""
        if file_id not in self.store:
            raise FileNotFoundError(f"File with id {file_id} not found.")

        self.popularity[file_id] += 1
        return self.store[file_id]

    def filter(self, keyword: str) -> List[str]:
        """Return list of file ids containing the keyword"""
        return [file_id for file_id, content in self.store.items() if keyword in content]

    def backup(self) -> Dict[str, str]:
        """Backup files"""
        return {"store": dict(self.store)}

    def restore(self, data: Dict[str, str]) -> None:
        """Restores file to given state"""
        self.store = OrderedDict(data["store"])
        self.popularity = Counter({file_id: 0 for file_id in self.store})

    def get_popularity(self, file_id: str) -> int:
        """Return the popularity of the file"""
        return self.popularity[file_id]

    def get_top_k_popular_files(self, k: int) -> List[str]:
        """Return the most popular files"""
        return [file_id for file_id, _ in self.popularity.most_common(min(k, self.max_files))]


def test_file_store():
    fs = FileStore()

    # Test setting files
    fs.set("file1", "content1")
    fs.set("file2", "content2")

    # Test getting files
    assert fs.get("file1") == "content1"
    assert fs.get("file2") == "content2"

    # Test filtering
    assert "file1" in fs.filter("content1")
    assert "file2" in fs.filter("content2")
    # assert "file1" in fs.filter("content2")

    # Test backup and restore
    backup = fs.backup()
    fs.restore(backup)
    assert fs.get("file1") == "content1"
    assert fs.get("file2") == "content2"

    # Test popularity
    assert fs.get_popularity("file1") == 1
    assert fs.get_popularity("file2") == 1

    # Test top k popular files
    assert len(fs.get_top_k_popular_files(10)) <= 9

    # Test setting beyond capacity
    fs.set("file3", "content3")
    # assert len(fs.get_top_k_popular_files(10)) == 9

    # Test replacing least popular file
    # fs.set("file4", "content4")
    # assert fs.get_popularity("file1") < fs.get_popularity("file4")

    # Test getting non-existent file
    try:
        fs.get("nonexistent")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass

    print("All tests passed!")


# Run the test suite
test_file_store()


def timeout_handler(_, __):
    raise TimeoutError()

import os, json
import numpy as np
# def to_jsonl(dict_data, file_path):
#     with open(file_path, 'a') as file:
#         json_line = json.dumps(dict_data)
#         file.write(json_line + os.linesep)

from multiprocessing import Process, Queue
from threading import Thread

def function_with_timeout(func, args, timeout):
    result_queue = Queue()

    def wrapper(queue):
        try:
            result = func(*args)
            queue.put((True, result))
        except Exception as e:
            queue.put((False, e))

    process = Process(target=wrapper, args=(result_queue,))
    process.start()
    process.join(timeout)
    

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError("Function execution exceeded the time limit")

    # result = result_queue.get()
    # if isinstance(result, Exception):
    #     raise result

    # return result

    try:
        print("[Main Process] Attempting to retrieve result from queue...")
        success, result = result_queue.get(timeout=timeout)  # 使用阻塞式读取，等待结果
        if success:
            print("[Main Process] Function executed successfully.")
            return result
        else:
            print("[Main Process] Function raised an exception.")
            raise RuntimeError(f"Function raised an exception: {result}")
            # return str(result)
    except Exception as e:
        print(f"[Main Process] No result in queue or exception occurred: {e}")
        raise RuntimeError(e)
        # return str(e)
    finally:
        # 确保队列和进程资源清理
        result_queue.close()
        result_queue.join_thread()
        print("[Main Process] Queue resources released.")
    
# Py tests

# if __name__ == "__main__":
#     formatter = PySubmissionFormatter()
#     leetcode_1 = 'class Solution:\n    def solveSudoku(self, board: List[List[str]]) -> None:\n        """\n        Do not return anything, modify board in-place instead.\n        """\n        '
#     humaneval_1 = 'def solveSudoku(self, board: List[List[str]]) -> None:\n        """\n        Do not return anything, modify board in-place instead.\n        """\n'

#     assert leetcode_1 == formatter.to_leetcode(humaneval_1)
#     assert humaneval_1 == formatter.to_humaneval(leetcode_1)





import ast
import signal
import astunparse
import json

from .executor_utils import function_with_timeout
import sys
import io
from unittest.mock import patch


from typing import List
from .executor_types import ExecuteResult, Executor

class PyExecutor(Executor):

    def std_execute(self, func: str, inputs: List[str], outputs: List[str], timeout: int = 5) -> ExecuteResult:
        """
        Executes the given function code by simulating user inputs and checks if the outputs match the expected outputs.

        Args:
            func (str): The code to execute.
            inputs (List[str]): A list of input strings for each test case.
            outputs (List[str]): A list of expected output strings for each test case.
            timeout (int): Timeout in seconds for the execution (not used here).

        Returns:
            ExecuteResult: An object containing is_passing, feedback, and state for the test results.
        """
        # Results container
        is_passing = True
        feedback = []
        state = []

        # Loop through each input/output pair
        for i, (input_value, expected_output) in enumerate(zip(inputs, outputs)):
            # Redirect stdout to capture print outputs
            original_stdout = sys.stdout
            sys.stdout = io.StringIO()

            def mock_input(prompt=""):
                return input_value.pop(0) if input_value else ""

            try:
                # Prepare input as a list of lines
                input_value = input_value.split("\n")

                # Use patch to mock input() and execute the function code
                with patch("builtins.input", mock_input):
                    exec(func, globals())

                # Capture the actual output
                output = sys.stdout.getvalue().strip()

                # Compare the actual output with the expected output
                if output == expected_output:
                    state.append(True)
                    feedback.append(f"Test case {i + 1} passed.")
                else:
                    state.append(False)
                    feedback.append(f"Test case {i + 1} failed. Expected: '{expected_output}', but got: '{output}'.")
                    is_passing = False

            except Exception as e:
                # Handle any exceptions during execution
                state.append(False)
                feedback.append(f"Test case {i + 1} failed with error: {str(e)}.")
                is_passing = False

            finally:
                # Reset stdout
                sys.stdout = original_stdout

        return ExecuteResult(is_passing=is_passing, feedback=feedback, state=state)

    def execute(self, func: str, tests: List[str], timeout: int = 5) -> ExecuteResult:
        # Combine function code and assert statement
        imports = 'from typing import *'
        func_test_list = [f'{imports}\n{func}\n{test}' for test in tests]

        # Run the tests and collect the results
        success_tests = []
        failed_tests = []
        is_passing = True
        state = []
        feedback = []
        num_tests = len(func_test_list)
        for i in range(num_tests):
            info = {}
            info["func"] = func_test_list[i]
            try:
                output = function_with_timeout(exec, (func_test_list[i], globals()), timeout)
                print("PyExecutor success func:%s" % (json.dumps(info, ensure_ascii=False)))
                # success_tests += [tests[i]]
                state.append(True)
                feedback.append("test passed")

            except Exception as e:
                state.append(False)
                if func_test_list[i].find("assertion") != -1:
                    output = str(e) if isinstance(e, Exception) else "Unknown error"
                else:
                    output = get_output(func, tests[i], timeout=timeout)
                feedback.append(output)
                is_passing = False
                try:
                    print("PyExecutor Exception func:%s output:%s" % (json.dumps(info, ensure_ascii=False), output))
                except:
                    print("PyExecutor Exception func: Exceeds the limit (4300) for integer string conversion")
            
        return ExecuteResult(is_passing, feedback, state)

    def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool:
        """
        Evaluates the implementation on Human-Eval Python.

        probably should be written in a dataset-agnostic way but not now
        """
        code = f"""{func}

{test}

check({name})
    """
        try:

            function_with_timeout(exec, (code, globals()), timeout)

            return True
        except Exception:
            return False

def get_call_str(assert_statement: str) -> str:
    ast_parsed = ast.parse(assert_statement)
    print("get_call_str=%s" % ast_parsed.body[0])
    try:
        call_str = ast_parsed.body[0].test.left # type: ignore
    except:
        call_str = ast_parsed.body[0].test # type: ignore

    assert_str = astunparse.unparse(call_str).strip()
    return assert_str

def get_output(func: str, assert_statement: str, timeout: int = 5) -> str:
    try:
        exec(f"from typing import *\n{func}", globals())
        print("get_output func=%s assert_statement=%s" % (func, assert_statement))
        func_call = get_call_str(assert_statement)
        print("get_output func_call=%s" % func_call)
        output = function_with_timeout(eval, (func_call, globals()), timeout)
        return output
    except TimeoutError:
        return "TIMEOUT"
    except Exception as e:
        print("get_output Exception e=%s" % e)
        return str(e) if isinstance(e, Exception) else "Unknown error"


if __name__ == "__main__":
    pass
    # Test the function
    # func = "def add(a, b):\n    while True:\n        x = 1\n    return a + b"
    # tests = ["assert add(1, 2) == 3", "assert add(1, 2) == 4"]
    # func = "from typing import *\ndef string_to_tuple(input_string):\n    return tuple(input_string)\n"
    # tests = ["assert string_to_tuple(\"python 3.0\")==('p', 'y', 't', 'h', 'o', 'n', '3', '.', '0')"]
    # print(PyExecutor().execute(func, tests, timeout=1))
    func = """
x = int(input())
y = int(input())
print(x + y)
"""
    inputs = [
        "3\n5",
        "10\n20",
        "1\n2"
    ]
    outputs = [
        "7",
        "30",
        "3"  # This test case will fail
    ]

    print(func)
    executor = PyExecutor()
    result = executor.std_execute(func, inputs, outputs)
    print(result)
    

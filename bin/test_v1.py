import json
import torch
from utils import write_jsonl
import os
import numpy as np
import argparse
import concurrent.futures
import random
import time
import logging
import copy
import re
logging.getLogger().setLevel(logging.ERROR)
from functools import partial
from request_kimi import *

from models.llama import LlamaInterface

from datetime import datetime
from executors import executor_factory
from utils import read_jsonl, read_jsonl_gz
import hashlib

all_feedback = []

def generate_md5(string):
    return hashlib.md5(string.encode()).hexdigest()

def prepare_planing_prompt(x):
    # def extract_task(p):
    #     return p.split('"""')[1].strip()
    #prompt = extract_task(x['prompt'])
    #prompt = extract_task(x['text'])
    prompt = x['text']
    #tests = '\n'.join(x['visible_tests'])
    tests = '\n'.join(x['test_list'])
    # prompt = f'You are an expert Python programmer, and here is your task: {prompt}\nYour code should pass these tests:\n\n{tests}\nYour response for the task should start with a [PYTHON] tag and end with a [/PYTHON] tag.'
    prompt = f'You are an expert Python programmer, and here is your task: {prompt}\nYour code should pass these tests:\n\n{tests}\nYour code for the task should start with a [PYTHON] tag and end with a [/PYTHON] tag. Write another 3 test cases for the code, your test cases should start with a [TEST] tag and end with a [/TEST] tag.'
    return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n"
    

def prepare_prompts(_iter, dataset, verification_results, test_case_for_tools, execution_feedback, evolve_task):
    prompts = []
    # print("verification_results, ", verification_results)
    # print("test_case_for_tools, ", test_case_for_tools)
    # print("execution_feedback, ", execution_feedback)
    # pooling_size = sum(len(p) for p in test_case_for_tools)
    if _iter == 0:
    # if True:
        for x in dataset:
            prompt = x['text']
            #tests = '\n'.join(x['test_list'][:3])
            tests = '\n'.join(x['test_list'])
            prompt = f'You are an expert Python programmer, and here is your task: {prompt}\nYour code should pass these tests:\n\n{tests}\nYour code for the task should start with a [PYTHON] tag and end with a [/PYTHON] tag. Write another 3 test cases for the code. The test cases should include a variety of tests related to regular testing, boundary testing and stress testing. Test cases are directly executable assert statements, without any comments. And your whole test cases should start with a [TEST] tag and end with a [/TEST] tag.'
            #prompt = f'You are an expert Python programmer, and here is your task: {prompt}\nYour code for the task should start with a [PYTHON] tag and end with a [/PYTHON] tag. Write another 20 assert test cases for the code. The test cases should include a variety of tests related to regular testing, boundary testing and stress testing. Test cases are directly executable assert statements, without any comments. And your whole test cases should start with a [TEST] tag and end with a [/TEST] tag.'
            # prompt = f'You are an expert Python programmer, and here is your task: {prompt}\nYour code for the task should start with a [PYTHON] tag and end with a [/PYTHON] tag. Write another 200 test cases for the code. Test cases are directly executable assert statements, without any comments. And your whole test cases should start with a [TEST] tag and end with a [/TEST] tag.'
            prompts.append(f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n")
    else:
        for i in range(len(verification_results)):
            if i not in evolve_task:
                continue
            text = dataset[i]["text"]
            #tests = '\n'.join(dataset[i]['test_list'][:3])
            tests = '\n'.join(dataset[i]['test_list'])
            # prompt = f'You are an expert Python programmer, and here is your task: {text}\nYour code should pass these tests:\n\n{tests}\nYour code for the task should start with a [PYTHON] tag and end with a [/PYTHON] tag. Write another 200 test cases for the code, your test cases should start with a [TEST] tag and end with a [/TEST] tag.'
            # prompt = "You are an expert Python programmer, and here is your task: {query}\nYour code should pass these tests:\n\n{tests}\nYour code for the task should start with a [PYTHON] tag and end with a [/PYTHON] tag. Update the expanded test cases {new_tests} based on the code, your test cases should start with a [TEST] tag and end with a [/TEST] tag."
            # 这里prompt有效
            # prompt = "You are an expert Python programmer, and here is your task: {query}\nFirst refine your generated test cases {new_tests} and ensure that your test cases meet the testing requirements of the task. Second write your code that pass these tests:\n\n{tests}. \nYour code for the task should start with a [PYTHON] tag and end with a [/PYTHON] tag. Your generated test cases should start with a [TEST] tag and end with a [/TEST] tag."
            prompt = "You are an expert Python programmer, and here is your task: {query}\nYour code should pass these tests:\n\n{tests}\nYour code for the task should start with a [PYTHON] tag and end with a [/PYTHON] tag. Write another 3 test cases for the code. The test cases should include a variety of tests related to regular testing, boundary testing and stress testing. Test cases are directly executable assert statements, without any comments. And your whole test cases should start with a [TEST] tag and end with a [/TEST] tag."
            #prompt = "You are an expert Python programmer, and here is your task: {query}\nYour code for the task should start with a [PYTHON] tag and end with a [/PYTHON] tag. Write another 3 test cases for the code. The test cases should include a variety of tests related to regular testing, boundary testing and stress testing. Test cases are directly executable assert statements, without any comments. And your whole test cases should start with a [TEST] tag and end with a [/TEST] tag."
            # prompt = "You are an expert Python programmer, and here is your task: {query}\nYour previous solution for the task is as follows: {Answer}\nBelow are the generated test cases and the execution results:\n{Execution}\nBased on these, try to first refine your generated test cases to meet the testing requirements of the task. Second refine your solution according to the execution feedback. Make sure your solution is different from the given code snippet in order to explore more robust, concise, and high-performance solutions. The new code for the task should start with a [PYTHON] tag and end with a [/PYTHON] tag. Your generated test cases should start with a [TEST] tag and end with a [/TEST] tag."
            # prompt = "You are an expert Python programmer, and here is your task: {query}\nYour code should pass these tests:\n\n{tests}\nYour code for the task should start with a [PYTHON] tag and end with a [/PYTHON] tag. Update the expanded test cases {new_tests} based on the code, your test cases should start with a [TEST] tag and end with a [/TEST] tag."
            # prompt = "You are an expert Python programmer, and here is your task: {query}\nYou write the code for the task: {code}. There are errors in your code. Please adjust your code logic according to the Task requirements to pass these tests: {tests}. Also, revise your own test cases accordingly based on the code: {new_tests}. Make sure your solution is different from the given code snippet. Output your refined code starts with a [PYTHON] tag and ends with a [/PYTHON] tag. And your test cases should start with a [TEST] tag and end with a [/TEST] tag."
            # prompt = "{\"Task\": \"{query}\", \"Code\": \"{Answer}\", \"Test\": {Execution}}\n\"input\" represents the test case, and \"output\" is the output of the code using that test case. If the output does not match the expected result in the input, determine whether it is a code error or a test case error based on the Task. If it is a test case error, update the test case with the output. If it is a code error, update the code logic according to the output. Make sure your solution is different from the given code snippet. Output your refined code starts with a [PYTHON] tag and ends with a [/PYTHON] tag. And your test cases should start with a [TEST] tag and end with a [/TEST] tag."
            # prompt = "{\"Task\": \"{query}\", \"Code\": \"{Answer}\", \"Test\": {Execution}}\n\"input\" represents the test case, and \"output\" is the output of the code using that test case. If the output does not match the expected result in the input, determine whether it is a code error or a test case error based on the Task. If it is a test case error, update the test case with the output. If it is a code error, update the code logic according to the output. Ensure that your code and test cases have undergone significant modifications. Output your refined code starts with a [PYTHON] tag and ends with a [/PYTHON] tag. And your refined test cases should start with a [TEST] tag and end with a [/TEST] tag."
            # prompt = "{\"Task\": \"{query}\", \"Code\": \"{Answer}\", \"Test\": {Execution}}\n\"input\" represents the test case, and \"output\" is the output of the code using that test case.\n1. Refine the test cases to ensure that the output of each test case aligns with the Task definition. You can refer to the outputs in the Test to make corrections.\n2. Carefully analyze the reasons for the discrepancies between the expected outputs in the test outputs and inputs, and use this to improve the implementation of the code.\n3. Ensure that your code and test cases have undergone significant modifications. Output your refined code starts with a [PYTHON] tag and ends with a [/PYTHON] tag. And your refined test cases should start with a [TEST] tag and end with a [/TEST] tag."
            # prompt = "{\"Task\": \"{query}\", \"Code\": \"{Answer}\", \"Test\": {Execution}}\n\"input\" represents the test case, and \"output\" is the output of the code using that test case.\n1. Refine the test cases to ensure that the output of each test case aligns with the Task definition. You can refer to the outputs in the Test to make corrections.\n2. Carefully analyze the reasons for the discrepancies between the expected outputs in the test outputs and inputs, and use this to improve the implementation of the code.\n3. Ensure that your code and test cases have undergone significant modifications.\n4. Output your refined code starts with a [PYTHON] tag and ends with a [/PYTHON] tag.\n5. Output your refined test cases starts with a [TEST] tag and ends with a [/TEST] tag."
            # prompt = "{\"Task\": \"{query}\", \"Code\": \"{Answer}\", \"Test\": {Execution}}\n\"input\" represents the test case, and \"output\" is the output of the code using that test case.\n1. Refine the test cases to ensure that the output of each test case aligns with the Task definition. You can refer to the outputs in the Test to make corrections.\n2. Compare the expected outputs displayed in the test case \'input\' with the actual outputs in the \'output\', identify the code segments causing the discrepancies, and update the code logic to pass the tests.\n3. Ensure that your code and test cases have undergone significant modifications.\n4. Output your refined code starts with a [PYTHON] tag and ends with a [/PYTHON] tag.\n5. Output your refined test cases starts with a [TEST] tag and ends with a [/TEST] tag.\n6. Ensure that your code have undergone significant modifications."
            # here
            # prompt = "{\"Task\": \"{query}\", \"Code\": \"{Answer}\", \"Test\": {Execution}}\n\"input\" represents the test case, and \"output\" is the output of the code using that test case.\n1. Refine the test cases to ensure that the output of each test case aligns with the Task definition. You can refer to the outputs in the Test to make corrections.\n2. Compare the expected outputs displayed in the test case \'input\' with the actual outputs in the \'output\', identify the code segments causing the discrepancies, and update the code logic to pass the tests.\n3. Ensure that your code and test cases have undergone significant modifications.\n4. Output your refined code starts with a [PYTHON] tag and ends with a [/PYTHON] tag.\n5. Output your refined test cases starts with a [TEST] tag and ends with a [/TEST] tag.\n6. Ensure that your refined code is different from the content in the \'Code\'."
            # here
            # prompt = "{\"Task\": \"{query}\", \"Code\": \"{Answer}\", \"Test\": {Execution}}\n\"input\" represents the test case, and \"output\" is the output of the code using that test case.\n1. Refine the test cases to ensure that the output of each test case aligns with the Task definition. You can refer to the outputs in the Test to make corrections.\n2. Compare the expected outputs displayed in the test case \'input\' with the actual outputs in the \'output\', identify the code segments causing the discrepancies, and update the code logic to pass the tests.\n3. Ensure that your code and test cases have undergone significant modifications.\n4. Output your refined code starts with a [PYTHON] tag and ends with a [/PYTHON] tag.\n5. Output your refined test cases starts with a [TEST] tag and ends with a [/TEST] tag.\n6. Your \'Code\' contains errors, so the improved code must have a different logic from the original \'Code\'."
            # prompt = "{\"Task\": \"{query}\", \"Code\": \"{Answer}\", \"Test\": {Execution}}\n\"input\" represents the test case, and \"output\" is the output of the code using that test case.\n1. Refine the test cases to ensure that the output of each test case aligns with the Task definition. You can refer to the outputs in the Test to make corrections.\n2. Compare the expected outputs displayed in the test case \'input\' with the actual outputs in the \'output\', identify the code segments causing the discrepancies, and update the code logic to pass the tests.\n3. Output your refined test cases starts with a [TEST] tag and ends with a [/TEST] tag.\n4. Output your refined code starts with a [PYTHON] tag and ends with a [/PYTHON] tag.\n5. Ensure that your refined code in [PYTHON] and [/PYTHON] is totally different from \'Code\'."
            # prompt = "{\"Task\": \"{query}\", \"Code\": \"{Answer}\", \"Test\": {Execution}}\n\"input\" represents the test case, and \"output\" is the output of the code using that test case.\n1. Refine the test cases to ensure that the output of each test case aligns with the Task definition. You can refer to the outputs in the Test to make corrections.\n2. Compare the expected outputs displayed in the test case \'input\' with the actual outputs in the \'output\', identify the code segments causing the discrepancies, and update the code logic to pass the tests.\n3. Output your response in the following format: [PYTHON]refined_code[/PYTHON], [TEST]refined_test[/TEST]. Ensure that your refined code in [PYTHON] and [/PYTHON] is totally different from \'Code\'."
            # prompt = "You are an expert Python programmer, and here is your task: {query}\nHere is your code for the task: {Answer}. Please update your code to pass these tests:\n\n{Execution}\nYour code for the task should start with a [PYTHON] tag and end with a [/PYTHON] tag."
            answer = verification_results[i]
            test_cases = test_case_for_tools[i]
            feedback = execution_feedback[i]
            new_test_cases = []
            test_feedback = []
            # all_feedback = []
            global all_feedback
            # print("DEBUG prepare_prompts test_cases=%s feedback=%s" % (len(test_cases), len(feedback)))
            # print("DEBUG prepare_prompts test_cases=%s feedback=%s" % (test_cases, feedback))
            if len(test_cases) > len(feedback):
                print("WARNING: Unmatched Test and Feedback")
                test_cases = test_cases[:len(feedback)]
            for j in range(len(test_cases)):
                if feedback[j] == "test passed":
                    continue
                # if feedback[j] in all_feedback:
                #     continue
                new_test_cases.append(test_cases[j])
                item = {}
                item["input"] = test_cases[j]
                item["output"] = feedback[j]
                test_feedback.append(item)
            print("DEBUG Each task_%s_prepare_prompts test_cases=%s feedback=%s" % (i, new_test_cases, test_feedback))
            if len(test_feedback) > 0:
                # prompt = "You are an expert Python programmer, and here is your task: {query}\nYour previous solution for the task is as follows: {Answer}. \nHere is your test case: {new_tests}, it is wrong with the output {Wrong}.\nRefine your code and make it output {Correct}.\nYour code for the task should start with a [PYTHON] tag and end with a [/PYTHON] tag."
                # prompt = "You are an expert Python programmer, and here is your task: {query}\nYour previous solution for the task is as follows: {Answer}. \nHere is your test case: {new_tests}, it is wrong with the output {Wrong}.\nRefine your code and make it correct.\nYour code should also pass these tests:\n\n{tests}\n\nYour code for the task should start with a [PYTHON] tag and end with a [/PYTHON] tag."
                # input_sample = random.sample(test_feedback, 1)[0]
                # if input_sample["output"] in all_feedback:
                #     continue

                for input_sample in test_feedback:
                    if input_sample["output"] not in all_feedback[i]:
                        # print("Debug Each task_%s_new_test sample=%s" % (i, json.dumps(input_sample, ensure_ascii=False)))
                        prompt = "You are an expert Python programmer, and here is your task: {query}\nYour previous solution for the task is as follows: {Answer}. \nHere is your test case: \"{new_tests}\", it is wrong with the output \"{Wrong}\".\nRefine your code and make it correct.\nYour code should also pass these tests:\n\n{tests}\n\nYour code for the task should start with a [PYTHON] tag and end with a [/PYTHON] tag. Write another 3 test cases for the code. The test cases should include a variety of tests related to regular testing, boundary testing and stress testing. Test cases are directly executable assert statements, without any comments. And your whole test cases should start with a [TEST] tag and end with a [/TEST] tag."
                        #prompt = "You are an expert Python programmer, and here is your task: {query}\nYour previous solution for the task is as follows: {Answer}. \nHere is your test case: \"{new_tests}\", it is wrong with the output \"{Wrong}\".\nRefine your code and make it correct.\n\nYour code for the task should start with a [PYTHON] tag and end with a [/PYTHON] tag. Write another 3 test cases for the code. The test cases should include a variety of tests related to regular testing, boundary testing and stress testing. Test cases are directly executable assert statements, without any comments. And your whole test cases should start with a [TEST] tag and end with a [/TEST] tag."
                        prompt = prompt.replace("{new_tests}", input_sample["input"])
                        prompt = prompt.replace("{Answer}", answer)
                        prompt = prompt.replace("{Wrong}", str(input_sample["output"]))
                        all_feedback[i].append(input_sample["output"])
                        break
                # prompt = prompt.replace("{Correct}", input_sample["input"].split("==")[1])
            # prompt = prompt.replace("{Execution}", json.dumps(test_feedback))
            prompt = prompt.replace("{query}", text)
            prompt = prompt.replace("{tests}", tests)
            # prompts.append(f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n")
            # prompts.append(prompt)
            # Qwen遵循指令能力较差，需要给定合适的instruction
            prompts.append(f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n")
    return prompts


def prepare_aggregate_prompt(dataset, verification_outs):
    prompts = []
    for i in range(len(verification_outs[0])):
        text = dataset[i]["text"]
        responses = [verification_outs[k][i] for k in range(len(verification_outs))]
        prompt = "Task: {Query}\n{Responses}\nPlease refer to the above codes for implementation. You can either select the correct code from them or improve upon the provided codes. The final code you provide should start with [PYTHON] and end with [/PYTHON], without any additional information."
        prompt = prompt.replace("{Query}", text)
        prompt = prompt.replace("{Responses}", '\n'.join([f'[Solution {idx+1}] {item}' for idx, item in enumerate(responses)]))
        prompts.append(prompt)
    return prompts


def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')

def kimis(prompts, temperature):
    batch_results = {}
    results = []
    batches = []
    for i in range(len(prompts)):
        prompt = prompts[i]
        request = {}
        request["messages"] = [{"role": "user", "content": prompt}]
        request["idx"] = i
        batches.append(request)
        if len(batches) == 60:
            answer = batch_request(batches, temperature, 60)
            for j in range(len(answer)):
                batch_results.setdefault(answer[j]["idx"], answer[j]["infer_answer"])
            batches = []
    if len(batches) > 0:
        answer = batch_request(batches, temperature, 60)
        for i in range(len(answer)):
            batch_results.setdefault(answer[i]["idx"], answer[i]["infer_answer"])
    for i in range(len(prompts)):
        results.append(batch_results[i])
        print("prompts %s md5 of prompt: %s text: %s md5 of answer: %s text: %s" % (i, generate_md5(prompts[i]), json.dumps({"prompt": prompts[i]}, ensure_ascii=False), generate_md5(batch_results[i]), json.dumps({"answer": batch_results[i]}, ensure_ascii=False)))
    return results

def EvaluateCertainCode(ans, tests_samples):
    acc = 0
    passing_flags = []
    exe = executor_factory('py', is_leet=False)
    # tests_samples = random.sample(tests_samples, min(50, len(tests_samples)))
    feedback_list = []
    # print("EvaluateCertainCode ans=%s tests=%s tests_samples=%s" % (ans, len(tests_samples), tests_samples))
    tests_samples = [test for test in tests_samples if test.find("assert") != -1]
    # tests_samples = [test.replace("\n", "\\n") for test in tests_samples if test.find("assert") != -1]
    if len(ans) == 0 or len(tests_samples) == 0:
        return 0, []
    is_passing, feedback_list, state = exe.execute(ans, tests_samples, timeout=0.1)
    print("EvaluateCertainCode is_passing=%s feedback_list=%s state=%s" % (is_passing, feedback_list, state))
    if len(state) == 0:
        return 0, []
    acc = sum(state)/len(state)
    # for test in tests_samples:
    #     if test.find("assert") != -1:
    #         # print("EvaluateCertainCode ans=%s test=%s" % (ans, [test]))
    #         is_passing, feedback, _ = exe.execute(ans, [test], timeout=0.1)
    #         # print("EvaluateCertainCode ans=%s test=%s is_passing=%s feedback=%s" % (ans, test, is_passing, feedback))
    #         passing_flags.append(is_passing)
    #         feedback_list.append(feedback)
    #         acc += int(is_passing)
    
    # return acc/len(tests_samples), feedback_list
    return acc, feedback_list

class TimeoutException(Exception):
    pass

def EvaluateCode(verification_results, dataset):
    acc = 0
    passing_flags = []
    for i, ans in enumerate(verification_results):
        # and = "def find_adverbs(sentence):\n    adverbs = re.finditer(r'\b\w+ly\b', sentence)\n    for adverb in adverbs:\n        yield f'{adverb.start()}-{adverb.end()}: {adverb.group()}"
        exe = executor_factory('py', is_leet=False)
        tests_i = dataset[i]['test_list']
        # tests_i = [tests_i[j].replace("\n", "\\n") for j in range(len(tests_i))]
        # tests_i = [dataset[i]['execution_test']]
        print("DEBUG EvaluateCode ans=%s tests_i=%s" % (ans, len(tests_i)))
        if len(ans) == 0 or len(tests_i) == 0:
            acc += 0
            passing_flags.append(False)
            continue
        print("DEBUG TimeoutException of ans=%s tests=%s" % (ans, tests_i))
        # try:
        is_passing, feedback, _ = exe.execute(ans, tests_i, timeout=5)
        print("Evaluate Code is_passing=%s feedback=%s ans=%s tests_i=%s" % (is_passing, feedback, ans, tests_i))
        # except TimeoutException:
            # is_passing = False
        
        passing_flags.append(is_passing)
        dataset[i]['is_solved'] = is_passing
        dataset[i]['implementation'] = ans
        acc += int(is_passing)

        # write_jsonl(outfilename, [dataset[i]], append=True)
    # print("EvaluateCode passing_flags=%s" % passing_flags)
    return acc/len(verification_results), passing_flags



def extract_task(response):
    response_case = []
    for i in range(len(response)):
        # print("extract_task before=%s" % response[i])
        ans = response[i]
        pattern = r'\[PYTHON\](.*?)\[/PYTHON\]'
        matches = re.findall(pattern, ans, re.DOTALL)
        def_functions = [match for match in matches if 'def' in match]
        test_pattern = r'\[TEST\](.*?)\[/TEST\]'
        matches_cases = re.findall(test_pattern, ans, re.DOTALL)
        test_cases = [match for match in matches_cases if 'assert ' in match]
        # print("test cases")
        # print(test_cases)
        if len(def_functions) > 0:
            ans = def_functions[0]
        if ans.find("```python") != -1:
            ans = ans.split("```python")[1]
        if ans.find("```") != -1:
            ans = ans.split("```")[0]
        piece = ans.split("\n")
        for j in range(len(piece)-1, 0, -1):
            if piece[j].find(" return ") != -1:
                ans = "\n".join(piece[:j+1])
                break
        piece = ans.split("\n")
        for j in range(0, len(piece)):
            if piece[j].find("def ") != -1 or piece[j].find("import ") != -1 or piece[j].find("from ") != -1:
                ans = "\n".join(piece[j:])
                break
        # print("extract_task after=%s" % ans)
        response[i] = ans.strip()
        cases = []
        if len(test_cases) > 0:
            # print("test_cases[0]")
            # print(test_cases[0])
            for test_case in test_cases:
                case = test_case.split("\n")
                case = [c for c in case if len(c) > 0 and c.find("assert ") != -1]
                cases.extend(case)
        # print("extract_task ans=%s cases=%s" % (ans, cases))
        response_case.append(cases)
        # print("extract_task response_case=%s" % response_case)
    return response, response_case

def extract_verification_task(response, answer1, answer2):
    for i in range(len(response)):
        print("extract_task before=%s" % response[i])
        ans = response[i]
        pattern = r'\[PYTHON\](.*?)\[/PYTHON\]'
        matches = re.findall(pattern, ans, re.DOTALL)
        def_functions = [match for match in matches]
        if len(def_functions) > 0:
            ans = def_functions[0]
            if ans == "2":
                ans = answer2[i]
            else:
                ans = answer1[i]
        else:
            ans = answer1[i]
        print("extract_task after=%s" % ans)
        response[i] = ans
    return response


def update_test_case(test_case, feedback):
    for i in range(len(test_case)):
        print(feedback[i])
        case = test_case[i]
        if feedback[i] != "test passed":
            test_case[i] = case.split("==")[0] + "== '" + feedback[i] + "'"
    return test_case


def run(dataset, gpts, evaluate=True, outfilename=None, do_sample=True):

    planing_budget = 5
    verification_budget = 5
    # execution_feedback = []
    verification_results = []
    for _ in range(len(dataset)):
        verification_results.append("")
    global all_feedback
    all_feedback = [[] for _ in range(len(dataset))]
    previous_max_acc = [[0] for _ in range(len(dataset))]
    max_acc_idx = [-1] * len(dataset)
    test_pooling = [[] for _ in range(len(dataset))]
    execution_feedback = [[] for _ in range(len(dataset))]
    stop_threshold = 0.8
    temperature = 0.5
    # temperature = 0.3
    # print("DEBUG Dataset: %s" % json.dumps(dataset, ensure_ascii=False))
    time0 = int(time.time()*1000)
    # evolve_task = [i for i, acc in enumerate(previous_max_acc) if acc < stop_threshold]
    # evolve_dataset = [dataset[i] for i in evolve_task]
    acc = 0
    for _iter4veri in range(verification_budget):
        print("previous_max_acc, ", previous_max_acc)
        print("verification_results, ", verification_results)
        print("test_pooling, ", test_pooling)
        print("execution_feedback, ", execution_feedback)
        evolve_task = [i for i, acc_list in enumerate(previous_max_acc) if acc_list[-1] < stop_threshold]
        evolve_dataset = [dataset[i] for i in evolve_task]
        # prompts->outs/test_case_for_tools->verification_results/feedback->prompts...
        if len(evolve_task) == 0:
            break
        # prompts/test_case_for_tools 长度和evolve_task/evolve_dataset一致
        # 准备prompt，生成code/tests
        prompts = prepare_prompts(_iter4veri, dataset, verification_results, test_pooling, execution_feedback, evolve_task)
        info = {}
        info["serial_iteration"] = _iter4veri
        info["evolve_task"] = evolve_task
        info["eovlve_size"] = len(evolve_task)
        info["prompts"] = prompts
        print("DEBUG Evolve Task: %s" % json.dumps(info, ensure_ascii=False))
        test_case_for_tools = [[] for _ in range(len(prompts))]
        outs = []
        outs_response = []
        gt_feedback = []
        # 横向探索
        for _iter in range(planing_budget):
            time1 = int(time.time()*1000)
            # 横向迭代，多次生成code/tests
            planing = kimis(prompts, temperature=temperature)
            # planing = gpts(prompts, temperature=temperature)
            response = copy.deepcopy(planing)
            time2 = int(time.time()*1000)
            print("DEBUG TIME of Single Planing: %s" % (time2-time1))
            for i in range(len(evolve_task)):
                info = {}
                info["prompt"] = prompts[i]
                print("DEBUG Each task_%s_prompt: %s" % (evolve_task[i], info))
                info = {}
                info["response"] = response[i]
                print("DEBUG Each task_%s_response: %s" % (evolve_task[i], info))
            # 抽取code/tests
            extract_planing, tests = extract_task(planing)
            for i in range(len(evolve_task)):
                info = {}
                info["planing"] = extract_planing[i]
                print("DEBUG Each task_%s_planing: %s" % (evolve_task[i], info))
                
            # tests后处理：去重/去掉gt_tests
            def check_duplicate(_input, _candidates):
                _input = _input.split("==")[0].strip()
                for candidate in _candidates:
                    if candidate.find(_input) != -1:
                        print("check_duplicate _input=%s duplicate=%s" % (_input, candidate))
                        return True
                return False

            for i in range(len(tests)):
                if len(tests[i]) > 0:
                    # # ignore in livecode/code_contest
                    # task_name = "_".join(evolve_dataset[i]["task_id"].split("_")[2:])
                    # # 去除和task_name不match的case
                    # tests[i] = [t.strip() for t in tests[i] if t.find(task_name) != -1]
                    # # 去除带有注释的test
                    # tests[i] = [t.strip() for t in tests[i] if t.find("#") == -1]
                    # # 去除和gt_test相同的test，确保不作弊
                    
                    # #gt_test = [test.replace("candidate", task_name) for test in evolve_dataset[i]["test_list"]]
                    # # print("Debug Cheating gt_test:%s generate_test:%s" % (gt_test, tests[i]))
                    # print("Debug Cheating remove task %s before: %s" % (task_name, len(tests[i])))
                    # tests[i] = [t for t in tests[i] if not check_duplicate(t, gt_test)]
                    # print("Debug Cheating remove task %s after: %s " % (task_name, len(tests[i])))
                    # 去重
                    test_case_for_tools[i].extend(tests[i])
                    test_case_for_tools[i] = list(set(test_case_for_tools[i]))
                info = {}
                info["tests"] = test_case_for_tools[i]
                print("DEBUG Each task_%s_tests: %s len of tests=%s" % (evolve_task[i], info, len(test_case_for_tools[i])))
            # 评估code质量，debug用，evaluate时可以省略
            acc, feedback = EvaluateCode(extract_planing, evolve_dataset)
            time3 = int(time.time()*1000)
            print("DEBUG TIME of Evaluate All Planing: %s" % (time3-time2))
            # print("DEBUG GT feedback: %s" % (feedback))
            info = {}
            info["stage"] = "Planing"
            info["serial_iteration"] = _iter4veri
            info["parallel_iteration"] = _iter
            info["feedback"] = {i: v for i, v in enumerate(feedback)}
            info["gt_acc"] = acc
            gt_feedback.append(feedback)
            print("DEBUG Eval Planing:%s" % (json.dumps(info, ensure_ascii=False)))
            # 记录plan结果
            outs.append(copy.deepcopy(extract_planing))
            outs_response.append(copy.deepcopy(response))
        time00 = int(time.time()*1000)
        print("DEBUG TIME of ALL Planing: %s" % (time00-time0))
        # 纵向聚合
        # 更新测试集
        # pooling_size = sum(len(p) for p in test_pooling)
        if _iter4veri == 0:
            test_pooling = copy.deepcopy(test_case_for_tools)
        else:
            for i in range(len(evolve_task)):
                if len(test_pooling[evolve_task[i]]) < 100:
                    test_pooling[evolve_task[i]].extend(test_case_for_tools[i])
                    test_pooling[evolve_task[i]] = list(set(test_pooling[evolve_task[i]]))
                # test_pooling[evolve_task[i]] = random.sample(test_pooling[evolve_task[i]], min(50, len(test_pooling[evolve_task[i]])))
        # 聚合答案
        for i in range(len(outs[0])):
            print("DEBUG TIME of Case %s" % evolve_dataset[i]["task_id"])
            max_acc = 0
            max_idx = -1
            max_ans = ""
            max_feedback = []
            # 记录表现最好的code
            for j in range(len(outs)):
                out = outs[j][i]
                # tests_samples = random.sample(test_case_for_tools[i], min(50, len(test_case_for_tools[i])))
                # 正确的代码在错误的历史case上不占优势，应该直接用当前test_case测试当前代码 acc, feedback = EvaluateCertainCode(out, test_pooling[evolve_task[i]])
                time11 = int(time.time()*1000)
                print("DEBUG TIME of EvaluateCertainCode pooling_size=%s" % (len(test_pooling[evolve_task[i]])))
                acc, feedback = EvaluateCertainCode(out, test_pooling[evolve_task[i]])
                time12 = int(time.time()*1000)
                print("DEBUG TIME of EvaluateCertainCode with test_pooling: %s" % (time12-time11))
                # acc/feedback记录当前solution的表现
                if acc > max_acc or len(max_ans) == 0:
                    max_acc = acc
                    max_ans = out
                    max_feedback = feedback
                    max_idx = j
            # 获得此次请求最大acc的solution，将其与历史值相比
            # max_acc比历史acc大，则更新solution
            if max_acc > previous_max_acc[evolve_task[i]][-1] or len(verification_results[evolve_task[i]]) == 0:
                verification_results[evolve_task[i]] = max_ans
                previous_max_acc[evolve_task[i]][-1] = max_acc
                max_acc_idx[evolve_task[i]] = max_idx
                execution_feedback[evolve_task[i]] = copy.deepcopy(max_feedback)
                for j in range(len(max_feedback)):
                    if isinstance(max_feedback[j], str) and max_feedback[j] == "test passed":
                        continue
                    else:
                        print("failed tests input: %s \noutput: %s" % (test_pooling[evolve_task[i]][j], max_feedback[j]))
                # 记录trajectory
                info = {}
                info["task_id"] = evolve_task[i]
                info["task"] = dataset[evolve_task[i]]
                info["prompts"] = prompts[i]
                info["previous_max_acc"] = previous_max_acc[evolve_task[i]]
                info["verification_results"] = verification_results[evolve_task[i]]
                info["planing"] = [outs[j][i] for j in range(len(outs))]
                info["response"] = [outs_response[j][i] for j in range(len(outs_response))]
                info["max_idx"] = max_acc_idx[evolve_task[i]]
                info["gt_feedback"] = [gt_feedback[j][i] for j in range(len(gt_feedback))]
                print("TRAJS_LOG: %s" % json.dumps(info, ensure_ascii=False))
            else:
                _, update_feedback = EvaluateCertainCode(verification_results[evolve_task[i]], test_pooling[evolve_task[i]])
                execution_feedback[evolve_task[i]] = copy.deepcopy(update_feedback)
        # step3: evaluation
        time000 = int(time.time()*1000)
        print("DEBUG TIME of ALL Verification: %s" % (time000-time00))
        if evaluate:
            acc, feedback = EvaluateCode(verification_results, dataset)
            info = {}
            info["stage"] = "Aggregation"
            info["serial_iteration"] = _iter4veri
            info["parallel_iteration"] = _iter
            info["feedback"] = {i: v for i, v in enumerate(feedback)}
            info["gt_acc"] = acc
            print("DEBUG Aggregation:%s" % (json.dumps(info, ensure_ascii=False)))
            # print("Step:%s Iter=%s Acc=%s" % ("Verification", i, acc))
            for i in range(len(evolve_task)):
                print("DEBUG Each task_%s_max_acc: %s plan_feedback=%s gt_feedback=%s test_len=%s" % (evolve_task[i],  previous_max_acc[evolve_task[i]], [gt_feedback[j][i] for j in range(len(gt_feedback))], feedback[evolve_task[i]], len(test_pooling[evolve_task[i]])))
        for i in range(len(previous_max_acc)):
            previous_max_acc[i].append(previous_max_acc[i][-1])
        time0000 = int(time.time()*1000)
        print("DEBUG TIME of ALL Aggregation: %s" % (time0000-time000))

    return acc


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, default='llama')
    args.add_argument('--temperature', type=float, default=0.3)

    args.add_argument('--task', type=str, default='mbpp-clean-test')
    args.add_argument('--task_split', type=str, default='train')
    args.add_argument('--task_start_index', type=int, default=0)
    args.add_argument('--task_end_index', type=int, default=100)

    args.add_argument('--evaluate', action='store_false')
    args.add_argument('--add_lora', action='store_true')
    args.add_argument('--random', action='store_true')

    args.add_argument('--modelpath', type=str, default='/mnt/moonfs/zhangxiaoqing-m2/models/CodeQwen1.5-7B-Chat/')
    args.add_argument('--peftpath', type=str, default='')
    args.add_argument('--do_sample', action='store_false')
    args.add_argument('--seed', type=int, default=-1)

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    #task = get_task(args.task, args.task_split)
    
    modelname = args.backend
    pathname = args.peftpath.replace('/', '_') if args.add_lora else args.modelpath.replace('/', '_')
    modelname += f"_{pathname}"
    time_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    outfilename = f"trajs/{args.task}__{modelname}_{args.temperature}_{time_str}.jsonl"
    print(outfilename)
    
    if args.seed >= 0:
        seed = args.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # For some operations in Python which are hash-based (like dictionaries), set:
        os.environ["PYTHONHASHSEED"] = str(seed)


    
    # dataset = read_jsonl(f'benchmarks/mbpp-clean-test-v1.part1.jsonl')
    dataset = read_jsonl(f'benchmarks/HumanEval.format.jsonl')
    # dataset = read_jsonl(f'../mbpp/benchmarks/mbppplus.20241213/mbpp.plus.20241213.shuf.part3.jsonl')
    # dataset = read_jsonl(f'../mbpp/benchmarks/mbppet/mbpp-et-part0.jsonl')
    # dataset = [dataset[1], dataset[7], dataset[8], dataset[12], dataset[17], dataset[18], dataset[23], dataset[30], dataset[34], dataset[52], dataset[66], dataset[70]]
    # dataset = [dataset[10], dataset[11], dataset[14], dataset[17], dataset[22], dataset[28], dataset[33], dataset[42], dataset[47], dataset[53], dataset[57]]
    # dataset = dataset[0:2]
    # dataset = dataset[10:15]
    #dataset = read_jsonl(f'/mnt/moonfs/zhangxiaoqing-m2/ThinkCoder/data/mbpp/mbpp.train.head.w.name.jsonl')
    #dataset = read_jsonl(f'/mnt/moonfs/zhangxiaoqing-m2/ThinkCoder/data/mbpp/mbpp.validation.w.name.jsonl')
    # dataset = read_jsonl(f'/mnt/moonfs/zhangxiaoqing-m2/ThinkCoder/data/mbpp/mbpp.test.jsonl')
    # dataset = read_jsonl(f'../data/human-eval/data/HumanEval.format.jsonl')
    #dataset = read_jsonl(f'../data/human-eval/data/humaneval.plus.test.format.jsonl')
    # dataset = read_jsonl(f'../data/human-eval/data/humaneval.plus.test.format.20241213.jsonl')
    # code_contest
    #dataset = read_jsonl(f'../data/code_contest/format_test_code_contest.jsonl')
    #dataset = read_jsonl(f'../data/livecode/format_test5.jsonl')
    # livecode
    # dataset = read_jsonl(f'../data/livecode/format_140.jsonl')
    #dataset = read_jsonl(f'benchmarks/mbpp/mbpp-clean-test-v1.part1.jsonl')
    # dataset = dataset[0:1]

    print(args.modelpath, args.peftpath, args.add_lora)
    llama = LlamaInterface(args.modelpath, args.peftpath, args.add_lora)
    model = llama.generate_responses_from_llama

    # acc = 0
    # for i in range(len(dataset)):
    #     acc += run([dataset[i]], model, outfilename=outfilename, evaluate=args.evaluate, \
    #                 do_sample=args.do_sample)
    #     print("Process idx=%s acc=%s" % (i, acc/(i+1)))

    run(dataset, model, outfilename=outfilename, evaluate=args.evaluate, \
                    do_sample=args.do_sample)

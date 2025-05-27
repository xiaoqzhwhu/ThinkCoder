#coding=utf-8
import sys
import openai
import time
import json
import os
from retrying import retry
import threading
import queue


# # kimi config
model_name = "set your model_name"
openai.api_type = "open_ai"
openai.api_base = "https://%s.app.msh.team/v1" % model_name
openai.api_version=""
openai.api_key = "set your api_key"


if True:
    # openai config
    openai.api_type = "open_ai"
    openai.api_base = "set your api_base"
    openai.api_version = ""
    openai.api_key = "set your api_key"
    model_name = "set your model_name"

@retry(stop_max_attempt_number=3, wait_fixed=1000)
# def get_response(data, result_queue, model=model_name):
def get_response(data, result_queue, temperature, model=model_name):
    response = ""
    total_len = 0
    input_messages = []
    input_messages.append(data["messages"][0])
    # for message in data["messages"][1:2]:
    #     input_messages.append(message)
    # print("input_messages")
    # print(input_messages)
    try:
        result = openai.ChatCompletion.create(model=model, messages=input_messages, temperature=temperature)
        response = result['choices'][0]['message']['content']
    except:
        try:
            result = openai.ChatCompletion.create(model=model, messages=input_messages, temperature=temperature)
            response = result['choices'][0]['message']['content']
        except Exception as e:
            print("get_response error=%s" % e)
            response = ""
    data["infer_answer"] = response
    result_queue.put(data)
    return data

@retry(stop_max_attempt_number=3, wait_fixed=1000)
def get_embedding(data, result_queue, model="text-embedding-ada-002"):
    data["text"] = data["text"].replace("\n", " ")
    data["emb"] = openai.embeddings.create(input = [data["text"]], model=model)['data'][0]['embedding']
    result_queue.put(data)
    return data


def batch_request(data_list, temperature, batch_size):
    batch_response = []
    result_queue = queue.Queue()
    for i in range(int(len(data_list)/batch_size)+1):
        # 创建并启动100个线程
        threads = []
        for j in range(i*batch_size, (i+1)*batch_size):
            if j < len(data_list):
                data = data_list[j]
                thread = threading.Thread(target=get_response, args=(data,result_queue,temperature,))
                threads.append(thread)
                thread.start()
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        # # 从队列中获取结果
        while not result_queue.empty():
            result = result_queue.get()
            batch_response.append(result)
            # decoded_result = result.decode('utf-8') if isinstance(result, bytes) else result
            # batch_response.append(json.loads(decoded_result))
            # print(json.dumps(result, ensure_ascii=False))
            # batch_response.append(result)
    return batch_response


if __name__ == "__main__":
    filename = sys.argv[1]
    data_list = []
    for line in open(filename, "r", encoding="utf-8"):
        line = line.strip("\n")
        # data = json.loads(line)
        # data_list.append(data)
        data = {}
        data["text"] = line
        data_list.append(data)

    batch_request(data_list, temperature, batch_size)


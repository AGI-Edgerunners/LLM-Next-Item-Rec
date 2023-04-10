# !pip install openai

import time
import numpy as np
import json
import openai
import random 
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--length_limit', type=int, default=8, help='')
parser.add_argument('--num_cand', type=int, default=19, help='')
parser.add_argument('--random_seed', type=int, default=2023, help='')
parser.add_argument('--api_key', type=str, default="sk-", help="")

args = parser.parse_args()

rseed = args.random_seed
random.seed(rseed)

def read_json(file):
    with open(file) as f:
        return json.load(f)

def write_json(data, file):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

data_ml_100k = read_json("./ml_100k.json")

# print (data_ml_100k[0][0])
# print (data_ml_100k[0][1])
# print (len(data_ml_100k))


open_ai_keys = [args.api_key]
open_ai_keys_index = 0
openai.api_key = open_ai_keys[open_ai_keys_index]


u_item_dict = {}
u_item_p = 0
for elem in data_ml_100k:
    seq_list = elem[0].split(' | ')
    for movie in seq_list:
        if movie not in u_item_dict:
            u_item_dict[movie] = u_item_p
            u_item_p +=1
print (len(u_item_dict))
u_item_len = len(u_item_dict)

user_list = []
for i, elem in  enumerate(data_ml_100k):
    item_hot_list = [0 for ii in range(u_item_len)]
    seq_list = elem[0].split(' | ')
    for movie in seq_list:
        item_pos = u_item_dict[movie]
        item_hot_list[item_pos] = 1
    user_list.append(item_hot_list)
user_matrix = np.array(user_list)
user_matrix_sim = np.dot(user_matrix, user_matrix.transpose())


pop_dict = {}
for elem in data_ml_100k:
    # elem = data_ml_100k[i]
    seq_list = elem[0].split(' | ')
    for movie in seq_list:
        if movie not in pop_dict:
              pop_dict[movie] = 0
        pop_dict[movie] += 1
        
        
        
i_item_dict = {}
i_item_id_list = []
i_item_user_dict = {}
i_item_p = 0
for i, elem in  enumerate(data_ml_100k):
    seq_list = elem[0].split(' | ')
    for movie in seq_list:
        if movie not in i_item_user_dict:
            item_hot_list = [0. for ii in range(len(data_ml_100k))]
            i_item_user_dict[movie] = item_hot_list
            i_item_dict[movie] = i_item_p
            i_item_id_list.append(movie)
            i_item_p+=1
#         item_pos = item_dict[movie]
        i_item_user_dict[movie][i] += 1
#     user_list.append(item_hot_list)
i_item_s_list = []
for item in i_item_id_list:
    i_item_s_list.append(i_item_user_dict[item])
#     print (sum(item_user_dict[item]))
item_matrix = np.array(i_item_s_list)
item_matrix_sim = np.dot(item_matrix, item_matrix.transpose())

id_list =list(range(0,len(data_ml_100k)))



### user filtering
def sort_uf_items(target_seq, us, num_u, num_i):

    candidate_movies_dict = {} 
    sorted_us = sorted(list(enumerate(us)), key=lambda x: x[-1], reverse=True)[:num_u]
    dvd = sum([e[-1] for e in sorted_us])
    for us_i, us_v in sorted_us:
        us_w = us_v * 1.0/dvd
#         print (us_i)
        us_elem = data_ml_100k[us_i]
#         print (us_elem[0])
#         assert 1==0
        us_seq_list = us_elem[0].split(' | ')#+[us_elem[1]]

        for us_m in us_seq_list:
#             print (f"{us_m} not in {target_seq}, {us_m not in target_seq}")
#             break

            if us_m not in target_seq:
                if us_m not in candidate_movies_dict:
                    candidate_movies_dict[us_m] = 0.
                candidate_movies_dict[us_m]+=us_w
                
#         assert 1==0
                
    candidate_pairs = list(sorted(candidate_movies_dict.items(), key=lambda x:x[-1], reverse=True))
#     print (candidate_pairs)
    candidate_items = [e[0] for e in candidate_pairs][:num_i]
    return candidate_items


### item filtering
def soft_if_items(target_seq, num_i, total_i, item_matrix_sim, item_dict):
    candidate_movies_dict = {} 
    for movie in target_seq:
#         print('ttt:',movie)
        sorted_is = sorted(list(enumerate(item_matrix_sim[item_dict[movie]])), key=lambda x: x[-1], reverse=True)[:num_i]
        for is_i, is_v in sorted_is:
            s_item = i_item_id_list[is_i]
            
            if s_item not in target_seq:
                if s_item not in candidate_movies_dict:
                    candidate_movies_dict[s_item] = 0.
                candidate_movies_dict[s_item] += is_v
#             print (item_id_list[is_i], candidate_movies_dict)
    candidate_pairs = list(sorted(candidate_movies_dict.items(), key=lambda x:x[-1], reverse=True))
#     print (candidate_pairs)
    candidate_items = [e[0] for e in candidate_pairs][:total_i]
#     print (candidate_items)
    return candidate_items



'''
In order to economize, our initial step is to identify user sequences that exhibit a high probability of obtaining accurate predictions from GPT-3.5 based on their respective candidates. 
Subsequently, we proceed to utilize the GPT-3.5 API to generate predictions for these promising user sequences.
'''
results_data_15 = []
length_limit = args.length_limit
num_u= 12
total_i = args.num_cand
count = 0
total = 0
cand_ids = []
for i in id_list[:1000]:
    elem = data_ml_100k[i]
    seq_list = elem[0].split(' | ')
    
    candidate_items = sort_uf_items(seq_list, user_matrix_sim[i], num_u=num_u, num_i=total_i)
    
#     print (elem[-1], '-',seq_list[-1])

    if elem[-1] in candidate_items:
#         print ('HIT: 1')
        count += 1
        cand_ids.append(i)
    else:
        pass
#         print ('HIT: 0')
    total +=1
print (f'count/total:{count}/{total}={count*1.0/total}')
print ('-----------------\n')


temp_1 = """
Candidate Set (candidate movies): {}.
The movies I have watched (watched movies): {}.
Step 1: What features are most important to me when selecting movies (Summarize my preferences briefly)? 
Answer: 
"""

temp_2 = """
Candidate Set (candidate movies): {}.
The movies I have watched (watched movies): {}.
Step 1: What features are most important to me when selecting movies (Summarize my preferences briefly)? 
Answer: {}.
Step 2: Selecting the most featured movies from the watched movies according to my preferences (Format: [no. a watched movie.]). 
Answer: 
"""

temp_3 = """
Candidate Set (candidate movies): {}.
The movies I have watched (watched movies): {}.
Step 1: What features are most important to me when selecting movies (Summarize my preferences briefly)? 
Answer: {}.
Step 2: Selecting the most featured movies (at most 5 movies) from the watched movies according to my preferences in descending order (Format: [no. a watched movie.]). 
Answer: {}.
Step 3: Can you recommend 10 movies from the Candidate Set similar to the selected movies I've watched (Format: [no. a watched movie - a candidate movie])?.
Answer: 
"""

count = 0
total = 0
results_data = []
for i in cand_ids[:]:#[:10] + cand_ids[49:57] + cand_ids[75:81]:
    elem = data_ml_100k[i]
    seq_list = elem[0].split(' | ')[::-1]
    
    candidate_items = sort_uf_items(seq_list, user_matrix_sim[i], num_u=num_u, num_i=total_i)
    random.shuffle(candidate_items)

    input_1 = temp_1.format(', '.join(candidate_items), ', '.join(seq_list[-length_limit:]))

    try_nums = 5
    kk_flag = 1
    while try_nums:
        try:
            response = openai.Completion.create(
                      engine="text-davinci-003",
                      prompt=input_1,
                      max_tokens=512,
                      temperature=0,
                      top_p=1,
                      frequency_penalty=0,
                      presence_penalty=0,
                      n = 1,
                  )
            try_nums = 0
            kk_flag = 1
        except Exception as e:
            if 'exceeded your current quota' in str(e):

                # open_ai_keys_index +=1
                openai.api_key = open_ai_keys[open_ai_keys_index]
            time.sleep(1) 
            try_nums = try_nums-1
            kk_flag = 0

    if kk_flag == 0:
        time.sleep(5) 
        response = openai.Completion.create(
                      engine="text-davinci-003",
                      prompt=input_1,
                      max_tokens=256,
                      temperature=0,
                      top_p=1,
                      frequency_penalty=0,
                      presence_penalty=0,
                      n = 1,
                  )

    predictions_1 = response["choices"][0]['text']
    
    
    input_2 = temp_2.format(', '.join(candidate_items), ', '.join(seq_list[-length_limit:]), predictions_1)

    try_nums = 5
    kk_flag = 1
    while try_nums:
        try:
            response = openai.Completion.create(
                      engine="text-davinci-003",
                      prompt=input_2,
                      max_tokens=512,
                      temperature=0,
                      top_p=1,
                      frequency_penalty=0,
                      presence_penalty=0,
                      n = 1,
                  )
            try_nums = 0
            kk_flag = 1
        except Exception as e:
            if 'exceeded your current quota' in str(e):

                # open_ai_keys_index +=1
                openai.api_key = open_ai_keys[open_ai_keys_index]
            time.sleep(1) 
            try_nums = try_nums-1
            kk_flag = 0

    if kk_flag == 0:
        time.sleep(5) 
        response = openai.Completion.create(
                      engine="text-davinci-003",
                      prompt=input_2,
                      max_tokens=256,
                      temperature=0,
                      top_p=1,
                      frequency_penalty=0,
                      presence_penalty=0,
                      n = 1,
                  )

    predictions_2 = response["choices"][0]['text']
    
    
    input_3 = temp_3.format(', '.join(candidate_items), ', '.join(seq_list[-length_limit:]), predictions_1, predictions_2)

    try_nums = 5
    kk_flag = 1
    while try_nums:
        try:
            response = openai.Completion.create(
                      engine="text-davinci-003",
                      prompt=input_3,
                      max_tokens=512,
                      temperature=0,
                      top_p=1,
                      frequency_penalty=0,
                      presence_penalty=0,
                      n = 1,
                  )
            try_nums = 0
            kk_flag = 1
        except Exception as e:
            if 'exceeded your current quota' in str(e):

                # open_ai_keys_index +=1
                openai.api_key = open_ai_keys[open_ai_keys_index]
            time.sleep(1) 
            try_nums = try_nums-1
            kk_flag = 0

    if kk_flag == 0:
        time.sleep(5) 
        response = openai.Completion.create(
                      engine="text-davinci-003",
                      prompt=input_3,
                      max_tokens=256,
                      temperature=0,
                      top_p=1,
                      frequency_penalty=0,
                      presence_penalty=0,
                      n = 1,
                  )

    predictions = response["choices"][0]['text']
    

    hit_=0
    if elem[1] in predictions:
        count += 1
        hit_ = 1
    else:
        pass
    total +=1
    
    
    
    # print (f"input_1:{input_1}")
    # print (f"predictions_1:{predictions_1}\n")
    # print (f"input_2:{input_2}")
    # print (f"predictions_2:{predictions_2}\n")
    # print (f"input_3:{input_3}")
    print (f"GT:{elem[1]}")
    print (f"predictions:{predictions}")
    
    # print (f"GT:{elem[-1]}")
    print (f'PID:{i}; count/total:{count}/{total}={count*1.0/total}\n')
    result_json = {"PID": i,
                   "Input_1": input_1,
                   "Input_2": input_2,
                   "Input_3": input_3,
                   "GT": elem[1],
                   "Predictions_1": predictions_1,
                   "Predictions_2": predictions_2,
                   "Predictions": predictions,
                   'Hit': hit_,
                   'Count': count,
                   'Current_total':total,
                   'Hit@10':count*1.0/total}
    results_data.append(result_json)

    
    
file_dir = f"./results_multi_prompting_len{length_limit}_numcand_{total_i}_seed{rseed}.json"
write_json(results_data, file_dir)
    
    

    
    
import math

import numpy as np
import matplotlib.pyplot as plt


from itertools import combinations,permutations
from matplotlib import rcParams

config = {
    "font.family": 'Times New Roman',  # 设置字体类型
    "axes.unicode_minus": False  # 解决负号无法显示的问题
}
rcParams.update(config)


def get_top(search_type, choose_num, val_results,top=1):
    start = np.zeros(choose_num)
    reward = np.zeros(choose_num)
    num = 0
    for i in range(len(search_type)):
        if len(search_type[i]) == choose_num:
            choose_index = list(search_type[i])
            break
    print(choose_index)
    greedy_list = []
    for j in range(choose_num):
        for i in range(len(search_type)):
            if len(search_type[i]) == 1 and choose_index[j] == search_type[i][0]:
                start[j] = val_results[i]
                break

    val_idx = np.argsort(start)[::-1]
    print(start, val_idx)
    choose = [choose_index[val_idx[i]] for i in range(top)]
    for i in range(len(search_type)):
        if set(search_type[i]) == set(choose):
            return search_type[i]

    assert 0

def get_greedy(search_type,choose_num,val_results):
    start = np.zeros(choose_num)
    reward = np.zeros(choose_num)
    num = 0
    for i in range(len(search_type)):
        if len(search_type[i]) == choose_num:
            choose_index = list(search_type[i])
            break
    print(choose_index)
    greedy_list = []
    greedy_metric = -100
    while True:
        metric_list = []
        for i in range(len(search_type)): 
            if len(search_type[i]) == len(greedy_list) + 1 and \
                set(greedy_list).issubset(set(search_type[i])) :
                    metric_list.append(i)
        if np.max([val_results[i] for i in metric_list]) > greedy_metric:
            i = metric_list[np.argmax([val_results[i] for i in metric_list])]
            greedy_metric = val_results[i]
            greedy_list = search_type[i]
        else:
            break
        
    return greedy_list

def get_reward(search_type,choose_num,val_results,CUT_n):
    start = np.zeros(choose_num)
    reward = np.zeros(choose_num)
    num = 0
    for i in range(len(search_type)):
        if len(search_type[i]) == choose_num:
            choose_index = list(search_type[i])
            break
    print(choose_index)
    for i in range(len(choose_index)):
        num = 0
        for j in range(len(search_type)): 
            if set([choose_index[i]]).issubset(set(search_type[j])) :
                for sub in range(len(search_type)): 
                    if set(search_type[sub]) == (set(search_type[j]) - set([choose_index[i]])):
                        reward[i] += val_results[j] - val_results[sub]
                        num = num + 1
                        break
        reward[i] /= num

    print("reward:",reward)
    # best_list = np.array(choose_index)[reward < 0]
    # if len(best_list) == 0 or len(best_list) == choose_num:
    #     best_list = np.array(choose_index)[reward < np.mean(reward)]


    best_list = np.array(choose_index)[reward > 0]
    if len(best_list) == choose_num:
    # if True:
        # best_list = np.array([choose_index[np.argmax(start)]])
        best_list = np.array([choose_index[np.argmax(reward)]])
    else:
        best_list = np.array(choose_index)[reward > 0]
        # reward_site = np.argsort(-reward)
        # best_list = np.array(choose_index)[reward_site[:4]]

    return best_list.tolist()
    

def draw():

    # {'iou': [], 'color_blind_iou': [], 'accuracy': [],'query_name':[],'support_name':[]}
    # shots = 64
    shots = 16
    choose_num = 6
    SHUNXU = True
    MEAN = True
    CUT_n = choose_num

    path = "./npys/output_dir_trn_16_s{0}_seed{1}/out.npy"



    results = np.load(path,allow_pickle=True).item()
    search_type = np.unique(np.asarray(results['support_name'], dtype = object))
    if 'color' in path:
        print(len(search_type),len(results['mse']))
    else:
        print(len(search_type),len(results['iou']))


    ori_seatch_type = search_type

    if not SHUNXU:
        for item in search_type:
            if len(item) == choose_num:
                choose_index = item
                break
        search_type = np.asarray([list(item) for now in range(choose_num) for item in combinations(choose_index, now+1)], dtype = object)
        print(len(search_type),search_type)

    REWARD = False 

    if REWARD:  
        reward_path = './output_dir_trn_class_20_s1_iou/out.npy'
        # # dev_list = ['2009_004492', '2009_001558'] # all
        # # dev_list = ['2009_002089', '2010_001819'] # 10
        # # dev_list = ['2009_001494', '2009_001848'] # 20 s0
        # # dev_list = ['2008_005010', '2011_001653', '2008_008654'] # 20 s1
        # dev_list = ['2008_000278', '2008_006731'] # 20 s2
        # # dev_list = ['train_122', 'train_422', 'train_496'] # 64 det
        # # dev_list = ['2009_001494', '2009_000130'] # 40

        # dev_list = ['2009_002946', '2011_002560', '2008_006855'] # 20 s0
        dev_list = ['2008_000660'] # 20 s1

        results_reward = np.load(reward_path,allow_pickle=True).item()
        if MEAN:
            reward_test_iou = np.mean(results_reward['iou'])
            reward_test_iou_blind = np.mean(results_reward['color_blind_iou'])
            reward_test_acc = np.mean(results_reward['accuracy'])
        else:
            reward_test_iou = np.max(results_reward['iou'])
            reward_test_iou_blind = np.max(results_reward['color_blind_iou'])
            reward_test_acc = np.max(results_reward['accuracy'])
        for index in range(len(search_type)): 
            if set(search_type[index]) == set(dev_list):
                reward_index = index
                break

        print("reward:",reward_test_iou,reward_test_iou_blind,reward_test_acc)


    # now_search_type = []
    # for item in search_type:
    #     # if len(item) == 1:
    #     #     now_search_type.append(item)
    #     #     print(item)
    #     now_search_type.append(item)

    # search_type = now_search_type

    # print(results)
    if 'color' in path:
        val_mse = np.zeros(len(search_type))
        num = 0
        for i in range(len(ori_seatch_type) * (shots - choose_num)):
            for j in range(len(search_type)):
                if results['support_name'][i] == search_type[j]:
                    num = num + 1
                    if MEAN:
                        val_mse[j] += results['mse'][i] / (shots - choose_num)
                    else:
                        val_mse[j] = max(val_mse[j],results['mse'][i] / (shots - choose_num))
        val_mse = - val_mse
        print(len(search_type) * (shots - choose_num), num)
        mse_greedy = get_greedy(search_type,choose_num,val_mse)
        print("greedy:",mse_greedy)
        top_1 = get_top(search_type,choose_num,val_mse,top=1)
        top_2 = get_top(search_type, choose_num, val_mse, top=2)
        top_4 = get_top(search_type, choose_num, val_mse, top=4)
        print("top:", top_1,top_2,top_4)

        test_mse = np.zeros(len(search_type))
        for i in range(len(ori_seatch_type) * (shots - choose_num), len(results['support_name'])):
            for j in range(len(search_type)):
                if results['support_name'][i] == search_type[j]:
                    num = num + 1
                    if MEAN:
                        test_mse[j] += results['mse'][i] / (
                                    (len(results['support_name']) - len(search_type) * (shots - choose_num)) // len(
                                search_type))
                    else:
                        test_mse[j] = max(test_mse[j],results['mse'][i] / (
                                (len(results['support_name']) - len(search_type) * (shots - choose_num)) // len(
                            search_type)))
        test_mse = -test_mse
        print(len(results['support_name']), num)
        for index in range(len(search_type)): 

            if set(search_type[index]) == set(mse_greedy):
                print("greedy TEST max: {},best {}, mean {}".format(np.max(test_mse),test_mse[index],np.mean(test_mse)))
            if set(search_type[index]) == set(top_1):
                print("top_1 TEST max: {},best {}, mean {}".format(np.max(test_mse),test_mse[index],np.mean(test_mse)))
            if set(search_type[index]) == set(top_2):
                print("top_2 TEST max: {},best {}, mean {}".format(np.max(test_mse),test_mse[index],np.mean(test_mse)))
            if set(search_type[index]) == set(top_4):
                print("top_4 TEST max: {},best {}, mean {}".format(np.max(test_mse),test_mse[index],np.mean(test_mse)))
        

        mse = np.argsort(val_mse)

        for index in range(len(search_type)): 
            if set(search_type[index]) == set(mse_greedy):
                greedy_index = index
                break
        test_mse = -test_mse
        plt.scatter(mse.tolist().index(greedy_index),test_mse[greedy_index],c='green')

        # plt.plot(val_mse[mse], test_mse[mse])
        
        plt.plot(range(len(search_type)), test_mse[mse])
        plt.title("mse")
        plt.xlabel('val')
        plt.ylabel('test')
        plt.show()
        plt.savefig("mse.jpg")
        plt.cla()

    else:
        val_iou = np.zeros(len(search_type))
        val_iou_blind = np.zeros(len(search_type))
        val_acc = np.zeros(len(search_type))
        num = 0
        for i in range(len(ori_seatch_type) * (shots - choose_num)):
            for j in range(len(search_type)):
                if results['support_name'][i] == search_type[j]:
                    num = num + 1
                    if MEAN:
                        val_iou[j] += results['iou'][i]/(shots - choose_num)
                        val_iou_blind[j] += results['color_blind_iou'][i] / (shots - choose_num)
                        val_acc[j] += results['accuracy'][i] / (shots - choose_num)
                    else:
                        val_iou[j] = max(val_iou[j],results['iou'][i]/(shots - choose_num))
                        val_iou_blind[j] = max(val_iou_blind[j],results['color_blind_iou'][i] / (shots - choose_num))
                        val_acc[j] = max(val_acc[j],results['accuracy'][i] / (shots - choose_num))


        print(len(search_type) * (shots - choose_num),num)
        iou_reward = get_reward(search_type,choose_num,val_iou,CUT_n)
        iou_blind_reward = get_reward(search_type,choose_num,val_iou_blind,CUT_n)
        acc_reward = get_reward(search_type,choose_num,val_acc,CUT_n)


        iou_greedy = get_greedy(search_type,choose_num,val_iou)
        iou_blind_greedy = get_greedy(search_type,choose_num,val_iou_blind)
        acc_greedy = get_greedy(search_type,choose_num,val_acc)

        print("reward:",iou_reward,iou_blind_reward,acc_reward)
        print("greedy:",iou_greedy,iou_blind_greedy,acc_greedy)

        top_1 = get_top(search_type, choose_num, val_iou, top=1)
        top_2 = get_top(search_type, choose_num, val_iou, top=2)
        top_4 = get_top(search_type, choose_num, val_iou, top=4)
        print("top:", top_1, top_2, top_4)

        # for index in range(len(search_type)):
        #     if set(search_type[index]) == set(iou_reward):
        #         print("reward VAL max: {},best {}".format(np.max(val_iou),val_iou[index]))
        #     if set(search_type[index]) == set(iou_blind_reward):
        #         print("reward VAL max: {},best {}".format(np.max(val_iou_blind),val_iou_blind[index]))
        #     if set(search_type[index]) == set(acc_reward):
        #         print("reward VAL max: {},best {}".format(np.max(val_acc),val_acc[index]))
        #
        #     if set(search_type[index]) == set(iou_greedy):
        #         print("greedy VAL max: {},best {}".format(np.max(val_iou),val_iou[index]))
        #     if set(search_type[index]) == set(iou_blind_greedy):
        #         print("greedy VAL max: {},best {}".format(np.max(val_iou_blind),val_iou_blind[index]))
        #     if set(search_type[index]) == set(acc_greedy):
        #         print("greedy VAL max: {},best {}".format(np.max(val_acc),val_acc[index]))

        test_iou = np.zeros(len(search_type))
        test_iou_blind = np.zeros(len(search_type))
        test_acc = np.zeros(len(search_type))
        for i in range(len(ori_seatch_type) * (shots - choose_num),len(results['support_name'])):
            for j in range(len(search_type)):
                if results['support_name'][i] == search_type[j]:
                    num = num + 1
                    if MEAN:
                        test_iou[j] += results['iou'][i] / ((len(results['support_name'])-len(search_type) * (shots - choose_num))//len(search_type))
                        test_iou_blind[j] += results['color_blind_iou'][i] / ((len(results['support_name'])-len(search_type) * (shots - choose_num))//len(search_type))
                        test_acc[j] += results['accuracy'][i] / ((len(results['support_name'])-len(search_type) * (shots - choose_num))//len(search_type))
                    else:
                        test_iou[j] = max(test_iou[j],results['iou'][i] / ((len(results['support_name'])-len(search_type) * (shots - choose_num))//len(search_type)))
                        test_iou_blind[j] = max(test_iou_blind[j],results['color_blind_iou'][i] / ((len(results['support_name'])-len(search_type) * (shots - choose_num))//len(search_type)))
                        test_acc[j] = max(test_acc[j],results['accuracy'][i] / ((len(results['support_name'])-len(search_type) * (shots - choose_num))//len(search_type)))
        
        for index in range(len(search_type)): 
            if set(search_type[index]) == set(iou_reward):
                print("reward TEST max: {},best {}, mean {}".format(np.max(test_iou),test_iou[index],np.mean(test_iou)))
            # if set(search_type[index]) == set(iou_blind_reward):
            #     print("reward TEST max: {},best {}, mean {}".format(np.max(test_iou_blind),test_iou_blind[index],np.mean(test_iou_blind)))
            # if set(search_type[index]) == set(acc_reward):
            #     print("reward TEST max: {},best {}, mean {}".format(np.max(test_acc),test_acc[index],np.mean(test_acc)))

            if set(search_type[index]) == set(iou_greedy):
                print("greedy TEST max: {},best {}, mean {}".format(np.max(test_iou),test_iou[index],np.mean(test_iou)))
            # if set(search_type[index]) == set(iou_blind_greedy):
            #     print("greedy TEST max: {},best {}, mean {}".format(np.max(test_iou_blind),test_iou_blind[index],np.mean(test_iou_blind)))
            # if set(search_type[index]) == set(acc_greedy):
            #     print("greedy TEST max: {},best {}, mean {}".format(np.max(test_acc),test_acc[index],np.mean(test_acc)))

            if set(search_type[index]) == set(top_1):
                print("top_1 TEST max: {},best {}, mean {}".format(np.max(test_iou),test_iou[index],np.mean(test_iou)))
            if set(search_type[index]) == set(top_2):
                print("top_2 TEST max: {},best {}, mean {}".format(np.max(test_iou),test_iou[index],np.mean(test_iou)))
            if set(search_type[index]) == set(top_4):
                print("top_4 TEST max: {},best {}, mean {}".format(np.max(test_iou),test_iou[index],np.mean(test_iou)))
        
        



draw()

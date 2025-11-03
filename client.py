

import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import time
from utils.util import logging
import random
import tenseal as ts
from threading import Thread
from encryption.ckks import ckks_enc,ckks_dec
# from encryption.bfv import bfv_enc,bfv_dec
# from encryption.paillier import paillier_enc,paillier_dec

import utils.min_hash as min_lsh

import pickle
from multiprocessing import shared_memory


def params_tolist(model):
    """
    Model parameters converted to list

    Args:
        model:
            Model to be converted.
    Returns:
        params_list
            The converted parameter list.
        params_num
            The amount of parameters for each layer.
        layer_shape
            Shape of each layer.
    """ 
    model.to('cpu')
    local_state = model.state_dict()
    params_list = []
    layer_shape = {}
    params_num = {}
    layer_params = []
    for key in model.state_dict().keys():
        layer_shape[key] = local_state[key].shape
        params_num[key] = int(np.prod(local_state[key].shape))
        layer_params = local_state[key].reshape(params_num[key]).tolist()
        params_list.append(layer_params)
    params_list = [b for a in params_list for b in a]
    return params_list,params_num,layer_shape


def params_tomodel(model,global_list,params_num,layer_shape,args,params_list):
    """
    Parameter list to model

    Args:
        model:
            The model obtained after parameter conversion.
        global_list
            Global model parameter list.
        params_num
            The amount of parameters for each layer.
        layer_shape
            Shape of each layer.
        args
            Hyper-parameters.
        params_list
            Local parameter list
    Returns:
        None
    """ 
    update_state = OrderedDict()
    model.to('cpu')
    idx_cnt = 0      
    if args.isSpars == 'topk' or args.isSpars == 'randk':
        for idx, key in enumerate(model.state_dict().keys()):
            layer_size = int(params_num[key])
            tmp = global_list[idx_cnt : idx_cnt + layer_size]

            # The part with a value of 0 is replaced by local parameters.
            for idx_tmp in range(len(tmp)):
                if tmp[idx_tmp] == 0 and ( idx_tmp == len(tmp)- 1 or tmp[idx_tmp+1]==0 ):
                    tmp[idx_tmp] = params_list[idx_cnt + idx_tmp]
                    # global_list[idx_cnt+idx_tmp] = tmp[idx_tmp]
            update_state[
                key] =  torch.from_numpy(np.array(tmp).reshape(layer_shape[key]))
            idx_cnt += layer_size            
    else:
        for idx, key in enumerate(model.state_dict().keys()):
            layer_size = int(params_num[key])
            tmp = global_list[idx_cnt:idx_cnt + layer_size]
            update_state[
                key] =  torch.from_numpy(np.array(tmp).reshape(layer_shape[key]))
            idx_cnt += layer_size

    model.load_state_dict(update_state)


def minHash(rank,random_R,global_list,params_list, args, quan_thres = 0.05):
    '''
    quan_thres: Tthreshold value used for quantization
    sim_len: Number of hash functions
    '''
    sim_len = args.sim_len

    mat = np.concatenate((np.array(global_list).reshape(-1,1),np.array(params_list).reshape(-1,1)),axis=1)

    quan_matrix = min_lsh.quan_params(mat,quan_thres)

    sim_mat = min_lsh.sigMatrixGen(quan_matrix,random_R, sim_len)

    # client_sim2 = min_lsh.dim_reduce_sim(sim_mat)

    minHash = (sim_mat[:,1]).tolist()
    return minHash

# Simulate straggler
def straggler(rank):
    timewait = np.random.randint(10,15)
    if rank == 1:
        time.sleep(timewait)
    if rank == 2:
        time.sleep(timewait)

# CKKS 스타일: Flatten → pack(P개) → 각 pack L2-norm(정확히는 ||·||^2) 계산 → 상위 ζ% pack 선택.
def pack_level_sparse(arr: np.ndarray, pack_size: int, zeta: float):
    """
    CKKS 스타일: Flatten → pack(P개) → 각 pack L2-norm(정확히는 ||·||^2) 계산 → 상위 ζ% pack 선택.
    반환은 '요소 인덱스 + 값'로 해서, 기존 server.py의 평문 스파스 처리(인덱스/값 쌍)와 호환시킴.
    """
    N = len(arr)
    P = int(pack_size)
    B_tot = int(N / P)          # 총 pack 개수
    # 1) pack별 L2^2 스코어 계산
    scores = []
    for j in range(B_tot):
        s = j * P
        e = min((j + 1) * P, N)
        block = arr[s:e]
        scores.append((j, float(np.dot(block, block))))  # ||block||^2

    # 2) 상위 ζ 비율의 pack을 선택
    B_sel = max(1, int(zeta * B_tot))
    top = sorted(scores, key=lambda x: x[1], reverse=True)[:B_sel]
    sel_packs = sorted([j for (j, _) in top])  # pack index 오름차순

    # 3) 선택된 pack 내부의 '요소 인덱스'를 모두 모아 전송용으로 구성
    idx_list = []
    vals = []
    for j in sel_packs:
        s = j * P
        e = min((j + 1) * P, N)
        local_idx = list(range(s, e))
        idx_list.extend(local_idx)
        vals.extend(arr[s:e].tolist())

    return idx_list, vals, B_tot, len(sel_packs)

def client_process(rank, args, model, device,dataset, test_dataset, kwargs,kwargs_IPC,train_weights):
    torch.manual_seed(args.seed + rank)
    queue = kwargs_IPC['queues'][rank]
    e = kwargs_IPC['e']
    lock = kwargs_IPC['lock']
    pipe = kwargs_IPC['client_pipes'][rank][0]
    flag = kwargs_IPC['flag']
    e_server = kwargs_IPC['e_server']
    acc_queue = kwargs_IPC['acc_queue']
    self_weight = train_weights[rank]
    acc_pipe = kwargs_IPC['send_pipes'][rank][0]

    if args.enc and args.algorithm == 'paillier':
        enc_tools = kwargs_IPC['enc_tools']
    else:
        enc_tools = {}

    if args.isSelection:
        random_R = kwargs_IPC['random_R']

    hash_queue = kwargs_IPC['hash_queue']
    train_loader = torch.utils.data.DataLoader(dataset, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    if rank == 0:
        spars_list = []
    sum_masks = []
    epoch = 0

    self_flag = True
    acc_list = []
    
    while not flag.value:
        
        # epoch_begin = time.time() ##

        train_begin = time.time() ##
        train_epoch(epoch, args, model, device, train_loader, optimizer,rank)
        train_end = time.time() ##
        logging("id:{},train time:{}".format(rank,train_end-train_begin),args) ##

        params_list,params_num,layer_shape = params_tolist(model)
        total_sum = sum(params_num.values())
        if args.enc and args.algorithm == 'paillier':
            enc_tools.update({'total_params':total_sum})

        # if selected 
        if self_flag:    
            # if epoch > 0 :   
            #     straggler(rank)
            if args.enc:
                if args.algorithm == 'paillier':
                    params_list = (np.array(params_list) * self_weight).tolist()
                    
                if args.algorithm == 'bfv':
                    params_list = (np.array(params_list) * self_weight).tolist()

                if args.isSpars == 'topk' :  
                    enc_begin = time.time() ##
                    cipher, mask = enc_params(params_list,enc_tools, args)
                    pipe.send([rank,mask,cipher])
                    enc_end = time.time() ##
                    logging("id:{},enc time:{}".format(rank,enc_end-enc_begin),args) ##
                    lock.acquire() ##
                    # logging("client {}, send mask {}.".format(rank,mask),args) ##
                    logging(f"client {rank}, mask ones: {sum(mask)}, len: {len(mask)}", args) ##
                    lock.release() ##

                elif args.isSpars == 'randk':
                    cipher, randk_list = enc_params(params_list,enc_tools,args,epoch = epoch)
                    if rank == 0:
                        logging("epoch:{},rand_K:{}".format(epoch,randk_list),args)
                    pipe.send([rank,cipher])                 

                elif args.isSpars == 'full':
                    enc_begin = time.time() ##
                    cipher = enc_params(params_list,enc_tools,args,epoch = epoch)
                    pipe.send([rank,cipher])
                    enc_end = time.time() ##
                    logging("id:{},enc time:{}".format(rank,enc_end-enc_begin),args) ##

            else:
                ''' 추가 '''
                # ===== 여기(평문 송신 지점)에 '스파스 + 트래픽 로깅'을 넣는다 =====
                arr = np.array(params_list, dtype=np.float32)

                if args.isSpars == 'topk':
                    # ✅ pack 기준 sparsification: enc_batch_size = pack 크기, topk = ζ(비율)
                    pack_size = int(args.enc_batch_size)
                    zeta = float(args.topk)  # 예: 0.2 → 상위 20% pack
                    idx_list, sparse_vals, B_tot, B_sel = pack_level_sparse(arr, pack_size, zeta)

                    # 업로드 바이트 추정(평문): 값 k개(4B) + 인덱스 k개(4B)
                    k = len(idx_list)
                    if args.cipher_count:
                        bytes_out = k * 4 + k * 4
                        logging(
                            f"[PT-PackSparse] client {rank} uplink ~{bytes_out} bytes "
                            f"(packs {B_sel}/{B_tot}, elements {k})", args
                        )

                    # 전송 (서버는 [idx_list, vals]를 받아 복원)
                    pipe.send([rank, idx_list, sparse_vals])

                else:
                    # full 평문 전송
                    if args.cipher_count:
                        bytes_out = len(arr) * 4
                        logging(f"[PT-Full] client {rank} uplink ~{bytes_out} bytes", args)
                    pipe.send([rank, arr.tolist()])
                ''' 추가 끝 '''
            # lock.acquire() ##
            # logging("client {}, send params {}.".format(rank,params_list[0]),args) ##
            # lock.release() ##

        if flag.value:
            break       

        # Waiting for server aggregation
        e.wait()

        global_list = queue.get()
        involved_frac = global_list[0]   # 평문=가중치합, CKKS-full=가중치합, CKKS-topk=pack별 합 sum_mask   
        global_weights = global_list[1]  # 평문=합계 벡터, CKKS=암호문 리스트


        if args.enc:
            if args.isSpars == 'topk':
                dec_begin = time.time() ##
                sum_masks = involved_frac
                global_weights = (dec_params(global_weights,sum_masks,enc_tools, args)).tolist() # pack별 정규화 내장
                dec_end = time.time() ##
                logging("id:{},dec time:{}".format(rank,dec_end-dec_begin),args) ##

            elif args.isSpars == 'randk':
                global_weights = (dec_params(global_weights,sum_masks,enc_tools, args, randk_list)).tolist()

            else: # full
                dec_begin = time.time() ##
                global_weights = (dec_params(global_weights,sum_masks, enc_tools,args) / involved_frac).tolist()
                dec_end = time.time() ##
                logging("id:{},dec time:{}".format(rank,dec_end-dec_begin),args) ##

            global_weights = global_weights[:total_sum]

        else: # plain-text
            global_weights = (np.array(global_weights) / involved_frac).tolist() ## 필수
            
        # lock.acquire() ##
        # print('client{},receive{}'.format(rank,global_weights[0])) ##
        # lock.release() ##
        params_list,params_num,layer_shape = params_tolist(model)

        params_tomodel(model,global_weights,params_num,layer_shape,args,params_list)

        if args.enc:
            client_acc,client_loss = test_epoch(model, device, test_loader)
            acc_pipe.send([rank,client_acc,client_loss])
            # print('client{},acc:{},loss:{}'.format(rank,client_acc,client_loss)) 

        if args.isSelection:
            client_hash = minHash(rank, random_R,global_weights,params_list,args)
            hash_queue.put([rank,client_hash])

            if flag.value:
                break

            # Wait for server to make client selection
            e_server.wait()
            
            selected_file = os.path.join(args.data_dir, args.dataset + 'selected')
            with open(selected_file, "rb") as f:
                clients_bytes = f.read()
                clients_share = list(pickle.loads(clients_bytes))[0]
                clients_weights = list(pickle.loads(clients_bytes))[1]

            if rank not in clients_share:
                self_flag = False
            else:
                self_flag = True
                idx = clients_share.index(rank)
                self_weight = clients_weights[idx]      
        # epoch_end = time.time() ##
 
        epoch += 1

    lock.acquire()
    logging("client {} finished!".format(rank),args)
    lock.release()
    return


def test(args, model, device, dataset, kwargs):
    torch.manual_seed(args.seed)

    test_loader = torch.utils.data.DataLoader(dataset, **kwargs)

    return test_epoch(model, device, test_loader)


def train_epoch(epoch, args, model, device, data_loader, optimizer,rank): # 표준 학습 루프 (CE Loss)
    model.to(device)
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(data_loader):
        
        output = model(data.to(device))
        target = target.to(device)
        # loss = F.nll_loss(output, target.to(device))
        # loss = torch.nn.CrossEntropyLoss()(output, target.to(device))
        loss = loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if batch_idx == len(data_loader) - 1:     ##  
        #     logging('client {}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         rank, epoch, batch_idx * len(data), len(data_loader.dataset),
        #         100. * batch_idx / len(data_loader), loss.item())) ##
     

def test_epoch(model, device, data_loader): # testset에서 평균 loss/acc 계산해 반환 (암호문일 때만 클라가 서버로 보냄)
    model.to(device)
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            # test_loss += F.nll_loss(output, target.to(device), reduction='sum').item() # sum up batch loss ##
            # test_loss += torch.nn.CrossEntropyLoss()(output, target.to(device),reduction='sum') ##
            test_loss +=F.cross_entropy(output, target.to(device), reduction='sum')
            pred = output.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.to(device)).sum().item()
    test_loss /= len(data_loader.dataset)
    # print("loss",test_loss) ##
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(data_loader.dataset),
    #     100. * correct / len(data_loader.dataset))) ##
    client_acc = correct / len(data_loader.dataset)

    # set the return variable format
    test_loss = round(float(test_loss),3) 
    client_acc = round(client_acc,4)*100 
    return client_acc, test_loss


# Encrypt all parameters of a client
def enc_params(params_list,enc_tools,args,epoch = 0):
    if args.algorithm == 'ckks':
        ckks_file = os.path.join(args.data_dir + 'context_params')
        with open(ckks_file, "rb") as f:
            params = f.read()
        ckks_ctx = ts.context_from(params) 
        return ckks_enc(params_list,ckks_ctx,isBatch=args.isBatch,batch_size=args.enc_batch_size,
                        topk=args.topk,round = epoch,randk_seed=args.randk_seed, is_spars = args.isSpars)
    elif args.algorithm =='paillier':
        from encryption.paillier import paillier_enc
        cls_paillier = enc_tools['cls_paillier']
        return  paillier_enc(params_list,cls_paillier,args)
    elif args.algorithm == 'bfv':
        from encryption.bfv import bfv_enc
        bfv_file = os.path.join(args.data_dir + 'bfv_ctx')
        with open(bfv_file, "rb") as f:
            params = f.read()
        bfv_ctx = ts.context_from(params)      
        return bfv_enc(params_list,bfv_ctx,args)
    else:
        raise ValueError("please select valid algorithm")
    
# Decrypt all parameters of a client
def dec_params(cipher_list,sum_masks, enc_tools,args, randk_list = []):
    if args.algorithm == 'ckks':
        ckks_file = os.path.join(args.data_dir + 'context_params')
        with open(ckks_file, "rb") as f:
            params = f.read()
        ckks_ctx = ts.context_from(params) 
        sk = ckks_ctx.secret_key()
        return ckks_dec(cipher_list,ckks_ctx,sk,args.isBatch,randk_list,sum_masks,args.enc_batch_size)
    elif args.algorithm =='paillier':
        from encryption.paillier import paillier_dec
        cls_paillier = enc_tools['cls_paillier']
        total_params = enc_tools['total_params']
        return  paillier_dec(cipher_list,cls_paillier,total_params,args)
    elif args.algorithm == 'bfv':
        from encryption.bfv import bfv_dec
        bfv_file = os.path.join(args.data_dir + 'bfv_ctx')
        with open(bfv_file, "rb") as f:
            params = f.read()
        bfv_ctx = ts.context_from(params) 
        sk = bfv_ctx.secret_key()
        return bfv_dec(cipher_list,bfv_ctx,sk,args.isBatch,args.quan_bits,args.n_clients,sum_masks,args.enc_batch_size)    
    else:
        raise ValueError("please select valid algorithm")  


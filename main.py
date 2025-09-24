
import datetime
import argparse
import os
import torch
import torch.multiprocessing as mp
from utils.util import logging
import random
import numpy as np
import tenseal as ts
from fed import run
from encryption.paillier import PaillierCipher


def arg_parse():
    parser = argparse.ArgumentParser()

    # dataset and parameters
    parser.add_argument('--dataset', type=str, default='MNIST',
                            help='datasets: MNIST, FashionMNIST, CIFAR10, CIFAR100')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    '''추가'''
    # parser.add_argument('--weighted',type=bool,default=True)
    parser.add_argument('--weighted', action=argparse.BooleanOptionalAction, default=True,
                        help='weighted or not')
    '''추가'''
    parser.add_argument('--n_clients', type=int, default= 5, metavar='N',
                        help='how many training processes to use (default: 5)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')

    
    # data split 
    parser.add_argument('--n_shards', type=int, default=5,
                        help='number of shards (parameter of pathological, default: 5)')
    parser.add_argument('--alpha', type=float, default=1,
                        help='parameter of dirichlet (default: 1)')
    parser.add_argument('--sgm', type=float, default=0.3,
                        help='parameter of unbalance (default: 0.3)')
    parser.add_argument('--split', type=str, default='iid',
                        help='split method: iid or non-iid')
    parser.add_argument('--noniid_method', type=str, default='dirichlet',
                        help='noniid method: pathological or dirichlet')  
    
    # modules 
    '''추가'''
    # parser.add_argument('--enc',type=bool,default=True,
    #                     help='enc or not')
    parser.add_argument('--enc', action=argparse.BooleanOptionalAction, default=True,
                        help='encrypt or not')
    # parser.add_argument('--isSelection',type=bool,default=True, 
                        # help='Client selection or not')
    parser.add_argument('--isSelection',action=argparse.BooleanOptionalAction,default=True, 
                        help='Client selection or not')
    '''추가'''
    parser.add_argument('--isSpars', type=str, default='topk',
                        help='sparsification method: topk or randk or full')   
         
    # sparsification
    parser.add_argument('--topk',type=float,default=0.2,
                        help='sparfication fraction')
                  
    # encryption params
    '''추가'''
    # parser.add_argument('--isBatch',type=bool,default=True,
    #                     help='Batch HE or not')
    parser.add_argument('--isBatch',action=argparse.BooleanOptionalAction,default=True, 
                        help='Batch HE or not')
    # parser.add_argument('--cipher_count',type=bool,default=True,
    #                     help='ciphertext size')
    parser.add_argument('--cipher_count',action=argparse.BooleanOptionalAction,default=True, 
                        help='ciphertext size')
    '''추가'''
    parser.add_argument('--algorithm',type=str,default='ckks',
                        help='HE algorithm: paillier,bfv, ckks')
    parser.add_argument('--quan_bits',type=int,default=16,
                        help='quantification bits (default: 16)')
    parser.add_argument('--enc_batch_size',type=int,default=4096,
                        help='Batch Encryption size (default: 4096)') 

    # selection
    parser.add_argument('--sim_len',type=int,default=200,
                        help='Locally Sensitive Hasing (lsh) matrix width (default: 200)')
    
    # device and logdir                   
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--mps', action='store_true', default=True,
                            help='enables macOS GPU training')
    parser.add_argument("--log_dir", type=str,
                            default="log", help="directory of logs")
    parser.add_argument("--data_dir", type=str,
                            default="data_dir/", help="directory of data (default: data_dir/)")                        
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--randk_seed', type=int, default=12, help='random k packages seed (default: 12)')

    return parser.parse_args()


def seed_everything(seed,is_cuda): # default: seed = 42, cuda = True
    """
    Seed function for randomization

    Args:
        seed (`int`):
            The seed in the parameters.
        is_cuda (`bool`):
            Whether to enable CUDA training.
    Returns:
        None
    """ 
    random.seed(seed) # 파이썬 내장 random 모듈의 시드를 고정 (random.random(), random.shuffle() 등 결과가 항상 같아짐)
    torch.manual_seed(seed) # 파이토치의 CPU 연산 난수 시드 고정 -> 텐서 초기화, 드롭아웃 등이 재현 가능해짐
    np.random.seed(seed) # 넘파이 난수 시드 고정 → np.random.rand() 같은 호출이 항상 동일 결과
    os.environ["PYTHONHASHSEED"] = str(seed) # 파이썬의 해시(seed)가 고정됨 → 딕셔너리 순서 같은 일부 내부 동작의 불확실성 제거
    # initialize the gpu device id
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" # GPU 디바이스 강제로 0번만 보이도록 설정 (멀티 GPU가 있을 때 하나만 쓰겠다는 의미)
    if is_cuda:
        torch.cuda.manual_seed_all(seed) #is_cuda # seed가 되어야 하지 않나... 모든 GPU 디바이스의 난수 발생기를 한 번에 시드 고정. 멀티 GPU 환경에서 쓰는 함수
        torch.backends.cudnn.deterministic = True # 연산을 완전 결정론적으로, 항상 같은 결과 보장
        torch.backends.cudnn.benchmark = False # 입력 크기에 따라 성능 최적화를 하지 않고, 일정한 커널만 사용


def init_logger(args):
    """
    Remove the historical log file 

    Args:
        log_dir (`arg_parse` ):
            The directory to save log files.
    Returns:
        None
    """
    # log_file = os.path.join(log_dir, dataset + '.log')
    # if os.path.exists(log_file):
    #     os.remove(log_file)
    # 실행 조건으로 run 이름 구성 (원하는 키만 더/덜 넣어도 됨)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = (
        f"{args.dataset}"
        f"_enc{args.enc}"
        f"_spars{args.isSpars}"
        f"_clients{args.n_clients}"
        f"_epochs{args.epochs}"
        f"_topk{args.topk}"
        f"_lr{args.lr}"
        f"_{ts}"
    )
    # log 파일 경로를 args에 심어둠
    args.log_file = os.path.join(args.log_dir, f"{run_name}.log")

    # 동일 이름의 파일이 있으면 지움 (대부분은 새 이름이라 안 지워도 상관없음)
    if os.path.exists(args.log_file):
        os.remove(args.log_file)

def ckks_init(data_dir):
    """
    Initialize and write context in CKKS encryption mode.

    Args:
        data_dir (`str`):
            Directory for data to store.

    Returns:
        None.
    """  
    ckks_ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    ckks_ctx.global_scale=2**40  
    ckks_ctx.generate_galois_keys()
    params = ckks_ctx.serialize(save_secret_key=True)
    ckks_file = os.path.join(data_dir + 'context_params')
    with open(ckks_file, "wb") as f:
        f.write(params)

def bfv_init(data_dir):
    """
    Initialize and write context in BFV encryption mode.

    Args:
        data_dir (`str`):
            Directory for data to store.

    Returns:
        None.
    """
    bfv_ctx = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=8192, plain_modulus=1032193)
    params = bfv_ctx.serialize(save_secret_key=True)
    ckks_file = os.path.join(data_dir + 'bfv_ctx')
    with open(ckks_file, "wb") as f:
        f.write(params)

def paillier_init():
    """
    Initialize context in paillier encryption mode.

    Args:
        None.

    Returns:
        enc_tools (`dict`):
            cls_paillier (`context`):
                PaillierCipher context.
            mod (`int`):
                mod size.
            num_bits_per_batch (`int`):
                bit length for one package.
    """
    enc_tools = {}
    cls_paillier = PaillierCipher()
    cls_paillier.generate_key(n_length=2048)
    mod = pow(cls_paillier.get_n(),2)

    enc_tools['cls_paillier'] = cls_paillier
    enc_tools['mod'] = mod
    enc_tools['num_bits_per_batch'] = (cls_paillier.get_n() ** 2).bit_length()
    return enc_tools


def IPC_init(n_clients):
    """
    IPC communication between processes.

    Args:
        n_clients (`int` ):
            The num of clients to participate.
    Returns:
        `dict`: The locks, pipes, queues, flag, event in multiprocessing communication.
                lock_print, queue_lock (`Lock`): 
                    Process lock for print and logging.
                flag (`Value`):
                    Record whether the iteration is terminated.
                e, e_server (`Event`):
                    Synchronization with clients.
                pipes, send_pipes (`Pipe`):
                    (Encrypted) gradients send to the server.
                queues, acc_queue, hash_queue, clients_queues (`Queue`):
                    Server send (encrypted) aggregated gradients to clients, 
                    clients send local accuracy to the server,
                    clients send hash value to the server,
                    server send clients selected in the next epoch.
    """ 
    lock_print = mp.Lock()
    queue_lock = mp.Lock()

    flag = mp.Value('b', False)

    e = mp.Event()
    e_server = mp.Event()

    pipes = [mp.Pipe() for _ in range(n_clients)]
    send_pipes = [mp.Pipe() for _ in range(n_clients)]

    queues = [mp.Queue(1) for _ in range(n_clients)]
    acc_queue = mp.Queue(n_clients)
    hash_queue = mp.Queue(n_clients)
    clients_queues = mp.Queue(n_clients)
    

    kwargs_IPC = {'lock':lock_print,'e':e,'client_pipes':pipes,'queues':queues,'flag':flag,'e_server':e_server,
               'acc_queue':acc_queue,'hash_queue':hash_queue,'queue_lock':queue_lock,'clients_queues':clients_queues
               ,'send_pipes':send_pipes,}
    return kwargs_IPC

def device_init(is_cuda,is_mps):
    """
    Determine which device to train on.

    Args:
        is_cuda (`bool`):
            Whether to enable CUDA training.
        is_mps (`bool`):
            Whether to enable mps training.
    Returns:
        None
    """
    use_cuda = is_cuda and torch.cuda.is_available()
    use_mps = is_mps and torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device     
               
def main():
    """
    Main function.

    """
    mp.set_start_method('spawn', force=True) # 새 프로세스로 깨끗하게 띄움 -> 서버 1개 + 클라이언트 n개를 별도 프로세스로 돌릴 준비
    args = arg_parse()

    seed_everything(args.seed, args.cuda) # 난수 발생기를 모두 동일한 시드로 고정하고, CUDA/cuDNN 설정을 통해 연산을 결정론적으로 만들어서 실행할 때마다 동일한 결과(재현성)를 보장
   
    # init_logger(args.log_dir,args.dataset) # Remove the historical log file
    init_logger(args)

    device = device_init(args.cuda,args.mps) # Determine which device to train on.

    kwargs_IPC = IPC_init(args.n_clients) # IPC communication between processes.

    if args.enc :
        if args.algorithm == 'ckks':
            ckks_init(args.data_dir)
        elif args.algorithm == 'paillier':
            enc_tools = paillier_init()
            kwargs_IPC.update({'enc_tools':enc_tools,})
        elif args.algorithm == 'bfv':
            bfv_init(args.data_dir)
        else:
            raise ValueError("invalid algorithm!")
    
    logging("Basic information: device {}, learning rate {}, num clients {}, epochs {},noniid_method {},\
            isEnc {},isBatch {},sparsification {}, client selection {},topk {}, enc_batch_size {}".format(
        device, args.lr,args.n_clients,args.epochs,args.noniid_method,args.enc,args.isBatch,args.isSpars,args.isSelection,args.topk,args.enc_batch_size),args)

    run(args,kwargs_IPC,device) # Run function to launch server and clients processes.


if __name__ == '__main__':
    main()
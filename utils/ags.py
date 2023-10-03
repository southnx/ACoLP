import argparse
# from re import T


def get_args(description='MILNCE'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--run_file_name', default='main.py',
                        type=str, help='the name of the excuted file')
    parser.add_argument('--exp_descript', type=str,
                        default='VidHOI')
    # parser.add_argument('--Model', type=str, default='KLDiv_adj2_adv',
    #                     help='Adj matrix generation function used in Model')
    parser.add_argument('--optimizer', type=str,
                        default='adam', help='opt algorithm')
    parser.add_argument('--lr_scheduler', type=str, default='StepLR')
    parser.add_argument('--batch_size', type=int,
                        default=24, help='batch size')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr_scale', default=2, type=int,
                        help='scale rate of learning rate')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use for single GPU training.')
    parser.add_argument('--classes', required=False,
                        help='number of classes', default=100)
    parser.add_argument('--w_loss_con_i2t', type=float, default=2e-3)
    parser.add_argument('--w_loss_con_t2i', type=float, default=1e-3)
    parser.add_argument('--w_loss_triplet', type=float, default=1e0)
    # parser.add_argument('--w_loss_bigraph', type=float, default=1)
    parser.add_argument('--local_rank', type=int, default=0,
                        help='node rank for distributed training')
    parser.add_argument('--world_size', type=int,
                        help='the number of GPUs', default=4)
    parser.add_argument('--step_size', type=int,
                        help='step size for learning rate decay', default=50)
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--warmup', default=True, type=bool,
                        help='warm up learning rate')
    parser.add_argument('--clip_grad', type=bool, default=False,
                        help='clip gradient to avoid gradient explosion')
    parser.add_argument('--clip_grad_max_norm', type=float, default=1.0,
                        help='max norm of clip gradient to avoid gradient explosion')
    parser.add_argument('--mode', default='KL', type=str, help='mode of GCN')
    parser.add_argument('--temp_scale', default=3.0, type=float,
                        help='temperature scaling for calibration')
    parser.add_argument('--distri', default=False, type=bool,
                        help='distributed training')
    parser.add_argument('--is_dynamic_gnn', default=False, type=bool,
                        help='Dynamic GNN is used')
    parser.add_argument('--re_size', default=(360, 480), type=tuple,
                        help='resize frames')

    args = parser.parse_args()
    # print(args)
    return args

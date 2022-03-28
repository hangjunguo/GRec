import torch
import torch.nn as nn
import generator_grectorch
import time
import math
import numpy as np
import argparse
import random
import collections
import Data_loader
import sys



# A PyTorch implementation of Future Data Helps Training: Modelling Future Contexts for Session-based Recommendation

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                              ["index", "label"])

def create_masked_lm_predictions_frombatch(item_batch, masked_lm_prob,
                                 max_predictions_per_seq, items, item_size):
    # rng=random.Random(123)
    rng = random.Random()
    output_tokens_batch=[]
    maskedpositions_batch=[]
    maskedlabels_batch=[]
    masked_lm_weights_batch=[]
    # item_batch_=item_batch[:, 1:-1]#remove start and end
    item_batch_ = item_batch[:, 1:]  # remove start
    for line_list in range(item_batch_.shape[0]):

        # output_tokens, masked_lm_positions, masked_lm_labels=create_masked_lm_predictions(item_batch[line_list],masked_lm_prob,max_predictions_per_seq,items,rng,item_size)
        output_tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(item_batch_[line_list],
                                                                                            masked_lm_prob,
                                                                                            max_predictions_per_seq,
                                                                                            items, rng, item_size)
        # print output_tokens
        output_tokens.insert(0,item_batch[line_list][0])
        output_tokens_batch.append(output_tokens)
        maskedpositions_batch.append(masked_lm_positions)
        maskedlabels_batch.append(masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_labels)
        # note you can not change here since it should be consistent with 'num_to_predict' in create_masked_lm_predictions
        num_to_predict = min(max_predictions_per_seq,
                              max(1, int(round(len(item_batch_[line_list]) * masked_lm_prob))))

        while len(masked_lm_weights) < num_to_predict:
            masked_lm_weights.append(0.0)
        masked_lm_weights_batch.append(masked_lm_weights)

    return output_tokens_batch, maskedpositions_batch, maskedlabels_batch, masked_lm_weights_batch

#from BERT
def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng, item_size):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        masked_token = None
        # 80% of the time, replace with [MASK]
        if rng.random() < 1.0:
            # masked_token = "[MASK]"
            masked_token=0  #item_size is "[MASK]"   0 represents '<unk>'
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)

def shuffleseq(train_set,padtoken):
    # create subsession only for training
    shuffle_seqtrain = []
    for i in range(len(train_set)):
        # print x_train[i]
        seq = train_set[i]
        lenseq = len(seq)
        # split=np.split(padtoken)
        copyseq=list(seq)
        padcount = copyseq.count(padtoken)  #the number of padding elements
        copyseq = copyseq[padcount:] # the remaining elements
        # copyseq=seq
        shuffle_indices = np.random.permutation(np.arange(len(copyseq)))
        # list to array
        copyseq= np.array(copyseq)
        copyseq_shuffle=copyseq[shuffle_indices]

        padtoken_list=[padtoken]*padcount
        # array to list, + means concat in list and  real plus in array
        seq=padtoken_list+list(copyseq_shuffle)
        shuffle_seqtrain.append(seq)


    x_train = np.array(shuffle_seqtrain)  # list to ndarray
    # Randomly shuffle data
    # np.random.seed(10)
    # shuffle_train = np.random.permutation(np.arange(len(x_train)))
    # x_train = x_train[shuffle_train]
    print("shuffling is done!")
    return x_train


def main():
    # df = pd.DataFrame(
    #     columns=['Batch', 'Train Loss', 'Test Loss', 'MRR_5', 'MRR_20', 'HR_5', 'HR_20', 'NDCG_5', 'NDCG_20'])
    # df.to_csv("result.csv", index=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=5,
                        help='Sample from top k predictions')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='hyperpara-Adam')
    parser.add_argument('--datapath', type=str, default='Data/Session/ratings_seq20_order.txt',
                        help='data path')
    parser.add_argument('--eval_iter', type=int, default=10,
                        help='Sample generator output every x steps')
    parser.add_argument('--save_para_every', type=int, default=10,
                        help='save model parameters every')
    parser.add_argument('--tt_percentage', type=float, default=0.2,
                        help='0.2 means 80%% training 20%% testing')
    parser.add_argument('--masked_lm_prob', type=float, default=0.5,
                        help='0.2 means 20%% items are masked')
    parser.add_argument('--max_predictions_per_seq', type=int, default=20,
                        help='maximum number of masked tokens')
    parser.add_argument('--max_position', type=int, default=100,
                        help='maximum number of for positional embedding, it has to be larger than the sequence lens')
    parser.add_argument('--has_positionalembedding', type=bool, default=False,
                        help='whether contains positional embedding before performing cnnn')
    parser.add_argument('--padtoken', type=str, default='-1',
                        help='is the padding token in the beggining of the sequence')
    parser.add_argument('--is_shuffle', type=bool, default=False,
                        help='whether shuffle the training and testing dataset, e.g., 012345-->051324')

    args = parser.parse_args()
    print(args)

    dl = Data_loader.Data_Loader({'model_type': 'generator', 'dir_name': args.datapath})
    all_samples = dl.items
    items_voc = dl.item2id  # key is the original token, value is the mapped value, i.e., 0, 1,2,3...
    itemlist = items_voc.values()
    item_size = len(items_voc)  # the first token is 'unk'
    print("shape: ", np.shape(all_samples))

    if args.padtoken in items_voc:
        padtoken = items_voc[args.padtoken]  # is the padding token in the beggining of the sentence
    else:
        padtoken = sys.maxsize
    max_predictions_per_seq = args.max_predictions_per_seq
    masked_lm_prob = args.masked_lm_prob

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(all_samples)))
    all_samples = all_samples[shuffle_indices]

    # Split train/test set
    dev_sample_index = -1 * int(args.tt_percentage * float(len(all_samples)))
    train_set, valid_set = all_samples[:dev_sample_index], all_samples[dev_sample_index:]

    if args.is_shuffle:
        train_set = shuffleseq(train_set, padtoken)

    model_para = {
        'item_size': item_size,
        'dilated_channels': 64,  # embedding dimension
        # if you use nextitnet_residual_block, you can use [1, 4, ],
        # if you use nextitnet_residual_block_one, you can tune and i suggest [1, 2, 4, ], for a trial
        'dilations': [1, 4, 1, 4, ],
        'kernel_size': 3,
        'learning_rate': 0.001,
        'batch_size': 128,  # a proper batch size helps a lot {e.g.,256, 512}
        'iterations': 40,
        'max_position': 100,
        # args.max_position,#maximum number of for positional embedding, it has to be larger than the sequence lens
        'has_positionalembedding': args.has_positionalembedding,
        'is_negsample': False,  # False denotes no negative sampling
        'top_k': args.top_k
    }
    print("dilations: ", model_para["dilations"])
    print("dilated_channels: ", model_para["dilated_channels"])
    print("batch_size: ", model_para["batch_size"])

    device = 'cpu'

    model = generator_grectorch.GRecArchi(model_para).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_para['learning_rate'], weight_decay=0)
    criterion = nn.CrossEntropyLoss()

    for i, (key, u) in enumerate(model.state_dict().items()):
        print(key, u.size())

    numIters = 1
    for iter in range(model_para['iterations']):
        model.train()
        batch_no = 0
        batch_size = model_para['batch_size']
        batch_num = train_set.shape[0] / batch_size
        while (batch_no + 1) * batch_size < train_set.shape[0]:
            start = time.time()
            item_batch = train_set[batch_no * batch_size: (batch_no + 1) * batch_size, :]
            # original input  1 2 3 4 5 6 7 8 9
            # item_batch[:,1:-1]   2 3 4 5 6 7 8
            # output_tokens_batch  2 0 4 5 0 7 8
            # maskedpositions_batch  [1 4]
            # maskedlabels_batch [3 6]
            output_tokens_batch, maskedpositions_batch, maskedlabels_batch, masked_lm_weights_batch = create_masked_lm_predictions_frombatch(
                item_batch, masked_lm_prob, max_predictions_per_seq, items=itemlist, item_size=item_size
            )
            optimizer.zero_grad()
            outputs = model(torch.LongTensor(output_tokens_batch).to(device), torch.LongTensor(item_batch).to(device), torch.LongTensor(maskedpositions_batch).to(device))  # [batch_size * masked_length, item_size)

            label_flat = torch.LongTensor(maskedlabels_batch).view(-1)
            loss = criterion(outputs, label_flat.to(device))
            l2_loss = torch.mean(torch.Tensor([torch.norm(param, 2) ** 2 / 2 for param in model.parameters()]))
            loss += 1e-3 * l2_loss

            loss.backward()
            optimizer.step()
            end = time.time()

            train_loss = loss.item()
            if numIters % args.eval_iter == 0:
                # result = np.zeros(9)
                # result[0:2] = numIters, train_loss
                print("-------------------------------------------------------train")
                print("LOSS: {}\tITER: {}\tBATCH_NO: {}\t STEP:{}\t total_batches:{}".format(
                    train_loss, iter, batch_no, numIters, batch_num))
                print("TIME FOR BATCH", end - start)

            if numIters % args.eval_iter == 0:
                model.eval()
                print("-------------------------------------------------------test")
                batch_no_valid=0
                batch_size_valid = batch_size
                with torch.no_grad():
                    if (batch_no_valid + 1) * batch_size_valid < valid_set.shape[0]:
                        start = time.time()
                        item_batch = valid_set[(batch_no_valid) * batch_size_valid: (batch_no_valid + 1) * batch_size_valid, :]
                        output_tokens_batch, maskedpositions_batch, maskedlabels_batch, masked_lm_weights_batch = create_masked_lm_predictions_frombatch(
                            item_batch, masked_lm_prob, max_predictions_per_seq, items=itemlist, item_size=item_size
                        )

                        outputs = model(torch.LongTensor(output_tokens_batch).to(device),
                                        torch.LongTensor(item_batch).to(device),
                                        torch.LongTensor(maskedpositions_batch).to(
                                            device))  # [batch_size * masked_length, item_size)

                        label_flat = torch.LongTensor(maskedlabels_batch).view(-1)
                        loss = criterion(outputs, label_flat.to(device))
                        l2_loss = torch.mean(torch.Tensor([torch.norm(param, 2) ** 2 / 2 for param in model.parameters()]))
                        loss += 1e-3 * l2_loss
                        end = time.time()
                        test_loss = loss.item()

                        # result[2] = test_loss
                        print("LOSS: {}\tITER: {}\tBATCH_NO: {}\t STEP:{}\t total_batches:{}".format(
                            test_loss, iter, batch_no_valid, numIters, valid_set.shape[0] / batch_size_valid))
                        print("TIME FOR BATCH", end - start)
            batch_no += 1

            if numIters % args.eval_iter == 0:  # compute test loss on the whole validset every arg.eval_iter train batches
                model.eval()
                batch_no_test = 0
                batch_size_test = batch_size * 1
                curr_preds_5 = []
                rec_preds_5 = []  # 1
                ndcg_preds_5 = []  # 1
                curr_preds_20 = []
                rec_preds_20 = []  # 1
                ndcg_preds_20 = []  # 1
                # acc = np.zeros(6)
                with torch.no_grad():
                    while (batch_no_test + 1) * batch_size_test < valid_set.shape[0]:

                        item_batch = valid_set[batch_no_test * batch_size_test: (batch_no_test + 1) * batch_size_test, :]
                        probs = model(torch.LongTensor(item_batch[:, 0:-1]).to(device),
                                      torch.LongTensor(item_batch[:, 0:-1]).to(device), test=True)

                        _, pred_items_5 = torch.topk(probs, k=args.top_k, sorted=True)  # top_k=5
                        _, pred_items_20 = torch.topk(probs, k=args.top_k + 15, sorted=True)
                        pred_items_5 = pred_items_5.data.cpu().numpy()
                        pred_items_20 = pred_items_20.data.cpu().numpy()

                        for bi in range(pred_items_5.shape[0]):
                            true_item = item_batch[bi][-1]

                            predictmap_5 = {ch: i for i, ch in enumerate(pred_items_5[bi])}
                            predictmap_20 = {ch: i for i, ch in enumerate(pred_items_20[bi])}

                            rank_5 = predictmap_5.get(true_item)
                            rank_20 = predictmap_20.get(true_item)
                            if rank_5 == None:
                                curr_preds_5.append(0.0)
                                rec_preds_5.append(0.0)  # 2
                                ndcg_preds_5.append(0.0)  # 2
                            else:
                                MRR_5 = 1.0 / (rank_5 + 1)
                                Rec_5 = 1.0  # 3
                                ndcg_5 = 1.0 / math.log(rank_5 + 2, 2)  # 3
                                curr_preds_5.append(MRR_5)
                                rec_preds_5.append(Rec_5)  # 4
                                ndcg_preds_5.append(ndcg_5)  # 4
                            if rank_20 == None:
                                curr_preds_20.append(0.0)
                                rec_preds_20.append(0.0)  # 2
                                ndcg_preds_20.append(0.0)  # 2
                            else:
                                MRR_20 = 1.0 / (rank_20 + 1)
                                Rec_20 = 1.0  # 3
                                ndcg_20 = 1.0 / math.log(rank_20 + 2, 2)  # 3
                                curr_preds_20.append(MRR_20)
                                rec_preds_20.append(Rec_20)  # 4
                                ndcg_preds_20.append(ndcg_20)  # 4

                        batch_no_test += 1
                        # mrr_5 = sum(curr_preds_5) / float(len(curr_preds_5))
                        # mrr_20 = sum(curr_preds_20) / float(len(curr_preds_20))
                        # hit_5 = sum(rec_preds_5) / float(len(rec_preds_5))
                        # hit_20 = sum(rec_preds_20) / float(len(rec_preds_20))
                        # ndcg_5 = sum(ndcg_preds_5) / float(len(ndcg_preds_5))
                        # ndcg_20 = sum(ndcg_preds_20) / float(len(ndcg_preds_20))
                        # acc += np.array([mrr_5, mrr_20, hit_5, hit_20, ndcg_5, ndcg_20])

                        if batch_no_test == 10:
                            print("BATCH_NO: {}".format(batch_no_test))
                            print("mrr_5:", sum(curr_preds_5) / float(len(curr_preds_5)), "mrr_20:", sum(
                                curr_preds_20) / float(len(curr_preds_20)), "hit_5:", sum(rec_preds_5) / float(
                                len(rec_preds_5)), "hit_20:", sum(rec_preds_20) / float(
                                len(rec_preds_20)), "ndcg_5:", sum(ndcg_preds_5) / float(
                                len(ndcg_preds_5)), "ndcg_20:", sum(ndcg_preds_20) / float(len(ndcg_preds_20)))
                # acc /= batch_no_test
                # result[3:] = acc
                # result = pd.DataFrame(result.reshape(1, -1))
                # result.to_csv('result.csv', mode='a', header=False, index=False)

            numIters += 1


if __name__=='__main__':
    main()

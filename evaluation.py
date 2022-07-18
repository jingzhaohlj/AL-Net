import numpy as np
import utils.surface_distance as metrics

def instance_AJI(true_instance,pred_instance,mask):
    true_masks = np.copy(true_instance)
    pred_masks = np.copy(pred_instance)
    pred_id_list=[]
    true_id_list=[]
 
    for i in range(0,len(pred_instance)):
        pred_id_list.append(i)
    
    for f in range(0,len(true_instance)):
        true_id_list.append(f)
    if len(pred_id_list)==1:
        aji_score=0.
        # print("111111")
        return aji_score

    pairwise_inter = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )
    pairwise_union = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )
    # return pred_id_list[1]

    for pred_id in pred_id_list[1:]:
        p_mask=pred_masks[pred_id]
        pred_true_overlap=mask[p_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for true_id in pred_true_overlap_id :
            if true_id == 0:  # ignore
                continue
            t_mask = true_masks[true_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter
    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    # print(paired_true.shape, paired_pred.shape)
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()
    # return overall_inter,overall_union

    paired_true = list(paired_true + 1)  # index to instance ID
    paired_pred = list(paired_pred + 1)
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array(
        [idx for idx in true_id_list[1:] if idx not in paired_true]
    )
    unpaired_pred = np.array(
        [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    )
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()

    aji_score = overall_inter / overall_union



    return aji_score
    # return 
def instance_HD(true_instance,pred_instance,mask):
    true_masks = np.copy(true_instance)
    pred_masks = np.copy(pred_instance)
    pred_id_list=[]
    true_id_list=[]
 
    for i in range(0,len(pred_instance)):
        pred_id_list.append(i)
    
    for f in range(0,len(true_instance)):
        true_id_list.append(f)

    if len(pred_id_list)==1:
        aji_score=0.
        # print("111111")
        return aji_score
    HD_score_list=[]
    for pred_id in pred_id_list[1:]:
        p_mask=pred_masks[pred_id]
        pred_true_overlap=mask[p_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        HD_list=[]
        for true_id in pred_true_overlap_id :
            if true_id == 0 and len(pred_true_overlap_id)==1:  # ignore
                # continue
                pred_2D_bool=p_mask>0
                mask_zeros = np.zeros([pred_2D_bool.shape[0], pred_2D_bool.shape[1]], dtype=pred_2D_bool.dtype)
                mask_bool=mask_zeros>0
                surface_distances = metrics.compute_surface_distances(mask_bool, pred_2D_bool, spacing_mm=(1.0, 1.0))
                HD=metrics.compute_robust_hausdorff(surface_distances, 95)
                # print(HD)
                continue
            elif true_id == 0 and len(pred_true_overlap_id)!=1:
                continue
            t_mask = true_masks[true_id]
            mask_bool=t_mask>0
            pred_2D_bool=p_mask>0
            surface_distances = metrics.compute_surface_distances(mask_bool, pred_2D_bool, spacing_mm=(1.0, 1.0))
            HD=metrics.compute_robust_hausdorff(surface_distances, 95)
            HD_list.append(HD)
        if len(HD_list)!=0:
            HD_score_list.append(min(HD_list))
    if (len(HD_score_list)!=0):
        HD_score=max(HD_score_list)
    else:
        HD_score=0
    return HD_score
def instance_pq(true_instance,pred_instance,mask,match_iou=0.5):
    assert match_iou >= 0.0, "Cant' be negative"
    true_masks = np.copy(true_instance)
    pred_masks = np.copy(pred_instance)
    pred_id_list=[]
    true_id_list=[]
 
    for i in range(0,len(pred_instance)):
        pred_id_list.append(i)
    
    for f in range(0,len(true_instance)):
        true_id_list.append(f)
    if len(pred_id_list)==1:
        PQ=0.
        # print("111111")
        return PQ
    # prefill with value
    pairwise_iou = np.zeros([len(true_id_list) -1, 
                             len(pred_id_list) -1], dtype=np.float64)
    # caching pairwise iou
    for pred_id in pred_id_list[1:]:
        p_mask=pred_masks[pred_id]
        pred_true_overlap=mask[p_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for true_id in pred_true_overlap_id :
            if true_id == 0:  # ignore
                continue
            t_mask = true_masks[true_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id-1, pred_id-1] = iou
    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1 # index is instance id - 1
        paired_pred += 1 # hence return back to original

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]

    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn+ 1.0e-6)
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)
    # return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]
    return dq * sq

def instance_Dice(true_instance,pred_instance,mask):
    true_masks = np.copy(true_instance)
    pred_masks = np.copy(pred_instance)
    pred_id_list=[]
    true_id_list=[]
 
    for i in range(0,len(pred_instance)):
        pred_id_list.append(i)
    
    for f in range(0,len(true_instance)):
        true_id_list.append(f)
    if len(pred_id_list)==1:
        aji_score=0.
        return aji_score

    pairwise_inter = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )
    pairwise_union = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )
    for pred_id in pred_id_list[1:]:
        p_mask=pred_masks[pred_id]
        pred_true_overlap=mask[p_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for true_id in pred_true_overlap_id :
            if true_id == 0:  # ignore
                continue
            t_mask = true_masks[true_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total
    pairwise_iou = 2*pairwise_inter / (pairwise_union + 1.0e-6)
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    # print(paired_true.shape, paired_pred.shape)
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()
    # return overall_inter,overall_union

    paired_true = list(paired_true + 1)  # index to instance ID
    paired_pred = list(paired_pred + 1)
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array(
        [idx for idx in true_id_list[1:] if idx not in paired_true]
    )
    unpaired_pred = np.array(
        [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    )
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()

    dice_score = 2*overall_inter / overall_union

    return dice_score
def Dice(mask,pred):
     # ? do we need this
    mask = np.copy(mask)
    pred = np.copy(pred)
    inter = (mask * pred).sum()
    total = mask.sum() + pred.sum()
    dice_score=(2*inter)/total
    return dice_score
def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] 
    not [0, 2, 4, 6]. The ordering of instances (which one comes first) 
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID.

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)

    """
    # print(pred.shape)
    # pred = torch.argmax(torch.from_numpy(pred), 1) 
    # pred = np.expand_dims(pred.numpy(), axis=1)

    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    # print(list(np.unique(new_pred)))
    return new_pred
def get_fast_aji(true, pred):
    """AJI version distributed by MoNuSeg, has no permutation problem but suffered from 
    over-penalisation similar to DICE2.

    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4] 
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no 
    effect on the result.

    """
    true = np.copy(true)  # ? do we need this
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))
    if len(pred_id_list)==1:
        aji_score=0.
        # print("111111")
        return aji_score

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_inter = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )
    pairwise_union = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise
    for true_id in true_id_list[1:]:  # 0-th is background
        if true_id>=len(true_masks):
            continue
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter

    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    # pair of pred that give highest iou for each true, dont care
    # about reusing pred instance multiple times
    # print(pairwise_iou)
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    # print(paired_true.shape, paired_pred.shape)
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()

    paired_true = list(paired_true + 1)  # index to instance ID
    paired_pred = list(paired_pred + 1)
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array(
        [idx for idx in true_id_list[1:] if idx not in paired_true]
    )
    unpaired_pred = np.array(
        [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    )
    for true_id in unpaired_true:
        if true_id>=len(true_masks):
            continue
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        if pred_id>=len(pred_masks):
            continue
        overall_union += pred_masks[pred_id].sum()

    aji_score = overall_inter / overall_union
    return aji_score
def get_fast_pq(true, pred, match_iou=0.5):
    """
    `match_iou` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique 
    (1 prediction instance to 1 GT instance mapping).

    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing. 

    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.    
    
    Fast computation requires instance IDs are in contiguous orderding 
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand 
    and `by_size` flag has no effect on the result.

    Returns:
        [dq, sq, pq]: measurement statistic

        [paired_true, paired_pred, unpaired_true, unpaired_pred]: 
                      pairing information to perform measurement
                    """
    assert match_iou >= 0.0, "Cant' be negative"
    # true = label(true, background=0)
    # pred = label(pred, background=0)
    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))
    if len(pred_id_list)==1:
        PQ=0.
        # print("111111")
        return PQ

    true_masks = [None,]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)
    
    pred_masks = [None,]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_iou = np.zeros([len(true_id_list) -1, 
                             len(pred_id_list) -1], dtype=np.float64)

    # caching pairwise iou
    for true_id in true_id_list[1:]: # 0-th is background
        if true_id>=len(true_masks):
            continue
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0: # ignore
                continue # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id-1, pred_id-1] = iou
    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1 # index is instance id - 1
        paired_pred += 1 # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence 
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum   
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair 
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn+ 1.0e-6)
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)
    # print(dq * sq)

    # return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]
    return dq * sq
from tqdm import tqdm 

import torch 
import torcheval.metrics as metrics

import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.metrics import RocCurveDisplay


# Accuracy 
def metric_accuracy(confmat): 
    '''
    IN: 
    confmat: a list of confusion matrices for one class represented as a dict with keys 'tp', 'tn', 'fp', 'fn'
    
    OUT: 
    acc: a list of the accuracy = number of correct predictions / number of elements per class 
    '''
    acc = []
    
    for i in range(len(confmat)): 
        acc.append(confmat[i]['tp'] / (confmat[i]['tp'] + confmat[i]['tn'] + confmat[i]['fp'] + confmat[i]['fn']))

    return acc 

# Precision 
def metric_precision(confmat): 
    '''
    IN: 
    confmat: a list of confusion matrices for one class represented as a dict with keys 'tp', 'tn', 'fp', 'fn'
    
    OUT: 
    precision: a list of the the precision tp / (tp + fp) per class 
    '''
    precision = []

    for i in range(len(confmat)): 
        if confmat[i]['tp'] + confmat[i]['fp'] == 0: 
            precision.append(1) 
        else: 
            precision.append(confmat[i]['tp'] / (confmat[i]['tp'] + confmat[i]['fp']))
    
    return precision

# Negative Predictive Value  
def metric_npv(confmat): 
    '''
    IN: 
    confmat: a list of confusion matrices for one class represented as a dict with keys 'tp', 'tn', 'fp', 'fn'

    OUT: 
    specificity: the negative predictive value tn / (tn + fn)
    '''
    npv = []

    for i in range(len(confmat)): 
        if confmat[i]['tn'] + confmat[i]['fn'] == 0: 
            npv.append(1) 
        else: 
            npv.append(confmat[i]['tn'] / (confmat[i]['tn'] + confmat[i]['fn']))
    
    return npv


# Recall / Sensitivity 
def metric_recall(confmat): 
    '''
    IN: 
    confmat: a list of confusion matrices for one class represented as a dict with keys 'tp', 'tn', 'fp', 'fn'

    OUT: 
    recall: the recall tp / (tp + fn)
    '''
    recall = []

    for i in range(len(confmat)): 
        if confmat[i]['tp'] + confmat[i]['fn'] == 0: 
            recall.append(1) 
        else: 
            recall.append(confmat[i]['tp'] / (confmat[i]['tp'] + confmat[i]['fn']))
    
    return recall

# Specificity 
def metric_specificity(confmat): 
    '''
    IN: 
    confmat: a list of confusion matrices for one class represented as a dict with keys 'tp', 'tn', 'fp', 'fn'

    OUT: 
    specificity: the specificity tn / (tn + fp)
    '''
    specificity = []

    for i in range(len(confmat)): 
        if confmat[i]['tn'] + confmat[i]['fp'] == 0: 
            specificity.append(1) 
        else: 
            specificity.append(confmat[i]['tn'] / (confmat[i]['tn'] + confmat[i]['fp']))
    
    return specificity

# F1-score 
def metric_f1_score(confmat): 
    '''
    IN: 
    confmat: a list of confusion matrices for one class represented as a dict with keys 'tp', 'tn', 'fp', 'fn'
    
    OUT: 
    f1-score: a list containing the scores per classes. IF n_classes == 2, the list only contains a single element \
              which is the F1-score for the binary classification 
    '''
    f1_score = [] 
    precs = metric_precision(confmat)
    recalls = metric_recall(confmat)
    for class_id in range(len(confmat)): 
        f1_score_class = 0 if precs[class_id]+recalls[class_id] == 0 else 2 * precs[class_id] * recalls[class_id] / (precs[class_id] + recalls[class_id])
        f1_score.append(f1_score_class)
    return f1_score

# Main test function 
def test(net, 
         device,
         dataloader, 
         n_classes, 
         threshold = 0.5, 
         binaryclass = True): 
    '''
    IN: 
    n_classes: number of classes to be classified 

    OUT: 
    metrics: A dictionary of the metrics we want: accuracy, precision, recall, F1-score, AUC (optional)
    '''
    
    net.eval() # Set model in evaluate mode 

    # 1. Initialize all metrics at 0 
    acc = 0 
    if n_classes == 2: 
        auc_metric = metrics.BinaryAUROC()
    else: 
        auc_metric = metrics.MulticlassAUROC(num_classes=n_classes)
    # f1_score = [0 for i in range(n_classes)] if n_classes > 2 else [0] 
    confusion_mat = [{} for i in range(n_classes)] if n_classes > 2 else [{}]  
    for i in range(len(confusion_mat)): 
        confusion_mat[i]['tp'] = 0
        confusion_mat[i]['fp'] = 0
        confusion_mat[i]['tn'] = 0
        confusion_mat[i]['fn'] = 0

    all_gt = []
    all_probs = []

    # 2. Begin batches loop 
    print('-' * 10)
    with torch.set_grad_enabled(False) and tqdm(total=len(dataloader.dataset), desc=f'Testing in progression', unit='img') as pbar:
        for batch in dataloader: 
            inputs = batch['input'].to(device=device)
            gt = batch['label'].to(device=device)

            # 3. Compute predictions                         
            outputs = net(inputs)
            if n_classes > 2: 
                preds = torch.argmax(outputs, dim=1) 
            else: 
                probs = torch.sigmoid(outputs).flatten()
                preds = torch.where(probs > threshold, 1, 0)

                # For ROC curve 
                all_gt += list(gt.cpu())
                all_probs += list(probs.cpu())

            # 4. Compute scores 
            acc += torch.sum(gt == preds.data)
            if n_classes > 2: 
                probs = torch.nn.functional.softmax(outputs, dim=1, dtype=torch.float)
            auc_metric.update(input=probs, target=gt)

            # 5. Compute confusion matrices 
            if n_classes > 2: 
                for class_id in range(n_classes): 
                    confusion_mat[class_id]['tp'] += torch.sum((gt==preds)*(gt==class_id))
                    confusion_mat[class_id]['fp'] += torch.sum((gt!=preds)*(gt!=class_id))
                    confusion_mat[class_id]['tn'] += torch.sum((gt==preds)*(gt!=class_id))
                    confusion_mat[class_id]['fn'] += torch.sum((gt!=preds)*(gt==class_id))
            else: 
                confusion_mat[0]['tp'] += torch.sum((gt==preds)*(preds==1))
                confusion_mat[0]['fp'] += torch.sum((gt!=preds)*(preds==1))
                confusion_mat[0]['tn'] += torch.sum((gt==preds)*(preds==0))
                confusion_mat[0]['fn'] += torch.sum((gt!=preds)*(preds==0))
                
            # Update progress bar (tqdm) at the end of batch process 
            pbar.update(inputs.size(0))
        
    # 6. Normalize the metrics 
    acc = float(acc)/len(dataloader.dataset)

    # 7. Calculate the metrics based on confusion matrices and add them to return value 
    scores = {} 
    
    precision = metric_precision(confmat=confusion_mat)
    npv = metric_npv(confmat=confusion_mat)
    recall = metric_recall(confmat=confusion_mat)
    specificity = metric_specificity(confmat=confusion_mat)
    f1_score = metric_f1_score(confmat=confusion_mat)
    scores['accuracy'] = acc
    scores['precision'] = precision
    scores['npv'] = npv
    scores['recall'] = recall 
    scores['specificity'] = specificity 
    scores['f1_score'] = f1_score 

    if n_classes > 2: 
        avg_acc = sum(metric_accuracy(confmat=confusion_mat))/n_classes 
        avg_prec = sum(precision)/n_classes 
        avg_npv = sum(npv)/n_classes
        avg_rec = sum(recall)/n_classes 
        avg_spec = sum(specificity)/n_classes 
        avg_f1 = sum(f1_score)/n_classes
    
        scores['average accuracy'] = avg_acc 
        scores['average precision'] = avg_prec 
        scores['average npv'] = avg_npv
        scores['average recall'] = avg_rec 
        scores['average specificity'] = avg_spec 
        scores['average f1 score'] = avg_f1 
    
    scores['confusion_matrices'] = confusion_mat
    scores['auc'] = auc_metric.compute()

    if n_classes == 2: 
        all_gt = np.array(all_gt).flatten()
        print(all_probs)
        all_probs = np.array(all_probs).flatten()
        RocCurveDisplay.from_predictions(y_true=all_gt, y_pred=all_probs)
        plt.show()

    return scores
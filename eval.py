import os
import csv
import torch
from networks.trainer import Patch5Model
from networks.resnet import resnet50
from options.test_options import TestOptions
from eval_config import *
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, roc_curve, auc
import sys
sys.path.append('./data')
from data import create_dataloader_test
import numpy as np
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
 
def validate(model, data_loader):
    i = 0
    with torch.no_grad():
        y_true, y_pred = [], [] 
        for data in data_loader:
            i += 1
            print("batch number {}/{}".format(i, len(data_loader)))
            input_img = data[0] #[batch_size, 3, height, width]
            cropped_img = data[1].cuda() #[batch_size, 3, 224, 224]
            label = data[2].cuda() #[batch_size, 1]
            scale = data[3].cuda() #[batch_size, 1, 2]

            logits = model(input_img, cropped_img, scale)
            y_pred.extend(logits.sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())


    y_true, y_pred = np.array(y_true), np.array(y_pred)
    oa = accuracy_score(y_true, y_pred > 0.5)

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    ap = average_precision_score(y_true, y_pred)
    return oa, roc_auc, ap


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)
    model_name = os.path.basename(model_path).replace('.pth', '')
    rows = [["{} model testing on...".format(model_name)],
        ['testmodel', 'oa', 'auc', 'ap']]

    model = Patch5Model()
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict['model']) # use this if testing model is trained on single GPU
    ## uncomment following lines if testing model is trained on multiple GPUs
    #from collections import OrderedDict
    #new_state_dict = OrderedDict()
    #for k, v in state_dict['model'].items():
    #    name = k[7:] # remove `module.`
    #    new_state_dict[name] = v
    #model.load_state_dict(new_state_dict)
    
    model.cuda()
    model.eval()
       
    for v_id, val in enumerate(vals):
        print("testing {}-generated images".format(val))
        opt.dataroot = '{}/{}'.format(dataroot, val)
        opt.no_resize = True    # testing without resizing by default
        data_loader = create_dataloader_test(opt)        
        oa, roc_auc, ap = validate(model, data_loader)
        rows.append([val, oa, roc_auc, ap])
        print("oa: {}; auc: {}; ap:{}".format(oa, roc_auc, ap))

        
    csv_name = results_dir + '/{}.csv'.format(model_name)
    with open(csv_name, 'a') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerows(rows)

from data_set.aff40k_dataset import aff40k_dataset
from data_set.aff_dataset import aff_dataset
from data_set.fer_dataset import fer2013_dataset
from data_set.fer_plus_dataset import fer2013_plus_dataset
from data_set.raf_dataset import raf_dataset
from data_set.sfew_dataset import sfew_dataloder

def return_dataset(name):
    if name=='aff40k':
        train_dataset=aff40k_dataset('./data/affectnet/update/train.csv')
        test_dataset=aff40k_dataset('./data/affectnet/update/test.csv')
    if name=='fer2013':
        train_dataset=fer2013_dataset('/home/wpy/CLIP4emo/data/fer2013/train_labels_new.csv','/home/wpy/CLIP4emo/data/fer2013/train_landmarks_data.npy','train')
        test_dataset=fer2013_dataset('/home/wpy/CLIP4emo/data/fer2013/test_labels_new.csv','/home/wpy/CLIP4emo/data/fer2013/test_landmarks_data.npy','test')
    if name=='fer_plus':
        train_dataset=fer2013_plus_dataset('./data/fer2013plus/train.csv')
        test_dataset=fer2013_plus_dataset('./data/fer2013plus/test.csv')
    if name=='raf':
        train_dataset=raf_dataset('train','./data/rafdb/train_labels_new.csv',"./data/rafdb/train_landmarks_data.npy")
        #test_dataset=raf_dataset('train','./data/rafdb/dataset/train_labels.csv')
        test_dataset=raf_dataset('test','./data/rafdb/test_labels_new.csv',"./data/rafdb/test_landmarks_data.npy")
    if name=='sfew':
        train_dataset=sfew_dataloder('./data/sfew/train.csv','train')
        test_dataset=sfew_dataloder('./data/sfew/val.csv','dev')

    return train_dataset,test_dataset

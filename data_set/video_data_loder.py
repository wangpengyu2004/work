from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import sys
import os
from torch.utils.data import Dataset
import numpy as np
import json
import ast
from proprecess.rawvideo_util import RawVideoExtractor
import pandas as pd
from model_all.clip4video.simple_tokenizer  import SimpleTokenizer
class meld_DataLoader(Dataset):
    def __init__(
            self,
            subset,
            data_path,
            video_features_path,
            tokenizer,
            max_words=77,
            feature_framerate=3.0,
            max_frames=5,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        self.video_features_path=video_features_path
        self.data_path = data_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]
    
        self.subset = subset
        assert self.subset in ["train", "test"]
        
        datapath_dict = {}
        datapath_dict["train"] = os.path.join(self.data_path, "updatatrain.csv")
        datapath_dict["test"] = os.path.join(self.data_path, "updatatest.csv")

        #self.video_feature_path=os.path.join(self.video_features_path,"videonpy")
        #video_mask_feature_path=os.path.join(self.video_features_path,"video_mask.npz")
        #self.video_mask_features=np.load(video_mask_feature_path)
        #self.data_csv=pd.read_csv(datapath_dict[self.subset]) 
        self.data_csv=pd.read_csv(datapath_dict[self.subset]) 
        #self.text=self.data_csv['Utterance'].to_list()
        self.emo=self.data_csv['label'].to_list()
        self.video_path=self.data_csv['path'].to_list()
        self.rawVideoExtractor = RawVideoExtractor(size=image_resolution,framerate=feature_framerate )
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        
    def __len__(self):
        return len(self.data_csv)

    def _get_text_train(self,sent):
        label=["collision with motorcycle","collision with stationary object","drifting or skidding","fire or explosions","head on collision","negative samples","objects falling","other crash","pedestrian hit","rear collision","rollover","side collision"]
        #得到情感prompt
        emo_name=[i for i in label if i!=sent]
        emo_str=','.join(emo_name)
        #input_text=[f"this photo that evokes {sent} , but not evokes {emo_str}",text]
        input_text=[f"the emotion is {sent}"]
        k = len(input_text)
        pairs_text = np.zeros((k, self.max_words), dtype=np.int32)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int32)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int32)
        
        for i in range(k):
            words = self.tokenizer(input_text[i])[0]
            words=words.tolist()
            #input_ids = simple_tokenizer.convert_tokens_to_ids(words[0])
            input_ids=[i for i in words if i!=0]
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)
        return pairs_text, pairs_mask, pairs_segment,k
    def _get_text_test(self):
        label=["collision with motorcycle","collision with stationary object","drifting or skidding","fire or explosions","head on collision","negative samples","objects falling","other crash","pedestrian hit","rear collision","rollover","side collision"]
        #得到情感prompt
        input_text=[]
        for emo in label:
            emo_name=[i for i in label if i!=emo]
            emo_str=','.join(emo_name)
            #input_text.append(f"this photo that evokes {emo} , but not evokes {emo_str}")
            input_text.append(f"the emotion is {emo}")
        k = len(input_text)
        pairs_text = np.zeros((k, self.max_words), dtype=np.int32)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int32)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int32)
        
        for i in range(k):
            words = self.tokenizer(input_text[i])[0]
            words=words.tolist()
            #input_ids = simple_tokenizer.convert_tokens_to_ids(words[0])
            input_ids=[i for i in words if i!=0]
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)
        return pairs_text, pairs_mask, pairs_segment,k

    def _get_rawvideo(self, video_path):
        j=1
        video_mask = np.zeros((j, self.max_frames), dtype=np.int32)
        max_video_length = [0] * j

        # Pair x L x T x 3 x H x W
        video = np.zeros((j, self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float32)
        try:
            for i in range(j):
                raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
                raw_video_data = raw_video_data['video']
              
                if len(raw_video_data.shape) > 3:
                    raw_video_data_clip = raw_video_data
                    # L x T x 3 x H x W
                    raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                    if self.max_frames < raw_video_slice.shape[0]:
                        if self.slice_framepos == 0:
                            video_slice = raw_video_slice[:self.max_frames, ...]
                        elif self.slice_framepos == 1:
                            video_slice = raw_video_slice[-self.max_frames:, ...]
                        else:
                            sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                            video_slice = raw_video_slice[sample_indx, ...]
                    else:
                        video_slice = raw_video_slice

                    video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                    slice_len = video_slice.shape[0]
                    max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                    if slice_len < 1:
                        pass
                    else:
                        video[i][:slice_len, ...] = video_slice
                else:
                    print("video path: {} error".format(video_path))
        except Exception as excep:
            print("video path: {} error.Error: {}".format(video_path,excep))
            pass
            # raise e

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length  
        return video, video_mask
    
     
    def __getitem__(self,index):
        video, video_mask = self._get_rawvideo(self.video_path[index])
        if self.subset=='train':
            pairs_text, pairs_mask, pairs_segment,k = self._get_text_train(self.emo[index])
        else:
            pairs_text, pairs_mask, pairs_segment,k = self._get_text_test()
            video = np.tile(video, (k,1,1,1,1,1))
            video_mask = np.tile(video_mask, (k,1))
        return pairs_text, pairs_mask, pairs_segment, k,video, video_mask,self.emo[index]
    
    def down_vdata(self,path):
        i=0
        df = pd.DataFrame(columns=['video', 'video_mask'])
        video_list={}
        video_mask_list={}
        for i in range(len(self.video_path)):
            video, video_mask = self._get_rawvideo(self.video_path[i])
            #video_list[f'video{i}'] = video
            video_mask_list[f"video_mask{i}"]=video_mask
            print(i)
            np.savez_compressed(f'{path}/video/videonpy/video{i}.npz', video)
        #np.savez_compressed(f'{path}/video/video_mask.npz', video)
        print("finish")


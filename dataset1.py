import os

# from tqdm import tqdm

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

from data import video_transforms as transforms
from data import video_sampler as sampler


class VideoDataset(Dataset):

    def __init__(self, dataset='ucf101', split='train', clip_len=16, preprocess=False):

        self.root_dir = '/home/yanxinyi/yxy/tsn/videos/input/HMDB51/hmdb51_org'
        self.output_dir = '/home/yanxinyi/yxy/semi-mfnet/video/hmdb51_frame'


        folder = os.path.join(self.output_dir, split)
        self.clip_len = clip_len
        self.split = split

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 240
        self.resize_width = 320
        self.crop_size = 224

        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')

        if preprocess:
            print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
            self.preprocess()

        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)

        assert len(labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        # if dataset == "ucf101":
        #     if not os.path.exists('/home/yanxinyi/yxy/pytorch-video-recognition/dataloaders/ucf_labels.txt'):
        #         with open('/home/yanxinyi/yxy/pytorch-video-recognition/dataloaders/ucf_labels.txt', 'w') as f:
        #             for id, label in enumerate(sorted(self.label2index)):
        #                 f.writelines(str(id+1) + ' ' + label + '\n')
        #
        # elif dataset == 'hmdb51':
        #     if not os.path.exists('/home/yanxinyi/yxy/pytorch-video-recognition/dataloaders/ucf_labels.txt'):
        #         with open('/home/yanxinyi/yxy/pytorch-video-recognition/dataloaders/hmdb_labels.txt', 'w') as f:
        #             for id, label in enumerate(sorted(self.label2index)):
        #                 f.writelines(str(id+1) + ' ' + label + '\n')


    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer = self.load_frames(self.fnames[index])
        buffer = self.crop_video_time(buffer, self.clip_len)
        buffer = np.transpose(buffer, [1, 2, 0, 3])
        buffer = buffer.reshape([buffer.shape[0], buffer.shape[1], buffer.shape[2] * buffer.shape[3]])
        buffer = self.transform(buffer)
        return buffer, self.label_array[index]

    def check_integrity(self):
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True

    def check_preprocess(self):
        # TODO: Check image size in output_dir
        if not os.path.exists(self.output_dir):
            return False
        elif not os.path.exists(os.path.join(self.output_dir, 'train')):
            return False

        for ii, video_class in enumerate(os.listdir(os.path.join(self.output_dir, 'train'))):
            for video in os.listdir(os.path.join(self.output_dir, 'train', video_class)):
                video_name = os.path.join(os.path.join(self.output_dir, 'train', video_class, video),
                                    sorted(os.listdir(os.path.join(self.output_dir, 'train', video_class, video)))[0])
                image = cv2.imread(video_name)
                if np.shape(image)[0] != 128 or np.shape(image)[1] != 171:
                    return False
                else:
                    break

            if ii == 10:
                break

        return True

    def transform(self, buffer):

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        video_transform = transforms.Compose([
            transforms.RandomScale(make_square=True,
                                   aspect_ratio=[0.8, 1. / 0.8],
                                   slen=[224, 288]),
            transforms.RandomCrop((224, 224)),  # insert a resize if needed
            transforms.RandomHorizontalFlip(),
            transforms.RandomHLS(vars=[15, 35, 25]),
            transforms.ToTensor(),
            normalize,
        ],
            aug_seed=(0 + 1))

        buffer = video_transform(buffer)
        return buffer

    def preprocess(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, 'train'))
            os.mkdir(os.path.join(self.output_dir, 'val'))

        # Split train/val/test sets
        for file in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, file)
            video_files = [name for name in os.listdir(file_path)]

            train = []
            val = []
            split_file = open('/home/yanxinyi/yxy/tsn/videos/input/HMDB51/hmdb51_splits/{}_test_split1.txt'.format(file))
            split_videos = split_file.readlines()
            for video in split_videos:
                t1 = video.strip()
                t1 = t1.split(' ')
                if int(t1[1]) == 1:
                    train.append(t1[0])
                if int(t1[1]) == 2:
                    val.append(t1[0])

            train_dir = os.path.join(self.output_dir, 'train', file)
            val_dir = os.path.join(self.output_dir, 'val', file)

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)

            for video in train:
                self.process_video(video, file, train_dir)
            print('\ntrain process video is done !')

            for video in val:
                self.process_video(video, file, val_dir)
            print('val process video is done !')

        print('Preprocessing finished.')

    def process_video(self, video, action_name, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Make sure splited video has at least 16 frames
        EXTRACT_FREQUENCY = 4
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= 16:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % EXTRACT_FREQUENCY == 0:
                # if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                #     frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer


    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            # frame -= np.array([[[90.0, 98.0, 102.0]]])
            # frame = frame - 128
            frame = frame / 255.0
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)

        # train_sampler = sampler.RandomSampling(num=self.clip_len,
        #                                        interval=2,
        #                                        speed=[1.0, 1.0],
        #                                        seed=(0))
        # count = train_sampler.sampling(range_max=frame_count)

        buffer = []
        for i, frame_name in enumerate(frames):
            frame = cv2.imread(frame_name)
            buffer.append(frame)
        buffer = np.array(buffer)
        return buffer.astype(np.float32)

    def crop_video_time(self, buffer, clip_len):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[0] - clip_len)
        buffer = buffer[time_index:time_index + clip_len, :, :, :]

        return buffer

    def crop_video_size(self, buffer, crop_size):

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[0] - crop_size)
        width_index = np.random.randint(buffer.shape[1] - crop_size)

        buffer = buffer[height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_data = VideoDataset(dataset='hmdb51', split='train', clip_len=16, preprocess=False)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

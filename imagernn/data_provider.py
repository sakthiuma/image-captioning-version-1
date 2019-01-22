data_provider.py

Type
Text
Size
5 KB (5,241 bytes)
Storage used
10 KB (10,446 bytes)
Location
imagernn
Owner
me
Modified
Jan 16, 2019 by me
Opened
Jan 21, 2019 by me
Created
Jan 16, 2019 with Google Drive Web
Add a description
Viewers can download
import json
import os
import random
import scipy.io
import codecs
from collections import defaultdict

class BasicDataProvider:
  def __init__(self, dataset):
    print 'Initializing data provider for dataset %s...' % (dataset, )

    # !assumptions on folder structure
    self.dataset_root = os.path.join('data', dataset) # join data folder to dataset name to get path to dataset
    self.image_root = os.path.join('data', dataset, 'imgs')  # create path to images folder

    # load the dataset into memory
    dataset_path = os.path.join(self.dataset_root, 'dataset.json')
    print 'BasicDataProvider: reading %s' % (dataset_path, )
    self.dataset = json.load(open(dataset_path, 'r')) # contains the json dataset


    # load the image features into memory
    features_path = os.path.join(self.dataset_root, 'vgg_feats.mat')
    print 'BasicDataProvider: reading %s' % (features_path, )
    features_struct = scipy.io.loadmat(features_path) # load .mat files in python as dict
    self.features = features_struct['feats'] # self.features is a numpy array
    print ' Image Features - dims :  ', self.features.shape # (4096, 8000)

    # group images by their train/val/test split into a dictionary -> list structure
    self.split = defaultdict(list)
    for img in self.dataset['images']:
      self.split[img['split']].append(img) # split elements include the image-id

    print 'Split:', self.split['train'][0]



  # "PRIVATE" FUNCTIONS
  # in future we may want to create copies here so that we don't touch the
  # data provider class data, but for now lets do the simple thing and
  # just return raw internal img sent structs. This also has the advantage
  # that the driver could store various useful caching stuff in these structs
  # and they will be returned in the future with the cache present
  def _getImage(self, img):
    """ create an image structure for the driver """

    # lazily fill in some attributes
    if not 'local_file_path' in img: img['local_file_path'] = os.path.join(self.image_root, img['filename'])
    if not 'feat' in img: # also fill in the features
      feature_index = img['imgid'] # NOTE: imgid is an integer, and it indexes into features
      img['feat'] = self.features[:,feature_index]
    return img

  def _getSentence(self, sent):
    """ create a sentence structure for the driver """
    # NOOP for now
    return sent

  # PUBLIC FUNCTIONS
  def getData(self,spl):
      """ return list of objects each with imageid, features and sentence tokens """
      data = []
      for img in self.split[spl] :
          dataObj = {}
          dataObj['imgid'] = img['imgid']
          dataObj['filename'] = img['filename']
          feature_index = img['imgid']
          dataObj['feats'] = self.features[:,feature_index]
          words = []
          for obj in img['sentences']:
              for word in obj['tokens']:
                if word not in words:
                    words.append(word)
          dataObj['tokens'] = words
          dataObj['tokenWeight']=[]
          data.append(dataObj)
      return data


  def getSplitSize(self, split, ofwhat = 'sentences'):
    """ return size of a split, either number of sentences or number of images """
    if ofwhat == 'sentences':
      return sum(len(img['sentences']) for img in self.split[split])
    else: # assume images
      return len(self.split[split])

  def sampleImageSentencePair(self, split = 'train'):
    """ sample image sentence pair from a split """
    images = self.split[split]

    img = random.choice(images)
    sent = random.choice(img['sentences'])

    out = {}
    out['image'] = self._getImage(img)
    out['sentence'] = self._getSentence(sent)
    return out

  def iterImageSentencePair(self, split = 'train', max_images = -1):
    for i,img in enumerate(self.split[split]):
      if max_images >= 0 and i >= max_images: break
      for sent in img['sentences']:
        out = {}
        out['image'] = self._getImage(img)
        out['sentence'] = self._getSentence(sent)
        yield out

  def iterImageSentencePairBatch(self, split = 'train', max_images = -1, max_batch_size = 100):
    batch = []
    for i,img in enumerate(self.split[split]):
      if max_images >= 0 and i >= max_images: break
      for sent in img['sentences']:
        out = {}
        out['image'] = self._getImage(img)
        out['sentence'] = self._getSentence(sent)
        batch.append(out)
        if len(batch) >= max_batch_size:
          yield batch
          batch = []
    if batch:
      yield batch

  def iterSentences(self, split = 'train'):
    for img in self.split[split]:
      for sent in img['sentences']:
        yield self._getSentence(sent)

  def iterImages(self, split = 'train', shuffle = False, max_images = -1):
    imglist = self.split[split]
    ix = range(len(imglist))
    if shuffle:
      random.shuffle(ix)
    if max_images > 0:
      ix = ix[:min(len(ix),max_images)] # crop the list
    for i in ix:
      yield self._getImage(imglist[i])

def getDataProvider(dataset):
  """ we could intercept a special dataset and return different data providers """
  assert dataset in ['flickr8k', 'flickr30k', 'coco'], 'dataset %s unknown' % (dataset, )
  return BasicDataProvider(dataset)
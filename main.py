#Libraries
# %tensorflow_version 2.x
import os.path
from os import path
import gc
import h5py
from keras.utils.io_utils import HDF5Matrix
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Concatenate, Multiply
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization, GlobalAveragePooling2D, Add, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from keras import metrics
import glob
import scipy.io as sio 
import os, zipfile
from copy import copy
try:
  import soundfile as sf 
except:
  !pip install soundfile
  import soundfile as sf                                             
from scipy import signal   
from matplotlib import mlab
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

#Generator 
def generator(x, label_b, batch_size):
	size=len(x)
	idx=0
	while 1:
		last_batch=idx+batch_size>size
		end=idx+batch_size if not last_batch else size
		yield x[idx:end], label_b[idx:end]
		idx= end if not last_batch else 0

def download_dataset(url, root_path):
  if not os.path.exists(root_path):
    file_unzip=url[-6:]
    !wget $url
    !unzip -qq $file_unzip
    !rm -rf $file_unzip
  return

#Feature Engineering
def convert_label(label, type):
  #label binary
  if type=='LA':
    if label=='-':
      return 0
    else:
      return int(label[1:3])

  #label multiclass
  if type=='PA':

    if label == 'AA':
      return 1
    if label == 'AB':
      return 2
    if label == 'AC':
      return 3
    if label == 'BA':
      return 4
    if label == 'BB':
      return 5
    if label == 'BC':
      return 6
    if label == 'CA':
      return 7
    if label == 'CB':
      return 8
    if label == 'CC':
      return 9
    else:
      return 0 

def features_and_labels(path_input, path_label, access, type_data):
  #This function computes spectrogram for each audio, creates the extended featureMAP, divides this in segments of lenght 400 
  #and creates the list of the inputs of the neural network and the list of the rispective labels
  #The lists are saved of hdf5 files
  #type_data defines the partition (train, dev, eval)
  
  list_input=[]
  list_label_b=[]
  list_label_m=[]
  UFeatureMAP=np.zeros((257, 1600)) 
  segment=np.zeros((257,400))

  M=400 #Lenght of the segments
  L=200 #Overlap between segments
  
  file_label = open(path_label,"r", encoding='utf8')
  rows=file_label.readlines();
  file_label.close()

  #File txt where for each audio where are saved the number of segments and the label
  file_segments=type_data+'_segments.txt'
  f=open(file_segments,'w', encoding='utf8')
  f.close()
  
  #File where segments and labels will be saved 
  filename_input=os.path.join(type_data+'_input.hdf5') #es. train_input.hdf5
  filename_label_m=os.path.join(type_data+'_label_m.hdf5')
  filename_label_b=os.path.join(type_data+'_label_b.hdf5')

  fx = h5py.File(filename_input, "a")
  fy_multi = h5py.File(filename_label_m, "a")
  fy_binary = h5py.File(filename_label_b, "a")

  
  dset = fx.create_dataset('dataset', (0,257,400,1), maxshape=(None,None,None,None), chunks=True, 
                           compression="gzip", compression_opts=9)
  

  dset_binary=fy_binary.create_dataset('dataset', (0,), maxshape=(None,), chunks=True, 
                                       compression="gzip", compression_opts=4)   
  
  dset_multi=fy_multi.create_dataset('dataset', (0,), maxshape=(None,), chunks=True, 
                                     compression="gzip", compression_opts=4)
  
  print('Start extracting features...')
  print('Creating hdf5 file...')

  N_files=len(glob.glob(path_input))
  S=3000 #How much segments are appended to the file hdf5 each time
  N_iterazioni=int(N_files/S)
  rest2=N_files-N_iterazioni*S
  counter=0
  
  #glob.glob return a string like this:  /content/LA/ASVspoof2019_LA_train/flac/LA_T_1929428.flac
  #We wanto to extract from this string the name of the audio that is in this case: LA_T_1929428
  #So, to do that we take filename[len(filename)-17:-5]
  for filename in glob.glob(path_input):
    counter=counter+1
    flag_found=0
    for row in rows:
      fields=row.split(" ")
      if fields[1]==filename[len(filename)-17:-5]:
        flag_found=1
        label_multi=fields[3]
        label_multi=convert_label(label_multi, access)
        if fields[4]=='bonafide\n':
          label_binary=0
        if fields[4]=='spoof\n':
          label_binary=1
        break
    
    if flag_found==1:
      #Spectrogram
      data,fs=sf.read(filename)
      [Ps, f, t] = mlab.specgram(data, NFFT=512, Fs=fs, window=signal.get_window('hamming', 512), noverlap=256, mode='psd')
      Ps=np.where(Ps==0, 1e-200 , Ps)
      PsdB=10*np.log10(Ps);
      [Nf,Nt]=PsdB.shape

      #For each audio define the lenght of the unified feature map
      multiple=int(np.ceil((Nt/400)))
      N=2*multiple-1
      lenght=multiple*400

      #Create UNIFIED FEATURE MAP
      Q=int (lenght/Nt )
      rest1 = lenght  - Nt*Q
      UFeatureMAP[:, 0:(Q*Nt)] = np.tile(PsdB,Q)
      UFeatureMAP[:, (Q*Nt):lenght] = PsdB[:, 0:rest1]

      #Divide feature map in segments
      #Append the segment to list of all segmets and the corrispective to list of labels
      for i in range(0,N):
        segment=np.copy(UFeatureMAP[: , (i*L):(i*L+M)])
        segment=np.expand_dims(segment, axis=2)
        list_input.append(segment)
        list_label_m.append(label_multi)
        list_label_b.append(label_binary)

      #Write on a file, for each audio, the number of the segments and the label (binary and multi-class) of the speech
      f=open(file_segments,'a',encoding='utf8')
      a = '{0}\t{1}\t{2}\n'.format(N,str(label_binary),str(label_multi))
      f.write(a)
      f.close()

      P=len(list_input)

    #Each time S audio are processed, the list of the segments is appendend on the file hdf5 and the list is reset
    if counter%S==0 and counter!=N_files and P!=0:
      print(counter)
      dset.resize(dset.shape[0]+P, axis=0) 
      dset[-P:,:,:,:] = list_input[0:P] 

      dset_binary.resize(dset_binary.shape[0]+P, axis=0)   
      dset_binary[-P:] = list_label_b[0:P]

      dset_multi.resize(dset_multi.shape[0]+P, axis=0)   
      dset_multi[-P:] = list_label_m[0:P]
        
      list_input=[]
      list_label_m=[]
      list_label_b=[]
        
    #This is the last iteration 
    if counter==N_files and P!=0:
      print(counter)
      dset.resize(dset.shape[0]+P, axis=0) 
      dset[-P:,:,:,:] = list_input[0:P] 

      dset_binary.resize(dset_binary.shape[0]+P, axis=0)   
      dset_binary[-P:] = list_label_b[0:P]

      dset_multi.resize(dset_multi.shape[0]+P, axis=0)   
      dset_multi[-P:] = list_label_m[0:P]
        
      list_input=[]
      list_label_m=[]
      list_label_b=[]

  fx.close()
  fy_multi.close()
  fy_binary.close()

  print('Done...')
  return

#Model
def residual(input, filters, type_resnet, stride):
  if type_resnet==34:
    conv_1=Conv2D(filters,(3,3), activation=None, strides=stride, padding='same')(input)
    batch_1=BatchNormalization()(conv_1)
    act_1=Activation('relu')(batch_1)
    
    conv_2=Conv2D(filters,(3,3),activation=None, strides=(1,1),padding='same')(act_1)
    batch_2=BatchNormalization()(conv_2)
    return batch_2
  
  if type_resnet==50:
    conv_1=Conv2D(filters,(1,1), activation=None, strides=stride, padding='valid')(input)
    batch_1=BatchNormalization()(conv_1)
    act_1=Activation('relu')(batch_1)

    conv_2=Conv2D(filters,(3,3), strides=(1,1) ,padding='same')(act_1)
    batch_2=BatchNormalization()(conv_2)
    act_2=Activation('relu')(batch_2)

    conv_3=Conv2D(2*filters,(1,1), activation=None, strides=(1,1), padding='valid')(act_2)
    batch_3=BatchNormalization()(conv_3)
    return batch_3

def SE(input, filters,N):
    r=16
    global_pooling = GlobalAveragePooling2D()(input)
    fully_connected_1 = Dense(N*filters//r, activation='relu')(global_pooling)
    fully_connected_2 = Dense(N*filters, activation='sigmoid')(fully_connected_1)
    return fully_connected_2

def se_res_block(X , filters , units , type_resnet, halving=1):
  #halving=1 means that the first convolution layer in the first block of each macro-block halves the size with a stride 2x2 
  if type_resnet==34: 
      N=1
  if type_resnet==50: 
      N=2

  if halving==0:
    stride=(1,1)
    input_reshaped=Conv2D(N*filters,(1,1), activation=None, strides=(1,1), padding='valid')(X)
  else:
    stride=(2,2)
    input_reshaped=Conv2D(N*filters,(1,1),activation=None, strides=(2,2), padding='valid')(X)
  
  
  res=residual(X, filters, type_resnet, stride=stride)
  output_SE=SE(res, filters, N) 

  scale= Multiply()([res, output_SE]) 
  X=Add()([input_reshaped, scale])
  X=Activation('relu')(X) 
  for i in range(1, units):
    input_reshaped=X
    res=residual(X,filters, type_resnet, stride=(1,1)) 
    output_SE=SE(res, filters, N)
    scale= Multiply()([res, output_SE])
    X=Add()([input_reshaped, scale])
    X=Activation('relu')(X)
  return X

def define_model(type_resnet):
  input_data = Input(shape=(257, 400 ,1))
  X=Conv2D(16, (7,7) , (2,2) ,activation=None,padding='same')(input_data)
  X=BatchNormalization()(X)
  x=Activation('relu')(X)
  X=MaxPooling2D(pool_size=(3,3), padding='same',strides=(2,2))(X)

  X=se_res_block(X, 16, 3, type_resnet, 0)
  X=se_res_block(X, 32, 4, type_resnet, 1)
  X=se_res_block(X, 64, 6, type_resnet, 1) 
  X=se_res_block(X, 128, 3, type_resnet, 1)

  X=GlobalAveragePooling2D()(X)
  X=Dense(1, activation=None)(X)
  Y_b=Activation('sigmoid')(X)
  return Model(input_data, Y_b)

#Callbacks
class ModelSelection(tf.keras.callbacks.Callback):
    def __init__(self, access, model, test_generator, steps, type_resnet):
        self.maxAcc=0
        self.access=access
        self.model=model
        self.epoch_count=[]
        self.dev_acc=[]
        self.train_acc=[]
        self.test_generator=test_generator
        self.steps=steps
        self.patience=15
        self.epoch_star=0
        self.type_resnet=type_resnet

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_count.append(epoch+1)

    def on_epoch_end(self, epoch, logs={}):
        score=self.model.evaluate(x=self.test_generator, steps=self.steps, verbose=1, max_queue_size=50)
        
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        acc=score[1]
        acc1=logs.get('accuracy')
        self.train_acc.append(acc1)
        self.dev_acc.append(acc)

        if acc>self.maxAcc:
          self.maxAcc=acc
          self.epoch_star=epoch
          model_name=self.access+'_SENET'+str(self.type_resnet)+'.h5'
          self.model.save(model_name)

        if epoch-self.epoch_star>=self.patience:
          self.model.stop_training = True

        plt.figure(0)
        plt.plot(self.epoch_count, self.train_acc, 'g')
        plt.plot(self.epoch_count, self.dev_acc, 'b')
        plt.title('Model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='best')
        photoname='Accuracy_'+self.access+'.png'
        plt.savefig(photoname)
        
class WarmUpLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, verbose=0):
        super(WarmUpLearningRate, self).__init__()
        self.step=1
        self.d_model=2*128
        self.verbose = verbose
        self.warmup_steps=1000
        self.incremento=1

    def on_epoch_end(self, epoch, logs=None):
        print('\n')
        print('Learning rate: ', K.get_value(self.model.optimizer.lr))

    def on_batch_end(self, batch, logs=None):
        self.step = self.step + self.incremento
        lr = K.get_value(self.model.optimizer.lr)

    def on_batch_begin(self, batch, logs=None):  
        lr = np.power(self.d_model, (-0.5))*np.minimum( (np.power(self.step, (-0.5))), (self.step*np.power(self.warmup_steps, (-1.5))) )
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
           print('\nBatch %05d: WarmUpLearningRateScheduler setting learning-rate to %s.' % (self.step, lr))

#Scoring
def compute_EER(p1, p2, thresholds, system):
  #p1->list of false negative
  #p2->list of false positive 
  if system=='CM':
    plt.figure(1)
    plt.plot(thresholds, p1, 'g')
    plt.plot(thresholds, p2, 'b')
    title='EER_'+system
    plt.title(title)
    plt.ylabel('value')
    plt.xlabel('s')
    plt.legend(['False negative', 'False positive'], loc='best')
    photoname='EER_'+system+'.png'
    plt.savefig(photoname)

  index_min=np.nanargmin(np.abs(p1-p2))
  operating_point=thresholds[index_min]
  return index_min, operating_point

def compute_ASV_error_rates(file_score):
  f=open(file_score, 'r')
  s=[]
  score_tar=[]
  score_non=[]
  score_spoof=[]

  rows=f.readlines()
  for row in rows:
    fields=row.split(' ')
    if fields[1]=='target':
      score_tar.append(float(fields[2]))    
      s.append(float(fields[2]))

    if fields[1]=='nontarget':
      score_non.append(float(fields[2]))
      s.append(float(fields[2]))

    if fields[1]=='spoof':
      score_spoof.append(float(fields[2]))

  s.sort()
  L_tar=len(score_tar)
  L_spoof=len(score_spoof)
  L_non=len(score_non)
  L_s=len(s)
  P_asv_miss=np.zeros(L_s)
  P_asv_fa=np.zeros(L_s)
  P_asv_miss_spoof=np.zeros(L_s)
  score_spoof=np.array(score_spoof)
  score_non=np.array(score_non)
  score_tar=np.array(score_tar)

  for i in range(0, L_s):
    P_asv_miss[i]=np.size(np.where(score_tar<=s[i]))/L_tar
    P_asv_fa[i]=np.size(np.where(score_non>s[i]))/L_non
    P_asv_miss_spoof[i]=np.size(np.where(score_spoof<=s[i]))/L_spoof
  
  return P_asv_miss, P_asv_fa, P_asv_miss_spoof, s

def compute_detection_score(input, file_segments, model, generator, steps, type):
  #Predict and average of all segments for each utterance
  #Save on a file the score of each utterance and the respective label
  #This file name 'LA_dev_score.txt'
  #Define two list: label contais the true value, detection_score contains the score
  f=open(file_segments, 'r')
  f1=open('risultati'+type+'.txt', 'w', encoding='utf8')
  label=[]
  y=[]
  rows=f.readlines()
  start=0
  score_bonafide=[]
  score_spoof=[]
  
  predicted=model.predict(x=generator, steps=steps, verbose=1, max_queue_size=10)  
  print(predicted.shape)
  for row in rows:
    fields=row.split('\t')
    N=int(fields[0])
    y_pred=np.sum(predicted[start: start+N])/N
    start=start+(N)
    
  #Compute log-probability of bonafide class and save results on a file txt
    prob=1-y_pred
    log_prob=np.log10(prob)
    label.append(int(fields[1]))
    f1.write(str(y_pred)+'\t'+str(fields[1])+'\n')
    y.append(y_pred)
    if int(fields[1])==0:
      score_bonafide.append(np.copy(log_prob))
    else: 
      score_spoof.append(np.copy(log_prob))
    
  label=np.array(label)
  y=np.array(y)
  print(len(score_bonafide))
  print(len(score_spoof))
  score_bonafide=np.array(score_bonafide)
  score_spoof=np.array(score_spoof)

  print('Accuracy: ', np.array(metrics.binary_accuracy(label, y)))
  f.close()
  f1.close()
  return score_bonafide, score_spoof

def compute_CM_error_rates(input, file_segments, model, generator, steps, type):
  
  score_bonafide, score_spoof=compute_detection_score(input, file_segments, model, generator, steps, type)
  s=np.concatenate((score_bonafide, score_spoof))
  L=s.shape[0]
  L_bonafide=score_bonafide.shape[0]
  L_spoof=score_spoof.shape[0]
  P_cm_miss=np.zeros(L)
  P_cm_fa=np.zeros(L)
  s.sort()
  for p in range(0,L):
    counter_miss=np.size(np.where(score_bonafide<=s[p]))
    counter_fa=np.size(np.where(score_spoof>s[p]))
    P_cm_miss[p]=counter_miss/L_bonafide
    P_cm_fa[p]=counter_fa/L_spoof

  return P_cm_miss, P_cm_fa, s

def compute_performance(x, file_segments, file_score_ASV, model, type, access, generator, steps):
  #Parameters:
  prior_tar=0.9405
  prior_spoof=0.05
  prior_non=0.0095
  C_asv_miss=1
  C_asv_fa=10
  C_cm_miss=1
  C_cm_fa=10

  #Parameters ASV
  P_asv_miss_list, P_asv_fa_list, P_asv_miss_spoof_list, s_asv=compute_ASV_error_rates(file_score_ASV)
  
  index_eer_asv, operating_point_asv=compute_EER(P_asv_miss_list, P_asv_fa_list, s_asv, 'ASV')

  P_asv_miss_spoof=P_asv_miss_spoof_list[index_eer_asv]
  P_asv_fa=P_asv_fa_list[index_eer_asv]
  P_asv_miss=P_asv_miss_list[index_eer_asv]


  #Parameters CM
  P_cm_miss_list, P_cm_fa_list, s_cm=compute_CM_error_rates(x, file_segments, model, generator, steps, type)

  index_eer_cm, operating_point_cm=compute_EER(P_cm_miss_list, P_cm_fa_list, s_cm, 'CM')
  eer_cm=(P_cm_miss_list[index_eer_cm]+P_cm_fa_list[index_eer_cm])/2*100

  C1=prior_tar*(C_cm_miss - C_asv_miss*P_asv_miss)-prior_non*C_asv_fa*P_asv_fa
  C2=prior_spoof * C_cm_fa* (1 - P_asv_miss_spoof)
  
  #Formula
  t_DCF = C1 * P_cm_miss_list + C2 * P_cm_fa_list
  t_DCF_norm = t_DCF/min(C1,C2)

  #Compute minimum t-DCF
  min_t_DCF=np.amin(t_DCF_norm)
  
  print('Summary performance of the '+access+' model computed on '+type+'-dataset:\nMinimum normalized tandem ' 
          'detection cost function (t-DCF): '+ str(min_t_DCF)+'\nERR %: '+str(eer_cm))
  return

#Function to download file from google drive
def download(ID, filename, destination):
  string="""--load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id="""+ID+"""' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id=""" +ID+ """" -O """+filename+""" && rm -rf /tmp/cookies.txt"""
  
  #!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1yUDQ7dur7Ow5QIZGws9HBZ9A7FiFZ2IC' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1yUDQ7dur7Ow5QIZGws9HBZ9A7FiFZ2IC" -O FILENAME.hdf5 && rm -rf /tmp/cookies.txt 
  if not os.path.exists(filename[:-4]):
    !wget $string
    !unzip -qq $filename -d $destination
    !rm -rf $filename
  return

#Main
def main(type_resnet, access):

  batch_size=64
  
  if access=='LA':
    download_dataset('https://datashare.is.ed.ac.uk/bitstream/handle/10283/3336/LA.zip', 'LA')
  if access=='PA':
    download_dataset('https://datashare.is.ed.ac.uk/bitstream/handle/10283/3336/PA.zip', 'PA')

  #Folders LA
  train_speech='/content/LA/ASVspoof2019_LA_train/flac/*'
  train_label='/content/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
  dev_speech='/content/LA/ASVspoof2019_LA_dev/flac/*'
  dev_label='/content/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'
  asv_scores='/content/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt'
  
  """
  #Folders PA
  train_speech='/content/PA/ASVspoof2019_PA_train/flac/*'
  train_label='/content/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt'
  dev_speech='/content/PA/ASVspoof2019_PA_dev/flac/*'
  dev_label='/content/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt'
  asv_scores='/content/PA/ASVspoof2019_PA_asv_scores/ASVspoof2019.PA.asv.dev.gi.trl.scores.txt'
  """
  
  features_and_labels(path_input=train_speech, path_label=train_label, access=access, type_data='train')
  !rm -r /content/PA/ASVspoof2019_PA_train #Delete folders to make space

  features_and_labels(path_input=dev_speech, path_label=dev_label, access=access, type_data='dev')
  !rm -r /content/PA/ASVspoof2019_PA_dev 

  train_input = HDF5Matrix('train_input.hdf5', 'dataset')
  train_label_m = HDF5Matrix('train_label_m.hdf5', 'dataset')
  train_label_b = HDF5Matrix('train_label_b.hdf5', 'dataset')
  print('Size training-set: ',len(train_input))
  
  dev_input = HDF5Matrix('dev_input.hdf5', 'dataset')
  dev_label_m= HDF5Matrix('dev_label_m.hdf5', 'dataset')
  dev_label_b = HDF5Matrix('dev_label_b.hdf5', 'dataset')
  print('Size dev-set: ', len(dev_input))

  train_samples=len(train_input)
  train_generator=generator(train_input, train_label_b, batch_size)
  
  test_sample=len(dev_input)
  test_generator=generator(dev_input, dev_label_b, batch_size)
  
  #Train
  network=define_model(type_resnet)
  network.summary()
  
  network.compile(Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9, amsgrad=False), loss=['binary_crossentropy'], metrics=['accuracy'])

  learningRate = WarmUpLearningRate()
  
  model_selection = ModelSelection(access=access, model=network, test_generator=test_generator, 
                                   steps=test_sample//batch_size+1, type_resnet=type_resnet)

  network.fit(x=train_generator, epochs=100, steps_per_epoch=train_samples//batch_size+1, verbose=1, 
              max_queue_size=50, callbacks=[learningRate, model_selection], 
              shuffle='batch', initial_epoch=0)
  
  #Compute performance on DEV
  model=load_model('') #Insert the name (.h5) of the model to be evaluated
  compute_performance(dev_input, 'dev_segments.txt', asv_scores, model, 'dev', access, test_generator, test_sample//batch_size+1)

  
  #EVALUATION
  download('12EfUbFf5j5tyxkDB_T_f-5fPLqNvuzUh', 'PA_eval.zip', 'PA_eval' )
  download('1-26ZMP5I7FB1BmDdLzVL_thkgkziOnt_', 'PA_protocols.zip' ,'PA_protocols')
  
  eval_speech='/content/LA/ASVspoof2019_LA_eval/flac/*'
  eval_label='/content/LA_protocols/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'

  features_and_labels(path_input=eval_speech, path_label=eval_label, access=access, type_data='eval')
  
  eval_input = HDF5Matrix('eval_input.hdf5', 'dataset')
  eval_label_m = HDF5Matrix('eval_label_m.hdf5', 'dataset')
  eval_label_b = HDF5Matrix('eval_label_b.hdf5', 'dataset')

  eval_sample=len(eval_input)
  eval_generator=generator(eval_input, eval_label_b, eval_label_m,  batch_size)

  #Evaluate model on EVAL
  model=load_model(' ') #Insert the name (.h5) of the model to be evaluated
  compute_performance(eval_input, 'eval_segments.txt', asv_score, model, 'eval', access, eval_generator, eval_sample//batch_size+1)

  return

main(type_resnet=50, access='LA')
#main(type_resnet=50, access='PA')
#main(type_resnet=34, access='LA')
#main(type_resnet=34, access='PA')



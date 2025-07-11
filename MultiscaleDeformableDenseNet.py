import torchvision.ops
import torch
import torch.nn as nn

class DeformableConv2d(nn.Module):
    def __init__(self,
                  in_channels,
                  out_channels,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  dilation=1,
                  bias=True):
        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation

        self.offset_conv = nn.Conv2d(in_channels,
                                      2 * kernel_size[0] * kernel_size[1],
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      bias= bias)
        
        nn.init.constant_(self.offset_conv.weight, 0.)
        if bias==True:
            nn.init.constant_(self.offset_conv.bias, 0.)

        self.dconv = torchvision.ops.DeformConv2d(in_channels = in_channels,
                                                out_channels = out_channels,
                                                kernel_size = kernel_size,
                                                stride = self.stride,
                                                padding = self.padding,
                                                dilation = self.dilation,
                                                groups = 1,
                                                bias = bias)
        
        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        bias=bias)
        
        nn.init.constant_(self.modulator_conv.weight, 0.)
        if bias==True:
            nn.init.constant_(self.modulator_conv.bias, 0.)
            
    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        modul = self.modulator_conv(x)


        x = self.dconv(x, offset, modul)

        return x

   


#%% TinySceneEmbedding

class TinySceneEmbedding(nn.Module):
    def __init__(self, nb_input_channels):
        super(TinySceneEmbedding, self).__init__()

        self.nb_input_channels = nb_input_channels
        # Define adaptive average pooling and adaptive max pooling layers
        self.GlobalAvgPool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.GlobalMaxPool = torch.nn.AdaptiveMaxPool2d((1, 1))

        self.conv = nn.Conv2d(in_channels=2*nb_input_channels, out_channels=nb_input_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()


    def forward(self, inputs):
        # Apply global average pooling and global max pooling
        global_avg_pool = self.GlobalAvgPool(inputs)
        global_max_pool = self.GlobalMaxPool(inputs)

        # Concatenate the pooled features along the channel dimension
        concat = torch.cat((global_avg_pool, global_max_pool), dim=1)

        # Apply a 1x1 convolution to the concatenated features
        convolved = self.conv (concat)
        convolved = self.sigmoid (convolved)

        # Multiply the original inputs with the convolved features
        tiny_scene_embedded = inputs * convolved

        # Add the embedded features to the original inputs
        return inputs + tiny_scene_embedded

 
#%% Densenet


class DefConvolutionBlock(nn.Module):
    def __init__(self, nb_input_channels, growth_rate, botneck_scale,kernel_size, padding, dropout_rate=None):
        super(DefConvolutionBlock, self).__init__()

        self.growth_rate = growth_rate
        self.dropout_rate = dropout_rate
########### bottle neck
        reduced=int(growth_rate*botneck_scale)
        self.batch_norm_botneck = nn.BatchNorm2d(nb_input_channels)
        self.conv_botneck = nn.Conv2d(nb_input_channels, out_channels=reduced, kernel_size=1, padding=0, bias=False)
        
########## Dconv layer            
        self.batch_norm = nn.BatchNorm2d(reduced)
        self.relu = nn.ReLU() #nn.LeakyReLU()#
        self.conv = DeformableConv2d(in_channels=reduced, out_channels=growth_rate, kernel_size=kernel_size, stride=1, padding=padding)

        
        if self.dropout_rate:
            self.dropout = nn.Dropout2d(dropout_rate)
            
       

    def forward(self, x):
        # bottle neck
        x = self.batch_norm_botneck(x)
        x = self.relu(x)
        x = self.conv_botneck(x)

        # Standard (BN-ReLU-dConv)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)

        # Adding dropout
        if self.dropout_rate:
            x = self.dropout(x)
        
       
        return x






import torchvision

class DenseNetBlock(nn.Module):
    def __init__(self, nb_input_channels, nb_layers, growth_rate, botneck_scale,
                   kernel_size, padding, dropout_rate=None):
        super(DenseNetBlock, self).__init__()
        
        # nb_input_channels (int): Number of input channels            
        # nb_blocks (int): Number of DenseNet Blocks
        # nb_layers (int): Num of Deformable Convlutional layers in DenseNet block
        # growth_rate (int): growth rate in  DenseNet block            
        # botneck_scale (int)
        # kernel_size: kernel size for deformeable convlution          
        
        self.nb_layers = nb_layers
        self.growth_rate = growth_rate
        self.dropout_rate = dropout_rate

        self.squeeze_excite_blocks = nn.ModuleList()
        self.convolution_blocks = nn.ModuleList()

        for i in range(nb_layers):
            # Add squeeze-and-excite block
            SE = torchvision.ops.SqueezeExcitation(input_channels= nb_input_channels , squeeze_channels=int(0.25 * nb_input_channels))
            self.squeeze_excite_blocks.append(SE)

            # Add convolution block
            CB = DefConvolutionBlock( nb_input_channels = nb_input_channels, growth_rate= self.growth_rate, botneck_scale=botneck_scale, kernel_size= kernel_size, padding=padding, dropout_rate = self.dropout_rate)
            self.convolution_blocks.append(CB)

            nb_input_channels += growth_rate  # Update total number of channels


    def forward(self, x):
        #x=[batch, channel, patch_size**2]
        for i in range(self.nb_layers):
            # Apply squeeze-and-excite
            se = self.squeeze_excite_blocks[i](x) #se= [batch, channel, patch_size, pathch_size ]

            # Apply convolution block
            cb = self.convolution_blocks[i](se) #cb=[batch, growth_rate, patch_size**2]

            # Concatenate the outputs
            x = torch.cat((cb, x), dim=1)

        return x

#%% Transition layer       
    
class TransitionLayer(nn.Module):
    def __init__(self, nb_input_channels, dropout_rate=None, transition_factor=1.0):
        super(TransitionLayer, self).__init__()
        # nb_input_channels (int): Number of input channels
        # transition_factor (float in [0-1]): Transition factor in transition layers after each block

        self.batch_norm = nn.BatchNorm2d(nb_input_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(nb_input_channels, int(nb_input_channels * transition_factor), kernel_size=1, padding=0, bias=False)

        self.dropout_rate = dropout_rate
        if self.dropout_rate:
            self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)

        # Adding dropout
        if self.dropout_rate:
            x = self.dropout(x)

       
        return x        




#%% Build Model
class MDD(nn.Module):
    def __init__(self, nb_input_channels, nb_blocks, nb_layers_in_block, growth_rate, 
                 nb_classes,botneck_scale,                     
                 dropout_rate= None, transition_factor= 1.0 ):
        super(MDD, self).__init__()
        
        # nb_input_channels (int): Number of input channels to the network
        # nb_blocks (int): Number of DenseNet Blocks
        # nb_layers_in_block (List of int): Num of Deformable Convlutional layers in each block (if you have nb_blocks=2, nb_layers_in_block should be a list like [6,6] meaning that there is 6 layers in each block)
        # growth_rate (int): growth rate in each DenseNet block
        # nb_classes (int): Total number of classes in data
        # botneck_scale (int)
        # transition_factor (float in [0-1]): Transition factor in transition layers after each block

        self.nb_input_channels = nb_input_channels
        self.nb_classes=nb_classes
        self.tiny_scene_embedding = TinySceneEmbedding(nb_input_channels)

         
        self.dense_blocks = nn.ModuleList()

        for block in range(nb_blocks):

            dense_block = DenseNetBlock(nb_input_channels= nb_input_channels, nb_layers= nb_layers_in_block[block], 
                                     growth_rate = growth_rate, botneck_scale=botneck_scale, dropout_rate = dropout_rate,
                                      kernel_size= 3, padding=1)
            self.dense_blocks.append(dense_block)
            nb_input_channels += nb_layers_in_block[block] * growth_rate

            transition = TransitionLayer(nb_input_channels, dropout_rate= dropout_rate, transition_factor= transition_factor)
            self.dense_blocks.append(transition)
            nb_input_channels = int(nb_input_channels * transition_factor)

        self.dense_blocks_5 = nn.ModuleList()
        nb_input_channels = self.nb_input_channels
        for block in range(nb_blocks):

            dense_block = DenseNetBlock(nb_input_channels= nb_input_channels, nb_layers= nb_layers_in_block[block], 
                                      growth_rate = growth_rate, botneck_scale=botneck_scale, dropout_rate = dropout_rate,
                                      kernel_size= 5, padding=2)
            self.dense_blocks_5.append(dense_block)
            nb_input_channels += nb_layers_in_block[block] * growth_rate

            transition = TransitionLayer(nb_input_channels, dropout_rate= dropout_rate, transition_factor= transition_factor)
            self.dense_blocks_5.append(transition)
            nb_input_channels = int(nb_input_channels * transition_factor)    


        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        


        self.fc = nn.Linear(nb_input_channels*(1**2), nb_classes)
        self.fc2 = nn.Linear(nb_input_channels*(1**2), nb_classes)


    def forward(self, x):
        x = self.tiny_scene_embedding(x)
        x0 = x
        
    
        for layer in self.dense_blocks:
            x = layer(x)

        for layer in self.dense_blocks_5:
            x0 = layer(x0)
        
  
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        
        x0 = self.global_avg_pool(x0)
        x0 = x0.view(x0.size(0), -1)
        x0 = self.fc2(x0)
        

        return x+x0
    
    
    
    
    
import numpy as np

def ExtractPatches(img,GT=None, windowSize=3):
# This function divied the input image into patches making it ready for training your algorithm
# img: your image
# GT= your ground truth image. If you do not have any GT, just leave it
# windowSize: the size of each patch

  if GT is None:
        GT=np.ones_like(img[0,:,:]) # if the GT is not define, we create a dummy GT. In this case, you should ignore the output 'labels'

  margin = int((windowSize - 1) / 2) # margin to be added into your image
  img=np.pad(img, pad_width=((0,0),(margin,margin),(margin,margin)),mode='edge') # padding the input image according to the margin

  GT=np.pad(GT, pad_width=((margin,margin),(margin,margin)) ) # padding GT
  pos_in_image=np.asarray(np.where(GT!=0)).T # find the labled data in ground truth gt
  labels= GT[pos_in_image[:,0],pos_in_image[:,1]] #label of the samples

  ExtractedPatches=[]
  for i in pos_in_image:
    b=img[:,i[0]-margin:i[0]+margin+1, i[1]-margin:i[1]+margin+1] # extract patches
    ExtractedPatches.append(b)

  return np.asanyarray(ExtractedPatches), labels, pos_in_image


def ScaleData(X,min_value,max_value):
  #This function sclae the data 'X' between 0 and 1
  min_value=np.float16(min_value)
  max_value=np.float16(max_value)
  X -= min_value
  X /= (max_value - min_value)
  return X

def train_split_n_sample_perclass(X, labels, per_class,randomstate, nb_class):

    X_train=[]
    y_train=[]
    for i in range(nb_class):
        a=np.where(labels == i+1)[0]
        
        random_state = np.random.RandomState(seed= randomstate)
    
        # Randomly select n elements
        selected_elements = random_state.choice(a, size=per_class, replace=False)
    
        b = X[selected_elements,:,:,:]
        X_train.append(b)
        y_train.append(labels[selected_elements])
        
    X_train=np.concatenate(X_train)  
    y_train=np.concatenate(y_train)  
    
    return X_train, y_train    



from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, cohen_kappa_score, accuracy_score, confusion_matrix
from tqdm import tqdm
def evaluate_model_accuracy(model, data, gt, windowSize, enc, ScaleData, min_value, max_value,
                             batch_size, chunk_size, device, model_path='best_model_2.pth', random_state=0):

    margin = int((windowSize - 1) / 2)
    data = np.pad(data, pad_width=((0, 0), (margin, margin), (margin, margin)), mode='edge')
    gt = np.pad(gt, pad_width=((margin, margin), (margin, margin)))
    pos_in_image = np.asarray(np.where(gt != 0)).T

    def data_chunk_generator(pos_in_image, chunk_size):
        num_samples = pos_in_image.shape[0]
        for start in range(0, num_samples, chunk_size):
            end = start + chunk_size
            yield pos_in_image[start:end, :]

    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_prediction = []

    for chunk in tqdm(data_chunk_generator(pos_in_image, chunk_size), 
                      total=(pos_in_image.shape[0] // chunk_size + 1), 
                      desc="Processing Chunks"):

        labels = gt[chunk[:, 0], chunk[:, 1]]
        X_test = []

        for i in chunk:
            patch = data[:, i[0] - margin:i[0] + margin + 1, i[1] - margin:i[1] + margin + 1]
            X_test.append(patch)

        X_test = np.asarray(X_test)
        y = enc.transform(labels.reshape(-1, 1)).toarray()
        X_test = ScaleData(X_test, min_value, max_value)

        X = torch.tensor(X_test, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).to(device)
        test_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)

        chunk_prediction = []

        with torch.no_grad():
            for inputs in test_loader:
                outputs = model(inputs[0])
                _, predicted = outputs.max(1)
                chunk_prediction.append(predicted.cpu().numpy())

        all_prediction.append(np.concatenate(chunk_prediction))

    result_array = np.concatenate(all_prediction)
    label = gt[pos_in_image[:, 0], pos_in_image[:, 1]] - 1

    # Compute metrics
    print(classification_report(result_array, label))
    Kappa = cohen_kappa_score(label, result_array)
    OA = accuracy_score(label, result_array)
    matrix = confusion_matrix(label, result_array)
    Producer_accuracy = matrix.diagonal() / matrix.sum(axis=1)

    # Save results
    # np.save(f'GT_{random_state}.npy', label)

    print(f"Overall Accuracy: {OA*100:.2f}")
    print(f"Kappa coefficient: {Kappa:.4f}")
    print(f"Mean PA: {np.mean(Producer_accuracy * 100):.2f}")
    print(f"Producer accuracy: {Producer_accuracy * 100}")

    return {
        'OA': OA,
        'Kappa': Kappa,
        'Mean_PA': np.mean(Producer_accuracy),
        'Producer_accuracy': Producer_accuracy,
        'Predicted': result_array,
        'GroundTruth': label
    }

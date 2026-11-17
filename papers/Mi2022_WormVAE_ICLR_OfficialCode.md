# Mi et al. (ICLR 2022) WormVAE — 官方开源代码合集（PDF受OpenReview拦截，以完整源码核验）

## File: `01_WormVAE_README.md`
```python
# wormvae

We provide offical PyTorch implementations for connectome-constrained latent variable model of whole-brain neural activity. 

The details of algorithm are presented in the paper [Connectome-constrained Latent Variable Model of Whole-brain Neural Activity](https://openreview.net/pdf?id=CJzi3dRlJE-), ICLR 2022. 

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Installation
- Clone this repo:
```bash
git clone https://github.com/TuragaLab/wormvae.git
cd wormvae
```
- Install dependencies by using `conda env create -f environment.yml`.
- Activate conda enviroment by using `conda activate wormvae`.

## Datasets Visualization

### Worm Connectivity
- Data folder `data/worm_connectivity` provides the connectome from C.elegans from the paper [The Structure of the Nervous System of the Nematode Caenorhabditis elegans (The Mind of a Worm)](https://www.wormatlas.org/MoW_built0.92/MoW.html).
- C.elegan Connectome.
  <div align="center">
  <img src="readme_imgs/connectome.png" width="400" height="200">
  </div>

### Worm Activity
- Data folder `data/worm_activity` provides the measured fluorescence trace from C.elegans given chemosensory stimulus from the paper [NeuroPAL: A Multicolor Atlas for Whole-Brain Neuronal Identification in C. elegans](https://www.sciencedirect.com/science/article/pii/S0092867420316822).
- NeuroPAL Calcium Imaging of C.elegans.
  <div align="center">
  <img src="readme_imgs/NeuroPAL_calcium_imaging.png" width="400" height="200">
  </div>

## Scripts Usage

### Model Training
Call the following script in `experiment/` to train a latent variable model for whole-brain neural activity.
```bash
python main.py --neuron_holdout list_of_neuron_holdout --train_worm list_of_worm_id --model_type model_type --constraint constraint --random_init_index random_init_index 
```
- `--neuron_holdout` defines the set of neurons to be held out during training. You can specify a list of neuron names to choose which neurons (typically a pair of symmetric neurons) to be held out, e.g. `--neuron_holdout ['BAGL','BAGR']`.
- `--train_worm` defines the set of worms to be trained. You can specify a list of worm ids to choose which worms to be trained on, e.g. `--train_worm [1]`.
- `--model_type` defines the types of synapse modeling. You can use `--model_type conductance` to specify a conductance-based modeling, while `--model_type current` to specify a current-based modeling.
- `--constraint` defines the constraint to synapse weight. `--constraint weight` indicates using connectome synapse count to constrain the weight; `--constraint sparsity` indicates using connectome synapse sparsity (binarized matrix) to constrain the sparsity, while the magnitude is still trainable; `--constraint unconstrained` indicates without connectome constrained, the weight is fully-connected matrix and the magnitude is trainable.
- `--random_init_index` defines the index of random initialization in each training trial. In the paper, we use 4 different random initializations to evaluate the models, e.g. `--random_init_index 0`.

## Results
Checkpints are saved as `experiment/chechpoints/task_name.pt` files , the training logging are saved as `experiment/logs/task_name.log` file; and the training loss trajectory are saved in the `experiment/loss_trajectories/task_name.pickle`. `task_name` is automatically defined with the options of arguments (neuron holdout, train worm, model types, constraint, etc.) described above.

### Sample Results
- Neuron holdout under different type of synapse model types and levels of connectome constraints.
  <div align="center">
  <img src="readme_imgs/neuron_holdout.png">
  </div>
- Measured fluorescence traces (including unrecorded neurons) and inferred voltage for whole-brain (300 neurons).
  <div align="center">
  <img src="readme_imgs/measure_fluorescence_infer_voltage.png" width="500" height="160">
  </div> 
  

```

## File: `01_WormVAE_data_worm_activity_data_readme.txt`
```python
fps = frames/second (all traces have been re-sampled to the same constant frame rate)
Note: heads are at 4Hz (the max possible without upsampling)
is_AWCL_OFF = for each file, is AWCL the OFF neuron
Note: traces are either length=21 for single-sided neurons or, 42 for L/R neurons.
When length=42, neurons are ordered 21 L then 21 R, in file order.
is_L = for each neuron trace, is it L (1), R (0), or neither (nan) redundant to the above
max_traces = the max(dF/F0) per trace
neurons = the names of the neurons (L/R are combined into a single class)
neurons_i = indices for the L/R neurons in a separate file (available if you want)
samples = the number of real-valued traces available per neuron
Note: traces contain empty ({}) when the neuron was unavailable in the indexed file
stim_names = a list of the 3 stimuli we give
stim_times = times when stimuli are turned on & off
stims = the stimulus order per file
traces = the traces for each neuron (note the empty {} place holders)
Note: all traces are ordered by file but, for L/R neurons, its 21 L then 21 R
norm_traces = the normalized traces for each neuron (note the empty {} place holders)
Note: dF/F0 has negative values with using Erdem�s simple histogram matching without the nonnegative constrai.

```

## File: `01_WormVAE_main.py`
```python
import sys
from pathlib import Path
import torch
import logging
sys.path.append('../')
from model import *
from model import data_loader
import torch.nn.functional as F
import pickle
import argparse
torch.set_default_tensor_type('torch.cuda.FloatTensor')

parser = argparse.ArgumentParser()
parser.add_argument("--neuron_holdout", default = [], help="neuron holdout list")
parser.add_argument("--train_worm", default = [1], help="train worm list")
parser.add_argument("--model_type", default = 'conductance', help="synapse model type")
parser.add_argument("--constraint", default = 'weight', help="connectome constraint for synapse weight")
parser.add_argument("--random_init_index", default = 0, help="model random initialize index")
args = parser.parse_args()

exp_name =  f'worm_vae_neuron_held_out_{args.neuron_holdout}_train_worm_{args.train_worm}_{args.model_type}_{args.constraint}_rand_{str(args.random_init_index)}_'

logging.basicConfig(filename = 'logs/'+exp_name+'logging.log',level=logging.DEBUG,format='%(message)s')

print('args.neuron_holdout:',str(args.neuron_holdout))
print('train_worm:',str(args.train_worm))
print('model_type:',str(args.model_type))
print('args.constraint:',str(args.constraint))
print('random_init_index:',str(args.random_init_index))

activity_path = '../data/worm_activity/'
cnctm_path = '../data/worm_connectivity/'
savepath = 'checkpoints/'+exp_name+'_checkpoint_'

worm_data_loader = Worm_Data_Loader(activity_path)
connectome_data = WhiteConnectomeData(cnctm_path, 'cuda')
num_train_datasets = len(args.train_worm)
stim_channels = worm_data_loader.odor_channels
acquisition_dt = worm_data_loader.step
N = connectome_data.N

# Simulation hyparameters
upsample_factor = 40
dt = acquisition_dt/upsample_factor
window_size = 30

# Training hyperparameters
device = 'cuda'
lr_params = {}
lr_params['lr'] = 3e-4
lr_params['step_size'] = 50
lr_params['epochs'] = 300
lr_params['gamma'] = 0.5
lr_params['grad_clip'] = 1.0
lr_params['eps'] = 1e-8
lr_params['ckpt_save_freq'] = 100

# Inference network
# network architecture hyperparameters 
inference_net_params = {'k1': 11, 'k2': 21, 'k3': 21, 'k4': 41, 'k5': 41, 'up1_factor': 4, 'up2_factor': 10}
channels_dict = {'conv1': 2 * N,'conv2': 2 * N,'conv3': 2 * N}
kernel_size_dict = {'conv1': inference_net_params['k1'],
                    'conv2': inference_net_params['k2'],
                    'conv3': inference_net_params['k3'],
                    'conv4': inference_net_params['k4'],
                    'conv5': inference_net_params['k5']}
scale_factor_dict = {'upsample1': inference_net_params['up1_factor'], 'upsample2': inference_net_params['up2_factor']}
inference_network = Worm_Inference_Network(
    neuron_num = N,
    inputs_channels = 2 * N,
    channels_dict = channels_dict,
    kernel_size_dict = kernel_size_dict,
    scale_factor_dict = scale_factor_dict,
    nonlinearity = F.relu)

# Sensory encoder
encoder = Worm_Sensory_Encoder(stim_channels,N)

# Worm VAE : inference network + generative model
initialization = {'neuron_bias': 1e-3*torch.rand((1, N)).to(device),
                  'neuron_tau': 1e-2*torch.rand((1, N)).to(device),
                  'calcium_bias': 1e-2*torch.rand((1, N)).to(device),
                  'voltage_to_calcium_nonlinearity': F.softplus}

signs_c = torch.empty(N,N).uniform_(-0.01,0.01)
network = WormVAE(
    connectome = connectome_data,
    window_size = window_size,
    initialization = initialization,
    upsample_factor = upsample_factor,
    signs_c = signs_c,
    encoder = encoder,
    inference_network = inference_network,
    device = device,
    dt = dt,
    nonlinearity = F.leaky_relu,
    model_type = args.model_type,
    constraint =args.constraint)

network.to(device)

# Train data loader
upsampled_df_list = worm_data_loader.interpolate_odor(worm_data_loader.odor_worms[args.train_worm],upsample_factor)
stim_features = torch.from_numpy(upsampled_df_list).to(torch.float32).cuda()
target_raw = torch.from_numpy(worm_data_loader.activity_worms[args.train_worm]).to(torch.float32).cuda()
full_target = connectome_data.make_full_connectome_activity(
    activity_data = target_raw,
    activity_neuron_list = worm_data_loader.neuron_names,
    holdout_list = args.neuron_holdout)
full_target = full_target.to(torch.float32).cuda()
target, missing_target = data_loader.generate_target_mask(full_target)
train_dataset = data_loader.TimeSeriesDataloader(data_param_dict = inference_net_params,
                                     data = [stim_features, target, missing_target],
                                     window_size = [int(window_size * upsample_factor), window_size, window_size])
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 1, shuffle = True, drop_last = False)

# Model train
loss_list, recon_loss_list, KLD_list = train(data_loader = trainloader,
                                             network = network,
                                             model_type = args.model_type,
                                             constraint =args.constraint,
                                             lr_params = lr_params,
                                             savepath = savepath)

loss_dict = {'loss': loss_list, 'recon_loss': recon_loss_list, 'KLD': KLD_list}
with open('loss_trajectories/' + exp_name+'_loss.pickle', 'wb') as handle:
    pickle.dump(loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


```

## File: `01_WormVAE_model_cnctm_vae.py`
```python
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
import cnctm.nn as cnctmnn
from cnctm.utils.data import ConnectomeConstructor
from model.loss import ELBO_loss
import pdb

class Worm_Sensory_Encoder(nn.Module):
    """Sensory encoder for odor inputs
    Attrbutes:
        inputs: odor inputs
        outputs: sensory inputs for sensory neurons
    
    Methods:
        linear layers, activation: ReLU
    """

    def __init__(self, n_input, n_output, nonlinearity = F.softplus):
        super(Worm_Sensory_Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(n_input, n_output)
        self.nonlinearity = nonlinearity

    def forward(self, x):
        return self.nonlinearity(self.linear1(x.permute(0,2,1)))

class Worm_Inference_Network(nn.Module):
    """Inference network for wormvae
    Attributes:
        structure parameters of inference network;
        inputs: observed fluorescence trace + odor inputs
        outputs: approximated posterior distribution of voltage dynamics
    Methods:
        network: 1dconv , linear
        latent variables for all neurons:
        mu: mean of volatge posterior distribution
        logvar: log variance of volatge posterior distribution
        inference network: Q(v|f,o)
        v: posterir distribution for neuron volatge
        o: odor inputs
        f: fluorescence trace
    """

    def __init__(self, neuron_num, inputs_channels, channels_dict, kernel_size_dict, scale_factor_dict, nonlinearity = F.relu):
        # convolution without padding, expand inputs beforehand to be cropped
        super(Worm_Inference_Network, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels = inputs_channels, out_channels = channels_dict['conv1'], kernel_size = kernel_size_dict['conv1'])
        self.conv2 = torch.nn.Conv1d(in_channels = channels_dict['conv1'], out_channels = channels_dict['conv2'], kernel_size = kernel_size_dict['conv2'])
        self.conv3 = torch.nn.Conv1d(in_channels = neuron_num, out_channels = channels_dict['conv3'], kernel_size = kernel_size_dict['conv3'])
        self.conv4 = torch.nn.Conv1d(in_channels = channels_dict['conv2'] + channels_dict['conv3'], out_channels = neuron_num, kernel_size = kernel_size_dict['conv4'])
        self.conv5 = torch.nn.Conv1d(in_channels = channels_dict['conv2'] + channels_dict['conv3'], out_channels = neuron_num, kernel_size = kernel_size_dict['conv5'])
        self.nonlinearity = nonlinearity
        self.bn1 = torch.nn.BatchNorm1d(channels_dict['conv1'])
        self.bn2 = torch.nn.BatchNorm1d(channels_dict['conv3'])
        self.upsample1 = torch.nn.Upsample(scale_factor= scale_factor_dict['upsample1'],mode = 'linear', align_corners=True)
        self.upsample2 = torch.nn.Upsample(scale_factor= scale_factor_dict['upsample2'],mode = 'linear', align_corners=True)
        self.eps = 1e-8
        
    def forward(self, fluorescence_trace_target, missing_target_mask, sensory_input, is_Training = True):

        inputs = torch.cat((fluorescence_trace_target, missing_target_mask),1)
        y1 = self.nonlinearity(self.conv1(inputs)) 
        up1 = self.upsample1(y1)
        y2 = self.nonlinearity(self.conv2(up1))
        up2 = self.upsample2(y2)
        sensory = self.nonlinearity(self.conv3(sensory_input))
        merge = torch.cat((up2, sensory), 1)
        voltage_latent_mu = self.conv4(merge)

        if is_Training == True:
            voltage_latent_std =  0.1 * F.softplus(self.conv5(merge)) + self.eps
        else:
            voltage_latent_std =  0.0 * F.softplus(self.conv5(merge)) 
            
        voltage_latent_std_train = 0.1 * F.softplus(self.conv5(merge)) + self.eps
        voltage_latent_logvar = torch.log(voltage_latent_std**2)
        noise = torch.randn_like(voltage_latent_std)

        # reparametrization
        voltage_latent_sample = voltage_latent_mu + noise * voltage_latent_std

        infer_posterior = {'voltage_latent_mu': voltage_latent_mu,
                           'voltage_latent_logvar': voltage_latent_logvar,
                           'voltage_latent_sample': voltage_latent_sample,
                           'voltage_latent_std_train': voltage_latent_std_train,
                           }

        return infer_posterior

class WormVAE(nn.Module):

    """ Network model
    Attributes:
        inputs: 
            observed fluorescence trace
            observed odor inputs
        outputs: 
            approximated posterior distribution of voltage dynamics
            prior distribution of voltage dynamics
            reconstructed flurorescence trace
            chemical synapstic input
            electrical synapstic input
            encoded sensory input
            calcium concentration
    Methods:
        sensory encoder: sensory_in(t) = MLP(odor_in(t))
        inference_network Q(v|f,o): approximated posterior distribution of voltage dynamics
        generative model P(v|o): prior distribution for voltage dynamics from sensory inputs
            network dynamics: tau dv(t)/dt + v(t) = chem_in(t) + elec_in(t) + v_rest + sensory_in(t) + std_v * noise_v(t)
        generative model P(f|v,o): map from voltage dynamics into caclum fluorescence trace
            voltage to calcium activation: tau'd[Ca](t)/dt + [Ca](t) = softplus(v(t))
            calcium to fluorescence: f(t) = wf * [Ca](t) + bf + std_f * noise_f(t)
    """

    def __init__(self, connectome, window_size, initialization, upsample_factor, signs_c, encoder, inference_network, device, dt, nonlinearity, model_type, constraint):

        super(WormVAE, self).__init__()
        self.dt = dt
        self.cnctm_data = connectome
        self.upsample_factor = upsample_factor

        cnctm_dict = {'sparsity_c': connectome.synapses_dict['chem_adj'],
                      'sparsity_e': connectome.synapses_dict['esym_adj'],
                      'signs_c': signs_c,
                      'magnitudes_c': connectome.synapses_dict['chem_weights'],
                      'magnitudes_e': connectome.synapses_dict['eassym_weights'],
                      'n': connectome.N}

        self.connectome = ConnectomeConstructor(cnctm_dict)
        self.sensory_mask = connectome.neuron_mask_dict['sensory']
        self.encoder = encoder
        self.inference_network = inference_network
        self.T_init_steps = 10
        self.eps = 1e-8

        if model_type == 'current':
            self.network_dynamics = cnctmnn.leaky_integrator_current(connectome = self.connectome,
                                                                     dt = dt,
                                                                     hidden_init_trainable = False,
                                                                     bias_init = initialization['neuron_bias'],
                                                                     tau_init = initialization['neuron_tau'],
                                                                     nonlinearity = nonlinearity,
                                                                     constraint = constraint)

        if model_type == 'conductance':
            self.network_dynamics = cnctmnn.leaky_integrator_conductance(connectome = self.connectome,
                                                                         dt = dt,
                                                                         hidden_init_trainable = False,
                                                                         bias_init = initialization['neuron_bias'],
                                                                         tau_init = initialization['neuron_tau'],
                                                                         nonlinearity = nonlinearity,
                                                                         constraint = constraint)
                                                                                  
        

        self.voltage_to_calcium_activation_filter = cnctmnn.NeuronWiseAffine(in_features = connectome.N)
        self.calcium_to_fluorescence_filter = cnctmnn.NeuronWiseLinear(in_features = connectome.N)
        self.loss = ELBO_loss(connectome.N, window_size, self.upsample_factor)
        self.calcium_tau = Parameter(torch.rand(1))
        
    def forward(self, fluorescence_full_target, fluorescence_target, missing_target_mask, odor_input, is_Training = True):
        # sensory inputs encoded from odor inputs
        sensory_input_extend = (self.encoder.forward(odor_input) * self.sensory_mask).permute(0,2,1)
        
        # voltage posterior estimated from inference network
        infer_posterior = self.inference_network.forward(fluorescence_full_target, missing_target_mask, sensory_input_extend, is_Training = is_Training)
        crop_sensory = int((sensory_input_extend.shape[2] - infer_posterior['voltage_latent_mu'].shape[2])/2)
        sensory_input = sensory_input_extend[:,:,crop_sensory: sensory_input_extend.shape[2] - crop_sensory]
        hidden_init = infer_posterior['voltage_latent_sample']
        
        # voltage prior from network dynamics 
        prior_voltage_mu, chem_input, elec_input = self.network_dynamics.forward(sensory_input, hidden_init, is_Training = is_Training)
        
        # neurotransmitter concentration for calcium activation
        calcium_activation = self.voltage_to_calcium_activation_filter.forward(infer_posterior['voltage_latent_sample'])
        
        # calcium initial state estimation
        init_calcium_raw = (torch.mean(fluorescence_target[:,:,0:self.T_init_steps],dim=2) - self.calcium_to_fluorescence_filter.shift[None,:])/(self.calcium_to_fluorescence_filter.scale[None,:]+self.eps)
        init_calcium_raw = init_calcium_raw.detach()
        calcium_raw = (fluorescence_target - self.calcium_to_fluorescence_filter.shift[None,:,None])/(self.calcium_to_fluorescence_filter.scale[None,:,None]+self.eps)
        calcium_raw = calcium_raw.detach()
        
        # calcium concentration from explonential filter
        calcium_tau = self.calcium_tau * torch.ones_like(calcium_activation[:,:,0])
        pred_calcium_mu = self.SCF_exponential(calcium_activation/(calcium_tau[:,:,None]+self.eps), init_calcium_raw, calcium_tau)
       
        # fluroscence magnitude using affine transform
        pred_fluorescence_mu = self.calcium_to_fluorescence_filter.forward(pred_calcium_mu)

        outputs = infer_posterior
        outputs['pred_calcium_mu']=pred_calcium_mu
        outputs['pred_fluorescence_mu']=pred_fluorescence_mu
        outputs['prior_voltage_mu']=prior_voltage_mu
        outputs['chem_input']=chem_input
        outputs['elec_input']=elec_input
        outputs['sensory_input']=sensory_input
        outputs['calcium_activation']=calcium_activation

        return outputs
    
    def SCF_exponential(self, x, init_calcium, calcium_tau):
        """Explonential filter to generate calcium concentration from voltage.
        Convolves x with exp(-alpha) = exp(-t/tau) for multiple neurons.

        Args:
        x (torch.tensor): (batch_size, n_cells, R * window_size)
        alpha (torch.tensor): dt/tau (n_cells)
        kernel_size (int): size of kernel in time points
        Returns:
        conv (torch.tensor): (batch_size, n_cells,  R * window_size)
        """
        alpha = self.dt/(calcium_tau.squeeze(0))
        timesteps = x.size()[2]
        kernel_size = timesteps
        t = torch.arange(0, kernel_size, step=1, dtype=torch.float32)
        kernel = torch.exp(-t*alpha[:, None]).flip(1)[:, None, :]
        conv = torch.nn.functional.conv1d(input=x,
                                          weight=kernel,
                                          groups=alpha.shape[0],
                                          padding=kernel_size-1)[:, :, :-kernel_size+1]
        step_t = torch.arange(0, timesteps, step=1, dtype=torch.float32)
        init_exp = torch.exp(-step_t*alpha[:, None])
        calcium_init_exp = (init_calcium[:, :, None] * init_exp[None,:,:])
        return conv * self.dt + calcium_init_exp
```

## File: `01_WormVAE_cnctm_nn_leaky_integrator_conductance.py`
```python
import math
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules import Module
import numpy as np

class leaky_integrator_conductance(Module):
    """Leaky integrator for voltage dynamics using conductance-based synapse model
    Attrbutes:
        inputs:
            connectome
            sensory input
            constraint setting (synapse count, synapse sparsity, unconstrained)
        outputs:
            voltage prior distribution
            chemical synaptic input
            electrial synaptic input
    Methods:
        leaky integrator for voltage dynamics
        network dynamics: tau dv(t)/dt + v(t) = chem_in(t) + elec_in(t) + v_rest + sensory_in(t) + std_v * noise_v(t)
        conductance-based chemical synapse model:
            chem_in_i(t) = \sum (Eji-vi(t)) * Wji_c * softplus(vj(t))
            Eji reversal potential (-\infinity, \infinity)
            Wji_c conductance chemical weight [0, \infinity)
        electrical synapse model:
            elec_in_i(t) = \sum Wji_e * (vj(t)-vi(t))
            Wji_e electrical weight (gap junctions) [0, \infinity)
    """
    def __init__(self, connectome, dt, hidden_init=None, hidden_init_trainable=False,  bias_init=None, tau_init=None, nonlinearity=F.relu, constraint = 'sparsity', is_Training = True):
        super(leaky_integrator_conductance, self).__init__()
        self.dt = dt
        self.n = connectome.n
        self.nonlinearity = nonlinearity
        self.is_Training = is_Training
        self.signs_c = Parameter(torch.Tensor(connectome.signs_c))

        magnitude_scaling_factor_chem = 1e-2
        magnitude_scaling_factor_elec = 1e-2
        
        if constraint == 'weight':
            #connectome synapse count constrained
            self.magnitude_scaling_factor_chem = Parameter(torch.tensor(magnitude_scaling_factor_chem))
            self.magnitude_scaling_factor_elec = Parameter(torch.tensor(magnitude_scaling_factor_elec))
            self.sparsity_c = Parameter(torch.Tensor(connectome.sparsity_c), requires_grad=False)
            self.sparsity_e = Parameter(torch.Tensor(connectome.sparsity_e), requires_grad=False)
            self.reversal_potentials = Parameter(torch.empty(1,self.n,self.n).uniform_(-0.05,0.05))
            self.magnitudes_c = Parameter(torch.Tensor(connectome.magnitudes_c), requires_grad=False)
            self.magnitudes_e = Parameter(torch.Tensor(connectome.magnitudes_e), requires_grad=False)

        if constraint =='sparsity':
            #connectome sparsity constrained, connectome synapse count initialize
            self.magnitude_scaling_factor_chem = Parameter(torch.tensor(magnitude_scaling_factor_chem), requires_grad=False)
            self.magnitude_scaling_factor_elec = Parameter(torch.tensor(magnitude_scaling_factor_elec), requires_grad=False)
            self.sparsity_c = Parameter(torch.Tensor(connectome.sparsity_c), requires_grad=False)
            self.sparsity_e = Parameter(torch.Tensor(connectome.sparsity_e), requires_grad=False)
            self.reversal_potentials = Parameter(torch.empty(1,self.n,self.n).uniform_(-0.05,0.05))
            self.magnitudes_c = Parameter(torch.Tensor(connectome.magnitudes_c))
            self.magnitudes_e = Parameter(torch.Tensor(connectome.magnitudes_e))

        if constraint == 'unconstrained':
            #unconstrained
            self.magnitude_scaling_factor_chem = Parameter(torch.tensor(magnitude_scaling_factor_chem), requires_grad=False)
            self.magnitude_scaling_factor_elec = Parameter(torch.tensor(magnitude_scaling_factor_elec), requires_grad=False)
            self.reversal_potentials = Parameter(torch.empty(1,self.n,self.n).uniform_(-0.05,0.05))
            self.sparsity_c = Parameter(torch.ones_like(connectome.sparsity_c), requires_grad=False)
            self.sparsity_e = Parameter(torch.ones_like(connectome.sparsity_e), requires_grad=False)
            self.magnitudes_c = Parameter(torch.FloatTensor(self.n,self.n).uniform_(0, 2 * torch.mean(connectome.magnitudes_c)))
            self.magnitudes_e = Parameter(torch.FloatTensor(self.n,self.n).uniform_(0, 2 * torch.mean(connectome.magnitudes_e)))
        
        if bias_init is None:
            self.bias = Parameter(torch.empty(self.n).uniform_(0.01, 0.02))
        else:
            self.bias = Parameter(bias_init)
        if tau_init is None:
            self.tau = Parameter(torch.empty(self.n).uniform_(self.dt, 0.2))
        else:
            self.tau = Parameter(tau_init)

        if hidden_init is None:
            hidden_init = torch.zeros(self.n)
        self.hidden_init = Parameter(torch.Tensor(hidden_init), requires_grad=hidden_init_trainable)

    def forward(self, input, hidden_states, is_Training):

        W_c = torch.mul(self.sparsity_c, self.magnitudes_c * self.magnitude_scaling_factor_chem)
        W_e = torch.mul(self.sparsity_e, (self.magnitudes_e + self.magnitudes_e.transpose(0,1)) * self.magnitude_scaling_factor_elec)
        timesteps = input.shape[2]

        x = torch.zeros(hidden_states.shape)
        x[:,:,1:] = hidden_states[:,:,:-1]
        x_trans = x.permute(0,2,1)[:, :, None, :]
        if is_Training:
            batch_W_c = W_c.unsqueeze(2).repeat(x.shape[0],1,1, x.shape[2])
            batch_reversals = self.reversal_potentials.unsqueeze(3).repeat(1,1,1,x.shape[2])
            gap_potentials = x_trans - x_trans.permute(0,1,3,2)
            chem_in = torch.sum(batch_W_c * (batch_reversals - x.unsqueeze(2).repeat(1,1,self.n,1)), dim = 2).squeeze(0) * self.nonlinearity(x)
            elec_in = torch.sum(torch.mul(W_e[None,None,:,:], gap_potentials), dim = 3).permute(0,2,1)
        else:
            elec_in = torch.zeros(input.shape)
            chem_in = torch.zeros(input.shape)
            batch_W_c = W_c.repeat(x.shape[0],1,1)
            for it in range(timesteps):
                x_t = x[:,:,it][:,None,:]
                chem_in[:,:,it] =  torch.sum(batch_W_c * (self.reversal_potentials - x_t.repeat(1,self.n,1) ), dim = 2).squeeze(0) * self.nonlinearity(x[:,:,it])
                gap_potentials = x_t - x_t.permute(0,2,1)
                elec_in[:,:,it] = torch.sum(torch.mul(W_e[None,:,:], gap_potentials), dim = 2)
        dt_tau = (self.dt / self.tau).squeeze(0)
        bias = self.bias.squeeze(0)
        recurrent_in = chem_in + elec_in
        mu_neuron_voltage_prob = torch.mul(dt_tau[None,:, None], recurrent_in + bias[None, :, None] + input - x) + x
        return mu_neuron_voltage_prob, chem_in, elec_in


```

## File: `01_WormVAE_data_loader.py`
```python
import numpy as np
import pandas as pd
import sys
import torch
from pathlib import Path
from torch.utils.data import Dataset
import glob
from os.path import dirname, join as pjoin
import scipy.io as sio

class Worm_Data_Loader(Dataset):
    def __init__(self, basepath) -> None:

        '''     
        A data preprocessing class for recorded neuron activity respect to olfactory odor inputs in each worm.
    
        Parameters:
            basepath: a path to the base directory containing relevant directory

        Attributes:
            activity_worms: list of dataframes containing calcium imaging data for each worm
            odor_worms: list of dataframes containing odor inputs for each worm
            neuron_names: list of neuron names
        '''
        # activity data attributes
        self.odor_channels = 3
        self.step = 0.25
        self.N_dataset = 21

        # mat file attributes
        N_cell = 189
        T = 960
        N_length = 109
        T_start = 160
        activity_datasets = np.zeros((self.N_dataset, N_cell, T))
        odor_datasets = np.zeros((self.N_dataset, self.odor_channels, T))
        
        mat_fname = pjoin(basepath, 'all_traces_Heads_new.mat')
        trace_variable = sio.loadmat(mat_fname)
        trace_arr = trace_variable['traces']
        is_L = trace_variable['is_L']
        neurons_name = trace_variable['neurons']
        stim_names = trace_variable["stim_names"]
        stimulate_seconds = trace_variable['stim_times']
        stims = trace_variable['stims']
        for idata in range(self.N_dataset):
            ineuron = 0
            for ifile in range(N_length):
                if trace_arr[ifile][0].shape[1] == 42:
                    data = trace_arr[ifile][0][0][idata]
                    if data.shape[0] < 1:
                        activity_datasets[idata][ineuron][:] = np.nan
                    else:
                        activity_datasets[idata][ineuron][0:data[0].shape[0]] = data[0]
                    ineuron+= 1
                    data = trace_arr[ifile][0][0][idata + 21]
                    if data.shape[0] < 1:
                        activity_datasets[idata][ineuron][:] = np.nan
                    else:
                        activity_datasets[idata][ineuron][0:data[0].shape[0]] = data[0]
                    ineuron+= 1
                else:
                    data = trace_arr[ifile][0][0][idata]
                    if data.shape[0] < 1:
                        activity_datasets[idata][ineuron][:] = np.nan
                    else:
                        activity_datasets[idata][ineuron][0:data[0].shape[0]] = data[0]
                    ineuron+= 1
        # add baseline 2
        self.activity_worms = activity_datasets[:,:, T_start:] + 2
        
        neuron_names = []
        for ifile in range(N_length):
            if is_L[ifile][0][0].shape[0] == 42:
                neuron_names.append(neurons_name[ifile][0][0] + 'L')
                neuron_names.append(neurons_name[ifile][0][0] + 'R')
            else:
                neuron_names.append(neurons_name[ifile][0][0])
        self.neuron_names = neuron_names
        
        time = np.arange(start = 0, stop = T * self.step , step = self.step)
        self.odor_list = ['butanone','pentanedione','NaCL']
        for idata in range(self.N_dataset):
            for it_stimu in range(stimulate_seconds.shape[0]):
                tim1_ind = time>stimulate_seconds[it_stimu][0]
                tim2_ind = time<stimulate_seconds[it_stimu][1]
                odor_on = np.multiply(tim1_ind.astype(np.int),tim2_ind.astype(np.int))
                stim_odor = stims[idata][it_stimu] - 1
                odor_datasets[idata][stim_odor][:] = odor_on
                
        self.odor_worms = odor_datasets[:,:, T_start:]
    
    def process_trace(self,trace):
        '''
        Returns activity traces with normalization based on mean and standard devation.
        '''
        worm_trace = (trace - np.nanmean(trace))/np.nanstd(trace)
        return worm_trace

    def __process_activity(self):
        '''
        Returns a list of matrices corresponding to the data missing in the activity columns of the activity_worms dataframes and
        a matrix of the activity with NaNs replaced by 0's
        '''
        missing_data, activity_data = [],[]
        for worm in self.activity_worms:
            worm = (worm - worm.mean())/worm.std()
            act_matrix = worm.values
            missing_act = np.zeros(act_matrix.shape)
            missing_act[np.isnan(act_matrix)] = 1
            act_matrix[np.isnan(act_matrix)] = 0
            missing_data.append(missing_act)
            activity_data.append(act_matrix)
        return activity_data, missing_data
    
    def interpolate_odor(self, odor_worms, upsample_factor):
        '''
        Inputs:
            odor_worms: odor inputs of worm
            upsample_factor: interpolation upsampling factor
        
        Returns:
            list of Dataframes containing upsampled odor input data for each worm 
        '''
        B, N, T = odor_worms.shape
        upsampled_odor_worms = np.zeros((B,N,int(T*upsample_factor)+1))

        for worm_id in range(B):
            interp_matrix = np.zeros(shape = (N, int(T*upsample_factor)+1))
            for i in range(N):
                inter_points = np.linspace(0, T, int(T*upsample_factor)+1)
                interp_matrix[i,:] = np.interp(inter_points, range(T), odor_worms[worm_id][i,:])
            upsampled_odor_worms[worm_id,:,:] = interp_matrix
        return upsampled_odor_worms

def generate_target_mask(target_data):
    '''
    Returns the target with extra dimensions concatenated as zero columns and the
    missing data vector: (target, missing_target)
    '''
    full_missing_target = torch.zeros(size = (target_data.shape))
    full_missing_target[torch.isnan(target_data)] = 1
    full_target = target_data
    full_target[torch.isnan(target_data)] = 0
    return full_target, full_missing_target

class TimeSeriesDataloader(Dataset):
    def __init__(self, data_param_dict, data, window_size):
        super(TimeSeriesDataloader, self).__init__()
        self.data = data
        self.window_size = window_size
        self.max_index = self.data[0].shape[2] // self.window_size[0] - 2
        self.crop_target = int((data_param_dict['k1'] - 1 + (data_param_dict['k2'] - 1)/data_param_dict['up1_factor'] + (data_param_dict['k4'] - 1)/(data_param_dict['up1_factor'] * data_param_dict['up2_factor']))/2)
        self.crop_stim_feature = int((data_param_dict['k3'] - 1   + data_param_dict['k4'] - 1)/2)
        
    def __getitem__(self, index):
        return self.data[0][:,:, (index+1)*self.window_size[0] - self.crop_stim_feature:(index+2)*self.window_size[0] + self.crop_stim_feature], self.data[1][:,:, (index+1)*self.window_size[1] - self.crop_target:(index+2)*self.window_size[1] + self.crop_target], self.data[2][:,:, (index+1)*self.window_size[2] - self.crop_target:(index+2)*self.window_size[2] + self.crop_target], self.data[1][:, :, (index+1)*self.window_size[1]:(index+2)*self.window_size[1]], self.data[2][:, :, (index+1)*self.window_size[2]:(index+2)*self.window_size[2]]

    def __len__(self):
        return self.max_index

class SingleBatchDataloader(Dataset):
    def __init__(self, data, window_size):
        super(SingleBatchDataloader, self).__init__()
        self.data = data
        self.window_size = window_size
        self.total_max_index = self.data[0].shape[0] // self.window_size[0]
        self.current_index = 0
        self.max_index = self.current_index + 1
    
    def __getitem__(self, _ ):
        out0 = self.data[0][self.current_index*self.window_size[0]:(self.current_index+1)*self.window_size[0], :]
        out1 = self.data[1][self.current_index*self.window_size[1]:(self.current_index+1)*self.window_size[1], :]
        out2 = self.data[2][self.current_index*self.window_size[2]:(self.current_index+1)*self.window_size[2], :]
        self.current_index += 1
        if self.current_index >= self.total_max_index:
            self.max_index = 0
        return out0, out1, out2

    def __len__(self):
        return self.max_index

if __name__ == "__main__":
    worm_data_loader = Worm_Data_Loder('../of_data/')
    

```

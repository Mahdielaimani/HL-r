import torch
import torch.nn as nn
import psutil
import os

from res_cell import ResidualCell
from sres_cell import SResidualCell
from inv_res_cell import InvertedResidualCell

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

def get_model_param_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    print(f"Parameter's memory usage {param_size / (1024 ** 2)} MB")

#------------------------------------------------------------------------------------------

class ByteHierarchyEncoder(nn.Module):
    """
    This Encoder for Byte Hierarchy level, Processes input through residual blocks, samples latent variables,
    and uses h as a learnable parameter.
    """
    
    def __init__(self, embedding_size, num_residual_levels=3):
        super(ByteHierarchyEncoder, self).__init__()
        
        self.embedding_size = embedding_size
        self.num_residual_levels = num_residual_levels

        print("PASS1.1")
        print_memory_usage() 

        
        # ResidualCells - all use embedding_size
        self.res_cell1 = ResidualCell(embedding_size,embedding_size)
        self.res_cell2 = ResidualCell(embedding_size,embedding_size)
        self.res_cell3 = SResidualCell(embedding_size,embedding_size)

        print("PASS1.2")
        print_memory_usage()
        get_model_param_size(self.res_cell1)
        get_model_param_size(self.res_cell2)
        get_model_param_size(self.res_cell3)

        # InvertedResidualCells 
        self.ir_cell1 = InvertedResidualCell(embedding_size,embedding_size)
        self.ir_cell2 = InvertedResidualCell(embedding_size,embedding_size)

        print("PASS1.2")
        get_model_param_size(self.ir_cell1)
        get_model_param_size(self.ir_cell2)

        print_memory_usage()
        
        # learnable parameter
        self.register_parameter('h', nn.Parameter(torch.zeros(1, embedding_size, 1), requires_grad=True))        
        print("PASS1.3")
        print_memory_usage()

        # Components for latent variables z1, z2, z3 
        self.z1_mu = nn.Conv1d(embedding_size, embedding_size, kernel_size=3, padding=1)
        self.z1_logvar = nn.Conv1d(embedding_size, embedding_size, kernel_size=3, padding=1)
        
        self.z2_mu = nn.Conv1d(embedding_size, embedding_size, kernel_size=3, padding=1)
        self.z2_logvar = nn.Conv1d(embedding_size, embedding_size, kernel_size=3, padding=1)
        print("PASS1.4")
        
        self.z3_mu = nn.Conv1d(embedding_size, embedding_size, kernel_size=3, padding=1)
        self.z3_logvar = nn.Conv1d(embedding_size, embedding_size, kernel_size=3, padding=1)

    #------------------------------------------------------------------------------------------
    
    def forward(self, x):
        print(f"Input x shape: {x.shape}")
        
        batch_size, num_segments, segment_size, input_embedding_size = x.shape
        
        assert input_embedding_size == self.embedding_size, f"Expected embedding_size {self.embedding_size}, got {input_embedding_size}"
        print(f"Input embedding_size matches model embedding_size: {input_embedding_size}")

        #  batch, num_segments, embedding_size, segment_size
        x_reshaped = x.permute(0, 1, 3, 2)  
        x_reshaped = x_reshaped.reshape(batch_size * num_segments, input_embedding_size, segment_size)

        print(f"x_reshaped shape: {x_reshaped.shape}")

        # Direct processing through residual cells 
        res1_out = self.res_cell1(x_reshaped)
        res2_out = self.res_cell2(res1_out)
        res3_out = self.res_cell3(res2_out)
        
        n_batch_size = batch_size * num_segments
        h_batch = self.h.expand(n_batch_size, -1, -1)
                
        # Process z1
        z1_input = res3_out  
        mu_z1 = self.z1_mu(z1_input)
        logvar_z1 = self.z1_logvar(z1_input)
        z1 = self.reparameterize(mu_z1, logvar_z1)  
        
        combined_output = z1 + h_batch
        
        # Feed to ir1
        ir1_out = self.ir_cell1(combined_output)
        
        # Process z2 
        combined_z2 = res2_out + ir1_out
        mu_z2 = self.z2_mu(combined_z2)
        logvar_z2 = self.z2_logvar(combined_z2)
        z2 = self.reparameterize(mu_z2, logvar_z2)
        
        # Feed z2 
        ir2_out = self.ir_cell2(ir1_out + z2)  
        
        # Process z3  
        combined_z3 = res1_out + ir2_out 
        mu_z3 = self.z3_mu(combined_z3) 
        logvar_z3 = self.z3_logvar(combined_z3)
        z3 = self.reparameterize(mu_z3, logvar_z3)
        
        return {
            'z1': z1, 'mu_z1': mu_z1, 'logvar_z1': logvar_z1,
            'z2': z2, 'mu_z2': mu_z2, 'logvar_z2': logvar_z2,
            'z3': z3, 'mu_z3': mu_z3, 'logvar_z3': logvar_z3,
            'res3_out': res3_out  
        }
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

#-----------------------------------------------------------------------------------------------------------------

class ByteHierarchyDecoder(nn.Module):
    def __init__(self, embedding_size, num_residual_levels=3):
        super(ByteHierarchyDecoder, self).__init__()
        
        self.embedding_size = embedding_size  
        self.num_residual_levels = num_residual_levels
        
        self.register_parameter('h', nn.Parameter(torch.zeros(1, embedding_size, 1), requires_grad=True))

        # Inverted Residual Cells 
        self.ir_cell1 = InvertedResidualCell(embedding_size,embedding_size)
        self.ir_cell2 = InvertedResidualCell(embedding_size,embedding_size)
        self.ir_cell3 = InvertedResidualCell(embedding_size,embedding_size)  

        # No output conv needed

    def forward(self, encoder_outputs, mode='train'):
        if mode == 'generate':
            z1 = self.reparameterize(encoder_outputs['mu_z1'], encoder_outputs['logvar_z1'])
            z2 = self.reparameterize(encoder_outputs['mu_z2'], encoder_outputs['logvar_z2'])
            z3 = self.reparameterize(encoder_outputs['mu_z3'], encoder_outputs['logvar_z3'])
        else:
            z1 = encoder_outputs['z1']
            z2 = encoder_outputs['z2'] 
            z3 = encoder_outputs['z3']

        batch_size = z1.size(0)
        h_batch = self.h.expand(batch_size, -1, -1)
        
        x1 = z1 + h_batch
        ir1_out = self.ir_cell1(x1)
        
        x2 = ir1_out + z2
        ir2_out = self.ir_cell2(x2)
        
        x3 = ir2_out + z3
        output = self.ir_cell3(x3)  # Final output directly from ir_cell3
        
        return {'output': output, 'reconstructed': output}

    def sample(self, encoder_outputs, temp=1.0):
        """
        Generation-mode sampling interface
        Args:
            encoder_outputs: Dict from encoder with mu/logvar for each level
            temp: Temperature parameter for sampling variance
        """
        if temp != 1.0:
            # Apply temperature scaling
            scaled_outputs = {}
            for key in encoder_outputs:
                if 'logvar' in key:
                    scaled_outputs[key] = encoder_outputs[key] + 2 * torch.log(torch.tensor(temp))
                else:
                    scaled_outputs[key] = encoder_outputs[key]
            return self.forward(scaled_outputs, mode='generate')
        else:
            return self.forward(encoder_outputs, mode='generate')

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
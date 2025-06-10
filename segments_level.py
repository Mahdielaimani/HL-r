import torch
import torch.nn as nn

from res_cell import ResidualCell
from special_resCell import SResidualCell
from inv_res_cell import InvertedResidualCell

# --------------------------------------------
class SegmentHierarchyEncoder(nn.Module):
    def __init__(self, embedding_size):
        super(SegmentHierarchyEncoder, self).__init__()
        self.embedding_size = embedding_size

        self.segment_processor = nn.Conv1d(embedding_size, embedding_size, kernel_size=3, padding=1)

        self.res_cell1 = ResidualCell(embedding_size,embedding_size)
        self.res_cell2 = ResidualCell(embedding_size,embedding_size)
        self.res_cell3 = SResidualCell(embedding_size,embedding_size)

        self.ir_cell1 = InvertedResidualCell(embedding_size,embedding_size)
        self.ir_cell2 = InvertedResidualCell(embedding_size,embedding_size)

        # Learnable parameter 
        self.register_parameter('h', nn.Parameter(torch.zeros(1, embedding_size, 1), requires_grad=True))

        self.z1_mu = nn.Conv1d(embedding_size, embedding_size, kernel_size=3, padding=1)
        self.z1_logvar = nn.Conv1d(embedding_size, embedding_size, kernel_size=3, padding=1)
        self.z2_mu = nn.Conv1d(embedding_size, embedding_size, kernel_size=3, padding=1)
        self.z2_logvar = nn.Conv1d(embedding_size, embedding_size, kernel_size=3, padding=1)
        self.z3_mu = nn.Conv1d(embedding_size, embedding_size, kernel_size=3, padding=1)
        self.z3_logvar = nn.Conv1d(embedding_size, embedding_size, kernel_size=3, padding=1)

    #------------------------------------------------------------------------------------------

    def forward(self, byte_res3_out):
        # byte_res3_out: (batch_size, num_segments, embedding_size)
        batch_size, num_segments, input_embedding_size = byte_res3_out.shape
        
        # Verify input dimensions
        assert input_embedding_size == self.embedding_size, f"Expected embedding_size {self.embedding_size}, got {input_embedding_size}"
        
        x = byte_res3_out.permute(0, 2, 1)  # -> (batch_size, embedding_size, num_segments)

        # Direct processing 
        x = self.segment_processor(x)

        res1_out = self.res_cell1(x)
        res2_out = self.res_cell2(res1_out)
        res3_out = self.res_cell3(res2_out)

        h_batch = self.h.expand(batch_size, -1, num_segments)

        # z1
        mu_z1 = self.z1_mu(res3_out)
        logvar_z1 = self.z1_logvar(res3_out)
        z1 = self.reparameterize(mu_z1, logvar_z1)
        z1_out = z1 + h_batch

        ir1_out = self.ir_cell1(z1_out)

        # z2
        mu_z2 = self.z2_mu(res2_out + ir1_out)
        logvar_z2 = self.z2_logvar(res2_out + ir1_out)
        z2 = self.reparameterize(mu_z2, logvar_z2)

        ir2_out = self.ir_cell2(ir1_out + z2)

        # z3
        mu_z3 = self.z3_mu(res1_out + ir2_out)
        logvar_z3 = self.z3_logvar(res1_out + ir2_out)
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

# --------------------------------------------
class SegmentHierarchyDecoder(nn.Module):
    def __init__(self, embedding_size):
        super(SegmentHierarchyDecoder, self).__init__()
        self.embedding_size = embedding_size

        # Learnable parameter 
        self.register_parameter('h', nn.Parameter(torch.zeros(1, embedding_size, 1), requires_grad=True))

        self.ir_cell1 = InvertedResidualCell(embedding_size,embedding_size)
        self.ir_cell2 = InvertedResidualCell(embedding_size,embedding_size)
        self.ir_cell3 = InvertedResidualCell(embedding_size,embedding_size)


    def forward(self, encoder_outputs, mode='train'):
        if mode == 'generate':
            z1 = self.reparameterize(encoder_outputs['mu_z1'], encoder_outputs['logvar_z1'])
            z2 = self.reparameterize(encoder_outputs['mu_z2'], encoder_outputs['logvar_z2'])
            z3 = self.reparameterize(encoder_outputs['mu_z3'], encoder_outputs['logvar_z3'])
        else:
            z1 = encoder_outputs['z1']
            z2 = encoder_outputs['z2']
            z3 = encoder_outputs['z3']

        batch_size, _, seq_len = z1.shape
        h_batch = self.h.expand(batch_size, -1, seq_len)

        x1 = z1 + h_batch
        ir1_out = self.ir_cell1(x1)

        x2 = ir1_out + z2
        ir2_out = self.ir_cell2(x2)

        x3 = ir2_out + z3
        output = self.ir_cell3(x3)  # Direct output from ir_cell3

        # Permute back to (batch_size, num_segments, embedding_size)
        output = output.permute(0, 2, 1)

        return {'output': output, 'reconstructed': output}

    def sample(self, encoder_outputs, temp=1.0):
        if temp != 1.0:
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
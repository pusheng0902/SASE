import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import *
from resnet_SASE import *

class Network(nn.Module):
    def __init__(self, num_classes):
        super(Network, self).__init__()
        self._num_classes = num_classes
        self._criterion = nn.CrossEntropyLoss().cuda()
        model = attention_resnet20(num_classes=self._num_classes)
        self.model = model
        self._initialize_alphas()

    def forward(self, x):
        weights = F.softmax(self.alphas_normal, dim=-1)
        return self.model(x, weights)

    def new(self):
        model_new = Network(self._num_classes).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = 6
        num_ops = len(squeeze_channel)

        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        # self.alphas_normal = Variable(torch.zeros(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parseNkeepAll(alphas_normal, PRIMITIVES):
            gene = []
            indices = torch.max(alphas_normal, dim=1)[1] # A tensor that looks like tensor([4, 4, 4, 2, 2, 0])
            for i, op_idx in enumerate(indices):
                gene.append((PRIMITIVES[i][op_idx], i))
            return gene

        # gene_normal = _parseNkeepAll(self.alphas_normal.data.cpu().numpy(), primitives_chsp['ch_sp'])
        gene_normal = _parseNkeepAll(self.alphas_normal.data, primitives_chsp['ch_sp'])

        return gene_normal

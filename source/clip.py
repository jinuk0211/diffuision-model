import torch
from torch import nn
from torch.nn import functional as f
from attention import selfattention

class CLIPembedding(nn.Module):
                 #49408,768,77
  def __init__(self, nvocab : int, embd : int, ntokens : int):
    super().__init__()

    self.tokenembedding = nn.embedding(nvocab,embd)
    self.positionembedding = nn.Parameter(torch.zeros(ntokens,embd))

  def __call__(self,tokens):
    x = self.tokenembedding(tokens)
    x += self.positionembedding
    return x

class CLIPlayer(nn.Module):
  def __init__(self,numheads,embd):
    super().__init__()
    self.layernorm1 = nn.LayerNorm(embd)
    self.attention = selfattetion(numheads,embd)
    self.layernorm2 = nn.LayerNorm(embd)
    self.linear1 = nn.Linear(embd,4*embd)
    self.linear2 = nn.Linear(4*embd,embd)

  def forward(self,x:torch.Tensor) -> torch.Tensor:
    # bs, seqlen, dim
    residue = x

    x = self.layernorm1(x)
    x = self.attention(x,causal_mask=True) #단어의 지나간 부분 안봄
    x += residue

    ##feedforward layer

    x = residue

    x = self.layernorm2(x)

    x = self.linear(x)

    x = x * torch.sigmoid(1.702 * x) #gelu function

    x = self.linear2(x)

    x += residue

    return x
class CLIP(nn.Module):

  def __init__(self):
    self.embedding = CLIPembedding(49408,768,77)

    self.layers == nn.Module([
        CLIPlayer(12,768) for i in range(12)
    ])

    self.layernorm = nn.LayerNorm(768)

  def __call__(self,token: torch.LongTensor)-> torch.FloatTensor:
    tokens = tokens.type(torch.long)

    #batchsize, seqlen -> batchsize, seqlen, dim
    #state = 현재 text data의 tokenembedding, positonalembedding 한 것
    state = self.embedding(tokens)

    for layer in self.layers:
      state = layer(state)

    #batchsize,seqlen,dim
    output = self.layernorm(state)

    return output

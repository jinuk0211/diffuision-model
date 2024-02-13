import torch
from torch import nn
from torch.nn import functional as f
import math
class selfattention(nn.Module):
                          # embd= embedding_dim
  def __init__(self,numheads,embd,in_bias=True,out_bias=True):
    super().__init__()

    #weight matrix 과정
    self.in_linear = nn.Linear(embd,embd*3,bias=in_bias)
    self.out_linear = nn.Linear(embd,embd,bias= out_bias)

    self.numheads = numheads
    self.headdim = embd / numheads

  def __call__(self,x:torch.Tensor,causal_mask =False):
    # x = batchsize, sequence_length, embeddingdim
    inputshape = x.shape
    batchsize,seqlen, embd = inputshape

    intermin_shape =(batchsize,seqlen,self.numheads,self.headdim)
    q,k,v = self.in_linear(x).chunk(3,dim=-1) #dim = -1 embeddingdim을 3개로 나눔
    
    #배치 사이즈, 시퀸스 길이, 임베딩 디멘션 ->
    #배치 사이즈, 시퀸스 길이, 헤드 개수, 헤드 디멘션 ->
    #배치 사이즈, 해드 개수, 시퀸스 길이, 헤드 디멘션
    q = q.view(intermin_shape).transpose(1,2) 
    k = k.view(intermin_shape).transpose(1,2) 
    v = v.view(intermin_shape).transpose(1,2) 

          # seqlen,headdim @ headdim,seqlen
          #-> seqlen,seqlen
    weight = q @ k.transpose(-1,-2)

    if causal_mask:
      mask = torch.ones_like(weight,dtype=torch.bool).triu(1)
      #triu u = up tril l = low
      #triu 상부 삼각부분 반환 대각선 밑의 부분은 0
      weight.masked_fill(mask,-torch.inf)
      
    #denominator self attention의 /root(d|k) 부분   
    weight /= math.sqrt(self.headdim)

    weight = f.softmax(weight,dim=-1)
    # (batchsize,numheads, seqlen,seqlen) @ (batchsize, numheads, seqlen, headim)
    #-> bs,numheads,seqlen,headdim
    output = weight @ v

    #->bs,seqlen,numheads,headdim
    output = output.transpose(1,2)

    # ->batchsize,seqlen,embeddingdim
    output = output.reshape(inputshape)

    output = self.out_linear(output)

    return output

import os
import math
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.functional import pad,log_softmax
from pathlib import Path
from tqdm import tqdm

# 工作目录，缓存文件放在该目录下
work_dir = Path("./dataset")
# 训练好的模型放在该目录下
model_dir = Path("./drive/MyDrive/model/transformer_checkpoints")
# 上次运行到的地方，如果是第一次运行，为None，如果中途暂停了，下次运行时，指定目前最新的模型即可。
model_checkpoint = None # 'model_10000.pt'

# 如果工作目录不存在，则创建一个
if not os.path.exists(work_dir):
    os.makedirs(work_dir)

# 如果模型目录不存在，则创建一个
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 英文句子的文件路径
en_filepath = './dataset/train.en'
# 中文句子的文件路径
zh_filepath = './dataset/train.zh'


# 获取文件行数的方法
def get_row_count(filepath):
    count = 0
    for _ in open(filepath, encoding='utf-8'):
        count += 1
    return count


# 英文句子数量
en_row_count = get_row_count(en_filepath)
# 中文句子数量
zh_row_count = get_row_count(zh_filepath)
assert en_row_count == zh_row_count, "英文和中文文件行数不一致！"
# 句子数量，主要用于后面显示进度。
row_count = en_row_count

# 句子最大长度，如果句子不够这个长度，则填充，若超出该长度，则裁剪
max_length = 72
print("句子数量为：", en_row_count)
print("句子最大长度为：", max_length)

# 英文和中文词典
en_vocab = None
zh_vocab = None

# batch_size
batch_size = 64
# epochs数量
epochs = 0
# 多少步保存一次模型
save_after_step = 5000

# 是否使用缓存，由于文件较大，初始化动作较慢，所以将初始化好的文件持久化
use_cache = True

# 定义训练设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 加载基础的分词器模型，使用的是基础的bert模型
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
def en_tokenizer(line):
    """
    定义英文分词器
    """
    # 使用bert进行分词，并获取tokens。add_special_tokens是不在结果中增加‘<bos>’和`<eos>`等特殊字符
    return tokenizer.encode(line, add_special_tokens=False).tokens

def yield_en_tokens():
    file = open(en_filepath, encoding='utf-8')
    print("-------开始构建英文词典-----------")
    for line in tqdm(file, desc="构建英文词典", total=row_count):
        yield en_tokenizer(line)
    file.close()

# 指定英文词典缓存文件路径
en_vocab_file = work_dir / "vocab_en.pt"
# 如果使用缓存，且缓存文件存在，则加载缓存文件
if use_cache and os.path.exists(en_vocab_file):
    en_vocab = torch.load(en_vocab_file, map_location="cpu")
# 否则就从0开始构造词典
else:
    # 构造词典
    en_vocab = build_vocab_from_iterator(
        yield_en_tokens(),
        # 最小频率为2，即一个单词最少出现两次才会被收录到词典
        min_freq=2,
        # 在词典的最开始加上这些特殊token
        specials=["<s>", "</s>", "<pad>", "<unk>"],
    )
    # 设置词典的默认index，后面文本转index时，如果找不到，就会用该index填充
    en_vocab.set_default_index(en_vocab["<unk>"])
    # 保存缓存文件
    if use_cache:
        torch.save(en_vocab, en_vocab_file)

# 效果
print("英文词典大小:", len(en_vocab))
print(dict((i, en_vocab.lookup_token(i)) for i in range(10)))

def zh_tokenizer(line):
    """
    定义中文分词器
    """
    return list(line.strip().replace(" ", ""))


def yield_zh_tokens():
    file = open(zh_filepath, encoding='utf-8')
    for line in tqdm(file, desc="构建中文词典", total=row_count):
        yield zh_tokenizer(line)
    file.close()

zh_vocab_file = work_dir / "vocab_zh.pt"
if use_cache and os.path.exists(zh_vocab_file):
    zh_vocab = torch.load(zh_vocab_file, map_location="cpu")
else:
    zh_vocab = build_vocab_from_iterator(
        yield_zh_tokens(),
        min_freq=1,
        specials=["<s>", "</s>", "<pad>", "<unk>"],
    )
    zh_vocab.set_default_index(zh_vocab["<unk>"])
    torch.save(zh_vocab, zh_vocab_file)

# 打印看一下效果
print("中文词典大小:", len(zh_vocab))
print(dict((i, zh_vocab.lookup_token(i)) for i in range(10)))


#定义Dataset
class TranslationDataset(Dataset):
    def __init__(self):
        # 加载英文tokens
        self.en_tokens = self.load_tokens(en_filepath, en_tokenizer, en_vocab, "构建英文tokens", 'en')
        # 加载中文tokens
        self.zh_tokens = self.load_tokens(zh_filepath, zh_tokenizer, zh_vocab, "构建中文tokens", 'zh')

    def __getitem__(self, index):
        return self.en_tokens[index], self.zh_tokens[index]

    def __len__(self):
        return row_count

    def load_tokens(self, file, tokenizer, vocab, desc, lang):
        # 定义缓存文件存储路径
        cache_file = work_dir / "tokens_list.{}.pt".format(lang)
        # 如果使用缓存，且缓存文件存在，则直接加载
        if use_cache and os.path.exists(cache_file):
            print(f"正在加载缓存文件{cache_file}, 请稍后...")
            return torch.load(cache_file, map_location="cpu")
        # 从0开始构建，定义tokens_list用于存储结果
        tokens_list = []
        with open(file, encoding='utf-8') as file:
            # 逐行读取
            for line in tqdm(file, desc=desc, total=row_count):
                # 进行分词
                tokens = tokenizer(line)
                # 将文本分词结果通过词典转成index
                tokens = vocab(tokens)
                # append到结果中
                tokens_list.append(tokens)
        # 保存缓存文件
        if use_cache:
            torch.save(tokens_list, cache_file)

        return tokens_list

dataset = TranslationDataset()
print(dataset.__getitem__(0))

#定义Dataloader
def collate_fn(batch):
    # 定义'<bos>'的index，在词典中为0，所以这里也是0
    bs_id = torch.tensor([0])
    # 定义'<eos>'的index
    eos_id = torch.tensor([1])
    # 定义<pad>的index
    pad_id = 2
    # 用于存储处理后的src和tgt
    src_list, tgt_list = [], []
    # 循环遍历句子对
    for (_src, _tgt) in batch:
        """
        _src: 英语句子
        _tgt: 中文句子
        """
        processed_src = torch.cat(
            # 将<bos>，句子index和<eos>拼到一块
            [
                bs_id,
                torch.tensor(
                    _src,
                    dtype=torch.int64,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    _tgt,
                    dtype=torch.int64,
                ),
                eos_id,
            ],
            0,
        )

        """
        将长度不足的句子进行填充到max_padding的长度的，然后增添到list中
        """
        src_list.append(
            pad(
                processed_src,
                (0, max_length - len(processed_src),),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_length - len(processed_tgt),),
                value=pad_id,
            )
        )
    # 将多个src句子堆叠到一起
    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)

    # tgt_y是目标句子去掉第一个token，即去掉<bos>
    tgt_y = tgt[:, 1:]
    # tgt是目标句子去掉最后一个token
    tgt = tgt[:, :-1]
    # 计算本次batch要预测的token数
    n_tokens = (tgt_y != 2).sum()
    # 返回batch后的结果
    return src, tgt, tgt_y, n_tokens

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
src, tgt, tgt_y, n_tokens = next(iter(train_loader))
src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)



#位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model).to(device)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 将x和positional encoding相加
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
#定义模型
class TranslationModel(nn.Module):

    def __init__(self, d_model, src_vocab, tgt_vocab, dropout=0.1):
        super(TranslationModel, self).__init__()

        # 定义原句子的embedding
        self.src_embedding = nn.Embedding(len(src_vocab), d_model, padding_idx=2)
        # 定义目标句子的embedding
        self.tgt_embedding = nn.Embedding(len(tgt_vocab), d_model, padding_idx=2)
        # 定义posintional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=max_length)
        # 定义Transformer
        self.transformer = nn.Transformer(d_model, dropout=dropout, batch_first=True)
        # 定义预测层，这里并没有定义Softmax，而是把它放在了模型外
        self.predictor = nn.Linear(d_model, len(tgt_vocab))

    def forward(self, src, tgt):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1]).to(device)
        # 掩盖住原句子中<pad>的部分，例如[[False,False,False,..., True,True,...], ...]
        src_key_padding_mask = TranslationModel.get_key_padding_mask(src)
        # 掩盖住目标句子中<pad>的部分
        tgt_key_padding_mask = TranslationModel.get_key_padding_mask(tgt)

        # 对src和tgt进行编码
        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)
        # 给src和tgt的token增加位置信息
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        # 将准备好的数据送给transformer
        out = self.transformer(src, tgt,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)
        return out

    @staticmethod
    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask
        """
        return tokens == 2

if model_checkpoint:
    model = torch.load(model_dir / model_checkpoint)
else:
    model = TranslationModel(256, en_vocab, zh_vocab)
model = model.to(device)
#优化器
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
#损失函数定义
class TranslationLoss(nn.Module):

    def __init__(self):
        super(TranslationLoss, self).__init__()
        # 使用KLDivLoss，不需要知道里面的具体细节。
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = 2

    def forward(self, x, target):
        """
        损失函数的前向传递
        :param x: 将Decoder的输出再经过predictor线性层之后的输出。
                  也就是Linear后、Softmax前的状态
        :param target: tgt_y。也就是label，例如[[1, 34, 15, ...], ...]
        :return: loss
        """

        """
        由于KLDivLoss的input需要对softmax做log，所以使用log_softmax。
        等价于：log(softmax(x))
        """
        x = log_softmax(x, dim=-1)

        """
        构造Label的分布，也就是将[[1, 34, 15, ...]] 转化为:
        [[[0, 1, 0, ..., 0],
          [0, ..., 1, ..,0],
          ...]],
        ...]
        """
        # 首先按照x的Shape构造出一个全是0的Tensor
        true_dist = torch.zeros(x.size()).to(device)
        # 将对应index的部分填充为1
        true_dist.scatter_(1, target.data.unsqueeze(1), 1)
        # 找出<pad>部分，对于<pad>标签，全部填充为0，没有1，避免其参与损失计算。
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)

        # 计算损失
        return self.criterion(x, true_dist.clone().detach())
criteria = TranslationLoss()


step = 0
if model_checkpoint:
    step = int('model_10000.pt'.replace("model_", "").replace(".pt", ""))

#模型训练
model.train()
for epoch in range(epochs):
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for index, data in enumerate(train_loader):
        # 生成数据
        src, tgt, tgt_y, n_tokens = data
        src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)
        # 清空梯度
        optimizer.zero_grad()
        # 进行transformer的计算
        out = model(src, tgt)
        # 将结果送给最后的线性层进行预测
        out = model.predictor(out)


        #计算损失
        loss = criteria(out.contiguous().view(-1, out.size(-1)), tgt_y.contiguous().view(-1)) / n_tokens
        # 计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()
        loop.set_description("Epoch {}/{}".format(epoch, epochs))
        loop.set_postfix(loss=loss.item())
        loop.update(1)

        step += 1

        del src
        del tgt
        del tgt_y

        if step != 0 and step % save_after_step == 0:
            torch.save(model, model_dir / f"model_{step}.pt")

moder=model.eval()
def translate(src: str):
    # 将与原句子分词后，通过词典转为index，然后增加<bos>和<eos>
    src = torch.tensor([0] + en_vocab(en_tokenizer(src)) + [1]).unsqueeze(0).to(device)
    # 首次tgt为<bos>
    tgt = torch.tensor([[0]]).to(device)
    # 一个一个词预测，直到预测为<eos>，或者达到句子最大长度
    for i in range(max_length):
        # 进行transformer计算
        out = model(src, tgt)
        # 预测结果，因为只需要看最后一个词，所以取`out[:, -1]`
        predict = model.predictor(out[:, -1])
        # 找出最大值的index
        y = torch.argmax(predict, dim=1)
        # 和之前的预测结果拼接到一起
        tgt = torch.concat([tgt, y.unsqueeze(0)], dim=1)
        # 如果为<eos>，说明预测结束，跳出循环
        if y == 1:
            break
    # 将预测tokens拼起来
    tgt = ''.join(zh_vocab.lookup_tokens(tgt.squeeze().tolist())).replace("<s>", "").replace("</s>", "")
    return tgt


print(translate("I like playing football. What would you like?"))

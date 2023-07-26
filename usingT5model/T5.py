import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import sentencepiece as spm
import torch
import json

t5_file = r'D:\Python\PycharmProjects\DL_code_practice\usingT5model\T5-base'
spm_path = r'D:\Python\PycharmProjects\DL_code_practice\usingT5model\T5-base\spiece.model'
model = T5ForConditionalGeneration.from_pretrained(t5_file)
#tokenizer = T5Tokenizer.from_pretrained(‘t5-base’)
#tokenizer from pretrained死活下不下来，服了

sp = spm.SentencePieceProcessor(model_file=spm_path)
#print(sp.get_piece_size())

#模型配置
#print(model.config)
#模型架构
#print(model)

device = torch.device('cuda')
model.to(device)

def summarize(text, max_length = 512):
    '''
    text:要生成摘要的文本
    max_length:摘要的最大长度
    '''

    #去除换行符
    presprocess_text = text.strip().replace('\n', '')

    t5_prepared_text = 'summarize: '+ presprocess_text
    print("增加完前缀的文本是：\n", t5_prepared_text)

    #不知道怎么用tokenzier（下不下来），只能用他给的sentencepiece模型模拟了，好烦
    tokenized_text = sp.Encode(t5_prepared_text)
    tokenized_text = torch.tensor([tokenized_text]).to(device)

    summary_ids = model.generate(tokenized_text,
                                 num_beams=4,
                                 no_repeat_ngram_size=2,
                                 min_length=30,
                                 max_length=max_length,
                                 early_stopping=True)

    # 将id转换为输出 summary_ids.shape = [1, 50] 50是最大长度
    summary_ids = summary_ids.cpu().numpy().tolist()
    output = sp.Decode(summary_ids[0])

    return output

text = """
    The United States Declaration of Independence was the first Etext
    released by Project Gutenberg, early in 1971.  The title was stored
    in an emailed instruction set which required a tape or diskpack be
    hand mounted for retrieval.  The diskpack was the size of a large
    cake in a cake carrier, cost $1500, and contained 5 megabytes, of
    which this file took 1-2%.  Two tape backups were kept plus one on
    paper tape.  The 10,000 files we hope to have online by the end of
    2001 should take about 1-2% of a comparably priced drive in 2001.
    """

print("Number of characters:", len(text))
summary = summarize(text, 60)
print("\n\n 摘要后的文章: \n", summary)
#个人感觉摘要结果一般，不咋地，可能和这是一个翻译模型有关

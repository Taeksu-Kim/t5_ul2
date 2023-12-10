import json
import re
from tqdm import tqdm 
from mecab import MeCab

with open('fixed_namu_wiki.json', 'r') as f:
    json_data = json.load(f)

def text_process(title, text):
    fixed_text = re.sub('\n *\[\[분류:.*\]\]', '', text)
    fixed_text = re.sub('\n *\[\[파일:.*\]\]\n?.*', '', fixed_text)
    fixed_text = re.sub('\[각주\]','',fixed_text)
    fixed_text = re.sub('\[youtube\(.*\)\].*\n.*\n*','',fixed_text)
    fixed_text = re.sub('\<tablebgcolor=.*\>','',fixed_text)
    fixed_text = re.sub('\|\|.*\|\|\n','',fixed_text)
    fixed_text = re.sub('\n\{\{\{.*\n.*\n.*\n','',fixed_text)
    fixed_text = re.sub("'''","'",fixed_text)
    fixed_text = re.sub('\(\.\.\.\)','',fixed_text)
    fixed_text = re.sub('~~.{0,30}~~','',fixed_text)
    fixed_text = re.sub('--','',fixed_text)
    fixed_text = re.sub('\n\[include(틀:스포일러)\]','',fixed_text)
    fixed_text = re.sub('\n\|\|<-2><bgcolor=.*','',fixed_text)
    fixed_text = re.sub('\'','',fixed_text)
    fixed_text = re.sub('\{\{\{\#*[a-z]* *','',fixed_text)
    fixed_text = re.sub('\}\}\}','',fixed_text)
    fixed_text = re.sub('\n== ','\n== == ', fixed_text)

    paras = re.split('\n== ',fixed_text)[1:]

    fixed_paras = []

    for i in range(len(paras)):
        para = paras[i]
        para = re.sub('==','=',para)

        pattern = '\[\[[^\]]*\]\]'
        find_iter = re.finditer(pattern, para)

        new_para = '문서 제목: '+title+'\n'

        add_start = 0
        for i in find_iter:
            start_index, end_index = i.span()
            new_para += para[add_start:start_index]
            
            span = i.group()
            span = re.sub('[{\[\[}{\]\]}]','',span)
            span = span.split('|')
            span = span[0] if len(span) == 1 else span[1]

            new_para += span

            add_start = end_index
        if add_start != 0:
            new_para += para[end_index:] 
        
        if add_start == 0:
            new_para += para

        one_splits = new_para.split('\n 1.')

        if len(one_splits) > 2:
            new_para = one_splits[0]

            part_num = 1
            for i in range(1, len(one_splits)):
                new_para += '\n {}.'.format(part_num)
                new_para += one_splits[i]
                part_num += 1
        
        new_para = re.sub('\[[^\[]*\]','',new_para)
        new_para = re.sub(' width=\d*','',new_para)
        new_para = re.sub('\n{3,}','\n\n',new_para)

        fixed_paras.append(new_para.strip())
    return fixed_paras

def preprocessor(text):

    text = re.sub('[^가-힣a-z1-9\s!?;:\'\"@#$%^&*()-=<>,./\[\]{}]', '', text)
    text = text.replace('(, , )','')
    text = text.replace('(, )','')
    text = text.replace('()','')
    text = text.replace(', , )',"")
    text = text.replace(', )',"")
    text = text.replace(', , ',', ')
    text = text.replace('(, ','(')
    # text = re.sub('\(.{0,7}[^가-힣a-z1-9]+.{0,7}\)',"",text)
    return text

mecab = MeCab()

with open("namuwiki_mecab.txt", "w") as f:
    for row_data in tqdm(json_data['data']):
        text = row_data['text']
        title = row_data['title']
        paras = text_process(title, text)
        for para in paras:
            para = preprocessor(para)
            para = para.split('\n')
            for i in range(len(para)):
                para[i] = ' '.join(mecab.morphs(para[i]))
            para = '\n'.join(para)

            f.write('\n\n'+para)
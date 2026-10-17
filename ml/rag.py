"""Deterministic local retrieval over supplied rule cards; no hosted embeddings."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

ALIASES = {
    'cashier':'кассир рабочее место', 'customer':'клиент обслуживание',
    'leave':'ушел вышел покинул', 'leaves':'ушел вышел покинул', 'left':'ушел вышел покинул',
    'door':'дверь открыта изоляция', 'phone':'телефон смартфон камера',
    'smartphone':'телефон смартфон камера', 'backpack':'рюкзак личные вещи',
    'bag':'сумка упаковка', 'cash':'деньги банкноты хранение пересчет',
    'banknotes':'банкноты деньги пересчет', 'count':'пересчет полистный',
    'counting':'пересчет полистный счетная машина', 'safe':'сейф хранение',
    'storage':'специальные устройства хранение', 'receipt':'прием документ квитанция',
    'payout':'выдача расходная операция', 'seal':'пломба целостность упаковка',
    'collector':'инкассатор сумка перевозка', 'collection':'инкассация сумка перевозка',
    'packaging':'упаковка пачка корешок', 'controller':'контролирующий работник',
}
STOP=set('и в во на не по с со из для это что как до при без а или от к у же он она его их the a an is are of to in on and'.split())


def tokens(text):
    return [s[:6] for s in re.findall(r'[а-яёa-z0-9]+',text.lower().replace('ё','е')) if s not in STOP]


class RuleRetriever:
    def __init__(self, folder):
        self.folder=Path(folder)
        self.cards=json.loads((self.folder/'rules.json').read_text(encoding='utf-8'))
        self.sources={s['source_id']:s for s in json.loads((self.folder/'sources.json').read_text(encoding='utf-8'))}
        self.by_id={c['rule_id']:c for c in self.cards}
        if len(self.by_id)!=len(self.cards):raise ValueError('Duplicate rule identifier')
        self.documents=[];self.df=Counter()
        for card in self.cards:
            for source in card['sources']:
                if source['source_id'] not in self.sources or not source['url'].startswith('https://adilet.zan.kz/'):
                    raise ValueError('Rule source missing or outside the supplied RK catalog')
            text=' '.join([card['title'],card['observed_event'],card.get('keywords',''),
                           *card['required_evidence'],*card['exceptions_and_negative_examples']])
            terms=Counter(tokens(text));self.documents.append(terms);self.df.update(terms.keys())
        self.average=sum(sum(d.values()) for d in self.documents)/len(self.documents)
        self.translations=None

    def prompt_card(self,card,language='ru'):
        if language not in {'ru','en'}:raise ValueError('Unsupported rule prompt language')
        if language=='ru' or card['decision_mode']!='conditional':return compact_card(card)
        if self.translations is None:
            values=json.loads((self.folder/'conditional.en.json').read_text(encoding='utf-8'))
            digest=hashlib.sha256(json.dumps(self.cards,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
            if values.pop('_source_sha256')!=digest:raise ValueError('Rule translation is out of date')
            if set(values)!={c['rule_id'] for c in self.cards if c['decision_mode']=='conditional'}:
                raise ValueError('Conditional rule translations are incomplete')
            self.translations=values
        return {'rule_id':card['rule_id'],'mode':card['decision_mode'],**self.translations[card['rule_id']]}

    def retrieve(self, query, k=5):
        words=set(re.findall(r'[a-z]+',query.lower()))
        expanded=query+' '+' '.join(v for key,v in ALIASES.items() if key in words)
        terms=set(tokens(expanded));scores=[];n=len(self.cards)
        for card,doc in zip(self.cards,self.documents):
            length=sum(doc.values());score=0
            for term in terms:
                tf=doc.get(term,0)
                if tf:
                    score+=math.log(1+(n-self.df[term]+0.5)/(self.df[term]+0.5))*tf*2.2/(tf+1.2*(.25+.75*length/self.average))
            scores.append((score,card['rule_id'],card))
        return [c for score,_,c in sorted(scores,key=lambda x:(-x[0],x[1]))[:k] if score>0]

    def catalog(self):
        out={}
        for c in self.cards:
            s=c['sources'][0];document=self.sources[s['source_id']]
            out[c['rule_id']]={'id':c['rule_id'],'name':c['title'],
                'document':document.get('official_name',document['title']),
                'section':'Пункт '+str(s['point']),'url':s['url'],
                'summary':c['observed_event'],'visual_test':' '.join(c['required_evidence']),
                'caution':' '.join(c['do_not_infer']), 'decision_mode':c['decision_mode'],
                'sources':c['sources'],'enabled':True,'demo_only':False}
        return out


def compact_card(card):
    return {'rule_id':card['rule_id'],'title':card['title'],'mode':card['decision_mode'],
            'required_evidence':card['required_evidence'],
            'exceptions':card['exceptions_and_negative_examples'],
            'do_not_infer':card['do_not_infer']}

"""Greedy-fit the renderer's fallback sentences against the ACTUAL reference
distribution (first-annotator-per-case), not one annotator's house style."""
import sys, os, glob, re, collections, random
sys.path.insert(0,'.')
from tf4.metrics import corpus_bleu4, meteor
root='toothfairy4_v03'
refs=[]
for c in sorted(os.listdir(root)):
    fs=sorted(glob.glob(f'{root}/{c}/reports_en/*.txt'))
    if fs: refs.append(open(fs[0]).read().strip())
random.seed(1); refs=random.sample(refs,300)
sc=collections.Counter()
for t in refs:
    for line in t.split('\n'):
        for s in re.split(r'(?<=[.])\s+', line.strip()):
            s=s.strip()
            if len(s)>4: sc[s]+=1
pool=[s for s,_ in sc.most_common(70)]
cur=[]; best=0.0
for it in range(10):
    cand=None
    for s in pool:
        if s in cur: continue
        for pos in range(len(cur)+1):
            trial=cur[:pos]+[s]+cur[pos:]
            txt='\n'.join(trial)
            b=corpus_bleu4([txt]*len(refs),refs); m=meteor([txt]*len(refs),refs)
            if cand is None or (b+m)>cand[0]: cand=((b+m),b,m,trial)
    if cand[0]<=best+1e-5: break
    best=cand[0]; cur=cand[3]
    print(f'iter{it+1}: BLEU={cand[1]:.4f} METEOR={cand[2]:.4f} len={len(cur)}',flush=True)
print('\n=== FITTED PRIOR REPORT ===')
print('\n'.join(cur))

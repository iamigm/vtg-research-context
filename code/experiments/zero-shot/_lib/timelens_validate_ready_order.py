import time, json, torch
class Rows:
    def __len__(self): return 8
    def __getitem__(self,i):
        time.sleep(1.5 if i==0 else .01)
        return {'id':i,'tensor':torch.arange(10)+i}
def main():
    outputs=[]
    for ordered in [True,False]:
        loader=torch.utils.data.DataLoader(Rows(),batch_size=None,num_workers=2,in_order=ordered,multiprocessing_context='spawn')
        outputs.append(list(loader))
    assert [r['id'] for r in outputs[0]]==list(range(8))
    assert outputs[1][0]['id']!=0
    a={r['id']:r['tensor'] for r in outputs[0]}; b={r['id']:r['tensor'] for r in outputs[1]}
    assert a.keys()==b.keys()
    assert all(torch.equal(a[k],b[k]) for k in a)
    print(json.dumps({'same_samples_and_tensors':True,'ordered':[r['id'] for r in outputs[0]],'ready_order':[r['id'] for r in outputs[1]]}))
if __name__=='__main__': main()

"""Config/shape based forward matmul FLOPs. One multiply-add counts as two."""
def text_flops(c,lengths,generated=0,lm_head=False):
    h=c['hidden_size'];m=c['intermediate_size'];layers=c['num_hidden_layers']
    heads=c['num_attention_heads'];dh=c.get('head_dim',h//heads)
    q=heads*dh;kv=c.get('num_key_value_heads',heads)*dh
    linear=2*(h*(q+2*kv)+q*h+3*h*m)
    tokens=sum(lengths)+max(0,generated-1)
    pairs=sum(n*(n+1)//2 for n in lengths)
    if generated:pairs+=sum(lengths[0]+j for j in range(1,generated))
    result=layers*(linear*tokens+4*q*pairs)
    if lm_head:result+=2*h*c['vocab_size']*generated
    return int(result)

def vision_flops(c,grids):
    h=c['hidden_size'];m=c['intermediate_size'];depth=c['depth']
    merge=c['spatial_merge_size']**2
    patches=sum(t*y*x for t,y,x in grids)
    pairs=sum(t*(y*x)**2 for t,y,x in grids)
    patch_dim=c['in_channels']*c['temporal_patch_size']*c['patch_size']**2
    patch=2*patches*patch_dim*h
    blocks=2*patches*(4*h*h+2*h*m)*depth+4*h*pairs*depth
    merger=2*(patches//merge)*((h*merge)**2+h*merge*c['out_hidden_size'])
    return int(patch+blocks+merger*(1+len(c.get('deepstack_visual_indexes',[]))))

def clip_flops(c,n_frames,text_tokens):
    def transformer(cfg,n,causal):
        h=cfg['hidden_size'];m=cfg['intermediate_size'];layers=cfg['num_hidden_layers']
        pairs=n*(n+1)//2 if causal else n*n
        return layers*(2*n*(4*h*h+2*h*m)+4*h*pairs)
    v=c['vision_config'];t=c['text_config'];patch=v['patch_size'];side=v['image_size'];n=(side//patch)**2
    vision=transformer(v,n+1,False)+2*n*3*patch**2*v['hidden_size']+2*v['hidden_size']*c['projection_dim']
    text=transformer(t,text_tokens,True)+2*t['hidden_size']*c['projection_dim']
    cosine=2*n_frames*c['projection_dim']
    return dict(vision_flops=int(vision*n_frames),text_flops=int(text),cosine_flops=int(cosine),
                total_flops=int(vision*n_frames+text+cosine))

def qwen_flops(c,grids,lengths,generated=0,lm_head=False,cosine_frames=0):
    vision=vision_flops(c['vision_config'],grids)
    text=text_flops(c['text_config'],lengths,generated,lm_head)
    cosine=2*cosine_frames*c['text_config']['hidden_size']
    return dict(vision_flops=vision,text_flops=text,cosine_flops=cosine,total_flops=vision+text+cosine)

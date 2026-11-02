"""Rebuild the GPT2Toy architecture diagram from the current toy model."""
from pathlib import Path
from html import escape
import re
import sys
import xml.etree.ElementTree as ET

import torch
import cairosvg

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT))
from gpt2toy import GPTModel, generate_text_simple
from gpt2toy_data import SimpleTokenizerV2
from run_gpt2toy import REPEATED_CONTEXT, SEED, START_CONTEXT, TOY_CONFIG, TOY_VOCAB

torch.manual_seed(SEED)
model = GPTModel(TOY_CONFIG).eval()
tokenizer = SimpleTokenizerV2(TOY_VOCAB)
vocab_size = model.tok_emb.num_embeddings
context_length = model.pos_emb.num_embeddings
emb_dim = model.pos_emb.embedding_dim
num_layers = len(model.trf_blocks)
first_block = model.trf_blocks[0]
qkv_projection_count = len((first_block.att.W_query, first_block.att.W_key, first_block.att.W_value))
num_heads = first_block.att.num_heads
head_dim = first_block.att.head_dim
ffn_dim = first_block.ff.layers[0].out_features
drop_rate = model.drop_emb.p
qkv_bias = first_block.att.W_query.bias is not None
output_bias = first_block.att.out_proj.bias is not None
weight_tied = model.tok_emb.weight.data_ptr() == model.out_head.weight.data_ptr()
param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
token_embedding_params = model.tok_emb.weight.numel()
position_embedding_params = model.pos_emb.weight.numel()
attention_params = sum(p.numel() for block in model.trf_blocks for p in block.att.parameters())
ffn_params = sum(p.numel() for block in model.trf_blocks for p in block.ff.parameters())
norm_params = sum(
    p.numel()
    for block in model.trf_blocks
    for module in (block.norm1, block.norm2)
    for p in module.parameters()
)
norm_params += sum(p.numel() for p in model.final_norm.parameters())
lm_head_params = sum(p.numel() for p in model.out_head.parameters())
causal_mask_shape = tuple(first_block.att.mask.shape)
start_ids = tokenizer.encode(START_CONTEXT)
repeated_ids = tokenizer.encode(REPEATED_CONTEXT)
repeated_token = REPEATED_CONTEXT.split()[-1]
repeated_positions = [i for i, token_id in enumerate(repeated_ids) if tokenizer.int_to_str[token_id] == repeated_token]
repeated_input = torch.tensor([repeated_ids], dtype=torch.long)
with torch.no_grad():
    repeated_logits = model(repeated_input)[0]
repeated_next_ids = [int(repeated_logits[pos].argmax().item()) for pos in repeated_positions]
repeated_logits_equal = torch.equal(
    repeated_logits[repeated_positions[0]], repeated_logits[repeated_positions[1]]
)
repeat_logits_label = 'logits equal' if repeated_logits_equal else 'logits differ'
repeat_prediction_labels = ", ".join(
    f"pos {pos} → {tokenizer.int_to_str[token_id]!r} (ID {token_id})"
    for pos, token_id in zip(repeated_positions, repeated_next_ids)
)
initial_ids = torch.tensor([start_ids], dtype=torch.long)
generated_ids = generate_text_simple(
    model, initial_ids, max_new_tokens=2, context_size=context_length
)[0].tolist()
generated_text = tokenizer.decode(generated_ids)
generation_input_lengths = [min(len(start_ids) + step, context_length) for step in range(2)]
final_sequence_length = len(generated_ids)

W, H = 1860, 1510
C = dict(bg='#020617', panel='#0b1426', text='#e2e8f0', muted='#a3b3c9',
         cyan='#22d3ee', green='#34d399', purple='#a78bfa', amber='#fbbf24', rose='#fb7185', gray='#64748b')
lines = []
def put(s):
    """追加 SVG（Scalable Vector Graphics，可缩放矢量图形）标记，供最终组装图像。"""
    lines.append(s)
def text(x,y,s,size=18,color='text',weight=400,anchor='start',extra=''):
    """绘制文本标签；转义内容，避免特殊字符破坏 SVG 的 XML 结构。"""
    put(f'<text x="{x}" y="{y}" font-size="{size}" fill="{C.get(color,color)}" font-weight="{weight}" text-anchor="{anchor}" {extra}>{escape(s)}</text>')
def rect(x,y,w,h,stroke='gray',fill=None,dash=False,r=12):
    """绘制可配置描边、填充和圆角的矩形，作为架构节点或背景面板。"""
    d=' stroke-dasharray="7 6"' if dash else ''
    put(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill or C["panel"]}" stroke="{C.get(stroke,stroke)}" stroke-width="1.4"{d}/>')
def arrow(path,color='cyan',dash=False):
    """绘制带箭头的连接线，用于表示前向数据流、残差旁路或自回归反馈。"""
    d=' stroke-dasharray="6 6"' if dash else ''
    put(f'<path d="{path}" fill="none" stroke="{C[color]}" stroke-width="2" marker-end="url(#{color})"{d}/>')
def box(x,y,w,h,title,details=(),color='cyan',tip='',title_size=20):
    """组合节点边框、标题、说明与悬浮提示，展示一个模型计算模块。"""
    put('<g class="node">')
    if tip: put(f'<title>{escape(tip)}</title>')
    rect(x,y,w,h,color)
    text(x+18,y+29,title,title_size,color,600)
    for i,s in enumerate(details): text(x+18,y+57+i*24,s,17,'muted')
    put('</g>')
def section(x,y,w,h,num,title,sub,color):
    """绘制总览、解码器块或注意力细节区域的分区标题和容器。"""
    rect(x,y,w,h,'#27374f',fill='#071020')
    text(x+22,y+34,num,17,color,700)
    text(x+58,y+34,title,23,'text',600)
    text(x+22,y+64,sub,16,'muted')

put(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">')
put('<title id="title">GPT2Toy 整体架构与张量数据流</title>')
put('<desc id="desc">GPT 解码器架构：词元与绝对位置嵌入、前置层归一化、因果多头自注意力、逐位置前馈网络、残差连接、语言模型头与自回归生成。</desc>')
put('<style>text{font-family:"Segoe UI","Microsoft YaHei",sans-serif} .node:hover rect{stroke-width:2.5}</style>')
put('<defs>')
for k in ['cyan','green','purple','amber','rose','gray']:
    put(f'<marker id="{k}" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="{C[k]}"/></marker>')
put('<pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse"><path d="M32 0H0V32" fill="none" stroke="#12203a" stroke-width="0.5"/></pattern>')
put('</defs>')
put(f'<rect width="{W}" height="{H}" fill="{C["bg"]}"/><rect width="{W}" height="{H}" fill="url(#grid)"/>')
text(40,56,'GPT2Toy',38,'text',700)
text(245,56,'解码器式 Transformer · 核心架构',29,'text',500)
text(40,94,'Decoder-only  /  Pre-LayerNorm  /  Causal Self-Attention  /  Autoregressive Generation',19,'muted')
chips=[('词表大小 V',str(vocab_size)),('上下文长度 C',str(context_length)),
       ('嵌入维度 d_model',str(emb_dim)),('头数 × 每头维度',f'{num_heads} × {head_dim}'),
       ('解码器层数 L',str(num_layers)),('可学习参数',str(param_count))]
for i,(a,b) in enumerate(chips):
    x=40+i*300
    rect(x,122,280,59,'#27374f')
    text(x+16,158,a,17,'muted')
    text(x+260,159,b,24,'cyan',600,'end')
text(40,200,
     f'Same-token probe: “{REPEATED_CONTEXT}” → IDs {repeated_ids}; {repeated_token} at {repeated_positions}; {repeat_logits_label}; measured next-token argmax: {repeat_prediction_labels}',
     15,'amber')

section(40,210,500,1140,'01','GPT 前向与自回归生成','GPTModel · 词元序列 → 隐状态 → 词表得分','cyan')
section(590,210,600,1140,'02','解码器块 (Decoder Block)',f'TransformerBlock × {num_layers} · 前置层归一化 Pre-LN','green')
section(1240,210,580,1140,'03','因果多头自注意力',f'Causal Multi-Head Self-Attention · {num_heads} heads × {head_dim} dims','purple')

# Main flow. Position IDs are generated locally, not vocabulary IDs.
arrow('M290 380 V410')
arrow('M290 480 V502 H180 V535')
arrow('M180 655 V673 H290 V690')
arrow('M400 655 V673 H290 V690')
arrow('M290 780 V810')
arrow('M290 900 V950')
arrow('M290 1020 V1070')
arrow('M290 1160 V1210')
arrow('M80 1260 H60 V443 H80','amber')
arrow('M500 855 H565 V245 H587','gray',True)
box(80,290,420,90,'分词器 (Tokenizer)',['SimpleTokenizerV2 · 正则拆词与标点分隔',f'“{START_CONTEXT}” → 词元 ID {start_ids}'])
box(80,410,420,70,'上下文窗口 (Context Window)',[f'idx[:, -{context_length}:] → 输入词元索引 [B,T]'],tip='generate_text_simple 每步输入整个裁剪后的上下文。')
box(80,535,200,120,'词元嵌入',['Token Embedding',f'tok_emb [{vocab_size},{emb_dim}]',f'→ [B,T,{emb_dim}]'])
box(300,535,200,120,'绝对位置嵌入',['Position Embedding',f'pos_emb [{context_length},{emb_dim}]',f'→ [T,{emb_dim}]（可学习）'],tip=f'Learned Absolute Positional Embedding：用 0..T-1 查可学习的位置表，形状为 {tuple(model.pos_emb.weight.shape)}；不根据 Token ID 查表。',title_size=19)
text(300,518,'arange(T)：0…T−1',16,'muted')
box(80,690,420,90,'嵌入融合 (Embedding Sum)',[f'词元向量 + 位置向量 → [B,T,{emb_dim}]',f'嵌入随机失活：Dropout (p={drop_rate:g})'])
box(80,810,420,90,'解码器块 (TransformerBlock)',['前置层归一化 · 因果自注意力 · 前馈网络',f'n_layers = {num_layers}；沿残差流依次堆叠'],'green')
box(80,950,420,70,'最终层归一化 (final_norm)',[f'Final LayerNorm · [B,T,{emb_dim}] → [B,T,{emb_dim}]'],'green')
box(80,1070,420,90,'语言模型头 (LM Head)',[f'out_head: Linear({emb_dim} → {vocab_size})，'+('含偏置' if model.out_head.bias is not None else '无偏置'),f'词表得分 logits: [B,T,{vocab_size}]；'+('权重绑定' if weight_tied else '权重未绑定')],'cyan')
box(80,1210,420,100,'贪婪解码 (Greedy Decoding)',[f'最后位置 logits[:, -1, :] → [B,{vocab_size}]',f'argmax → 下一词元 [B,1] → 追加到序列'],'amber')
text(100,1330,'自回归反馈：追加词元，重新预测下一词元',16,'amber')

# Block flow and bypasses; the bypass carries the pre-normalization tensor.
arrow('M930 295 V330','green')
arrow('M930 420 V470','green')
arrow('M930 560 V595','green')
arrow('M930 640 V680','green')
arrow('M930 750 V790','green')
arrow('M930 880 V930','green')
arrow('M930 1050 V1085','green')
arrow('M930 1130 V1170','green')
arrow('M930 1240 V1280','green')
arrow('M930 299 H635 V710 H720','rose')
arrow('M930 760 H665 V1200 H720','rose')
arrow('M1140 515 H1214 V245 H1237','gray',True)
text(930,293,f'x: [B,T,{emb_dim}]',18,'green',600,'middle')
box(720,330,420,90,'前置层归一化 (norm1)',['Pre-LayerNorm · 每个词元沿特征维归一化','γ ⊙ (x−μ) / √(σ²+10⁻⁵) + β'],'green')
box(720,470,420,90,'多头自注意力 (att)',['Multi-Head Self-Attention · 因果掩码',f'上下文信息聚合：[B,T,{emb_dim}] → [B,T,{emb_dim}]'],'purple')
rect(720,595,420,45,'gray'); text(930,624,'残差分支随机失活 (drop_shortcut)',18,'muted',400,'middle')
rect(720,680,420,70,'rose'); text(930,705,'残差相加 ① (Residual Add)',18,'rose',600,'middle'); text(930,733,'x₁ = x + Attention(LN₁(x))',19,'text',400,'middle')
box(720,790,420,90,'前置层归一化 (norm2)',[f'Pre-LayerNorm · [B,T,{emb_dim}] → [B,T,{emb_dim}]',f'可学习缩放 γ[{emb_dim}] 与平移 β[{emb_dim}]'],'green')
box(720,930,420,120,'逐位置前馈网络 (FFN / MLP)',[f'Linear({emb_dim} → {ffn_dim})  →  GELU  →  Linear({ffn_dim} → {emb_dim})',f'ff：[B,T,{emb_dim}] → [B,T,{ffn_dim}] → [B,T,{emb_dim}]','GELU：高斯误差线性单元（tanh 近似）'],'green')
rect(720,1085,420,45,'gray'); text(930,1114,'残差分支随机失活 (drop_shortcut)',18,'muted',400,'middle')
rect(720,1170,420,70,'rose'); text(930,1195,'残差相加 ② (Residual Add)',18,'rose',600,'middle'); text(930,1223,'x₂ = x₁ + FFN(LN₂(x₁))',19,'text',400,'middle')
text(930,1307,f'Residual stream output x₂: [B,T,{emb_dim}]',20,'green',600,'middle')
text(620,555,'Identity shortcut · x',17,'rose',400,'middle',extra='transform="rotate(-90 620 555)"')
text(650,1000,'Identity shortcut · x₁',17,'rose',400,'middle',extra='transform="rotate(-90 650 1000)"')

# Attention details.
for a,b in [(380,420),(510,550),(650,690),(780,820),(890,930)]:
    arrow(f'M1530 {a} V{b}','purple')
box(1280,290,500,90,'① 查询 / 键 / 值投影 (Q / K / V)',[f'Query / Key / Value · 独立 Linear({emb_dim} → {emb_dim})',f'{qkv_projection_count} × [B,T,{emb_dim}]；各权重 [{emb_dim},{emb_dim}]，'+('含偏置' if qkv_bias else '无偏置')],'purple',tip='对应 W_query、W_key、W_value 三个可学习线性投影。')
box(1280,420,500,90,'② 注意力头拆分 (Split Heads)',[f'view + transpose: [B,T,{emb_dim}] → [B,{num_heads},T,{head_dim}]',f'每个头均保留序列长度 T；每头特征维度为 {head_dim}'],'purple')
box(1280,550,500,100,'③ 缩放点积注意力得分',[f'QKᵀ / √d_head → [B,{num_heads},T,T]；d_head = {head_dim}',f'因果掩码形状 {causal_mask_shape}；未来位置 j > i → −∞'],'purple',tip=f'Scaled Dot-Product Attention：当前实现先对 QKᵀ 应用 {causal_mask_shape} 因果掩码，再除以 √{head_dim}。')
box(1280,690,500,90,'④ 注意力权重归一化 (Softmax)',[f'沿 key 维度归一化 → A: [B,{num_heads},T,T]',f'注意力随机失活 (Attention Dropout, p={first_block.att.dropout.p:g})'],'purple')
box(1280,820,500,70,'⑤ 值向量加权聚合 (Weighted Sum)',[f'A @ V: [B,{num_heads},T,T] @ [B,{num_heads},T,{head_dim}] → [B,{num_heads},T,{head_dim}]'],'purple')
box(1280,930,500,100,'⑥ 多头拼接与输出投影',[f'Concat Heads → 拼接结果 [B,T,{emb_dim}]',f'out_proj: Linear({emb_dim} → {emb_dim})，'+('含偏置' if output_bias else '无偏置')+f' → [B,T,{emb_dim}]'],'purple')
rect(1280,1060,500,260,'#334155')
text(1298,1093,f'可学习参数分布 · {param_count}',20,'text',600)
rows=[(f'词元嵌入 {token_embedding_params} + 位置嵌入 {position_embedding_params}',str(token_embedding_params+position_embedding_params)),
      (f'自注意力 ({num_layers} 层)',str(attention_params)),
      (f'前馈网络 ({num_layers} 层, {emb_dim}→{ffn_dim}→{emb_dim})',str(ffn_params)),
      (f'层归一化 ({2*num_layers+1} 个)',str(norm_params)),
      (f'语言模型头 ({vocab_size}×{emb_dim})',str(lm_head_params))]
for i,(name,value) in enumerate(rows):
    text(1298,1128+i*31,name,17,'muted')
    text(1760,1128+i*31,value,18,'cyan',600,'end')
text(1298,1304,f'mask buffer {causal_mask_shape}；Dropout / GELU 无可学习参数',16,'muted')

# Compact reading key; the diagram focuses on GPT computation only.
text(40,1395,'读图约定',18,'text',600)
for x,color,label in [(155,'cyan','模型前向'),(380,'rose','残差连接'),(605,'amber','自回归反馈'),(870,'gray','细节展开')]:
    arrow(f'M{x} 1389 H{x+40}',color,color=='gray');text(x+53,1395,label,17,'muted')
text(1170,1395,f'B：批大小  ·  T：序列长度  ·  本例 B=1，输入 T={"→".join(map(str,generation_input_lengths))}，输出长 {final_sequence_length}',17,'muted')
text(40,1440,f'实例设定：seed={SEED}，随机初始化且未训练；Dropout p={drop_rate:g}；嵌入与 LM Head 权重{"绑定" if weight_tied else "独立"}；无 KV Cache。',18,'muted')
text(40,1480,f'依据 gpt2toy.py / run_gpt2toy.py 当前符号与 CPU 模型实例生成 · mask={causal_mask_shape} · 参数={param_count} · 示例输出：{generated_text}',15,'muted')
put('</svg>')
svg='\n'.join(lines)
ET.fromstring(svg)
(OUT/'gpt2toy-architecture.svg').write_text(svg,encoding='utf-8')

# Preserve the architecture-diagram skill's export implementation and pinned CDN scripts.
template=Path('C:/Users/qiuxueliang/.agents/skills/architecture-diagram/resources/template.html').read_text(encoding='utf-8')
cdn='\n'.join(re.findall(r'<script src=.*?</script>',template)).replace('<script src=', '<script defer src=')
export_script=template[template.rfind('  <script>'):template.rfind('</body>')]
export_script=export_script.replace("'diagram.png'","'gpt2toy-architecture.png'").replace("'diagram.pdf'","'gpt2toy-architecture.pdf'")
html='''<!doctype html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GPT2Toy · 整体架构</title>'''+cdn+'''
<style>
*{box-sizing:border-box}body{margin:0;background:#020617;color:#e2e8f0;font:16px "Segoe UI","Microsoft YaHei",sans-serif;padding:32px}.container{max-width:1900px;margin:auto}header{display:flex;align-items:center;gap:18px;margin:0 0 20px}h1{font-size:24px;margin:0}a{color:#67e8f9}.toolbar{display:flex;gap:8px;margin-left:auto;align-items:center}.toolbar-actions{display:none;gap:8px}.toolbar.expanded .toolbar-actions{display:flex}button{background:#142238;color:#e2e8f0;border:1px solid #334155;border-radius:6px;padding:8px 12px;cursor:pointer}button:hover{border-color:#22d3ee}.toolbar-toggle{font-size:20px}.controls{display:flex;gap:8px;align-items:center;margin:12px 0}.hint{color:#a3b3c9;font-size:14px}.viewport{overflow:auto;border:1px solid #27374f;border-radius:14px}svg{display:block;width:100%;height:auto;min-width:1000px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin-top:20px}.card{background:#0b1426;border:1px solid #27374f;border-radius:12px;padding:20px;line-height:1.8}.card h2{font-size:18px;margin:0 0 8px;color:#34d399}.card p{margin:0;color:#a3b3c9}.sources{line-height:2;color:#a3b3c9;margin-top:18px} @media(max-width:900px){body{padding:16px}.cards{grid-template-columns:1fr}header{flex-wrap:wrap}}@media print{.toolbar,.controls{display:none!important}body{padding:0}}
</style></head><body><div class="container" id="report-container"><header><h1>GPT2Toy 架构阅读器</h1><span class="hint">从总览到计算细节；悬停节点查看说明</span><div class="toolbar"><div class="toolbar-actions"><button onclick="copyAsImage(this)">复制图片</button><button onclick="downloadPNG(this)">PNG</button><button onclick="downloadPDF(this)">PDF</button></div><button class="toolbar-toggle" onclick="this.parentElement.classList.toggle('expanded')" aria-label="导出选项">⋯</button></div></header>
<div class="controls"><button onclick="zoom('100%')">适应宽度</button><button onclick="zoom('1860px')">原始尺寸</button><button onclick="zoom('2400px')">放大细节</button><span class="hint">静态图可离线阅读；工具栏导出需加载 CDN。亦可直接打开下方现成 PNG / SVG。</span></div>
<div class="viewport">'''+svg+'''</div><div class="cards">
<article class="card"><h2>自注意力与前馈网络</h2><p>因果自注意力（Causal Self-Attention）在当前位置及历史位置之间聚合信息；逐位置前馈网络（Position-wise FFN / MLP）独立变换每个位置的特征。两者共享同一条 __EMB_DIM__ 维残差流。</p></article>
<article class="card"><h2>前置层归一化与残差连接</h2><p>Pre-LN 把层归一化放在注意力或前馈子层之前；恒等旁路保留子层输入，再与子层输出相加。LayerNorm 沿特征维计算统计量，学习缩放 γ 与平移 β；最终层归一化位于全部解码器块之后。</p></article>
<article class="card"><h2>语言模型头与自回归生成</h2><p>语言模型头（LM Head）把隐藏状态投影为每个位置的词表得分（logits）。生成时只取最后位置，用贪婪解码选出下一词元，追加后再次前向。本例生成 IDs <code>__GENERATED_IDS__</code>（<code>__GENERATED_TEXT__</code>）；产生 end-of-text token 不会停止循环。本例未实现 KV Cache，每步重新计算上下文。</p></article>
</div><div class="sources">文件：<a href="gpt2toy-architecture.png">高清 PNG</a> · <a href="gpt2toy-architecture.svg">SVG 矢量图</a><br>源码：<a href="../gpt2toy.py">gpt2toy.py</a>。参数数量、维度与示例输出由 run_gpt2toy.py 常量和 CPU 初始化模型计算生成。</div>
</div><script>function zoom(w){document.querySelector('.viewport svg').style.width=w}</script>'''+export_script+'</body></html>'
html=(html.replace('__EMB_DIM__',str(emb_dim))
          .replace('__GENERATED_IDS__',escape(str(generated_ids)))
          .replace('__GENERATED_TEXT__',escape(generated_text)))
(OUT/'gpt2toy-architecture.html').write_text(html,encoding='utf-8')
cairosvg.svg2png(bytestring=svg.encode('utf-8'), write_to=str(OUT/'gpt2toy-architecture.png'))
print('Generated SVG + HTML + PNG; XML valid.')

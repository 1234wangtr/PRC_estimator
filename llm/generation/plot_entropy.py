
from pathlib import Path
import json


def ext(res_path):
    _det = []
    _correct_rate = []
    _temperature = []
    data_entries = []
    from collections import defaultdict
    temps_cnt = defaultdict(int)
    for fn in sorted(Path(res_path).rglob("*.json"), key=lambda x: int(x.stem)):
        with fn.open() as f:
            data = json.load(f)
            temperature = data['generation_config']['temperature'] if 'generation_config' in data else 1.0
            dets = data['det']
            sentences = data['watermark_sentence']
            assert len(dets) == 16, f"Expected 16 sentences, got {len(dets)} in {fn}"
            assert temps_cnt[temperature] < 64, f"Already have 64 entries for temperature {temperature}"
            print(f"Processing {fn} with temperature {temperature}")
            temps_cnt[temperature] += 1
            _temperature.append(temperature)
            correct_rates = data['correct_rate']
            avg_entropys = data['avg_entropy']
            _det.append(dets)
            _correct_rate.append(correct_rates)
            data_entries.extend(
            {
                'temperature': temperature,
                'det': det,
                'correct_rate': correct_rate,
                'avg_entropy': avg_entropy,
                'prompt_index': i,
                'sentence': sentence
            }
            for i, (det, correct_rate, avg_entropy,sentence) in enumerate(
                zip(dets, correct_rates, avg_entropys, sentences)
            )
        )
            
    return _det,_correct_rate,_temperature,data_entries

_,_,_, data_entries = ext("llm/data/gen_result")

import pandas as pd
df = pd.DataFrame(data_entries)


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", font=["Times New Roman", "Droid Sans Fallback"],font_scale=1.8)
plt.figure(figsize=(5,3))
axes = sns.relplot(data=df, x='avg_entropy', y='correct_rate', col='det', hue='temperature', s=45,alpha=0.75,palette='Set1',)

axes.set_titles("Watermark Detection: {col_name}")

axes.set_axis_labels("Average Token Entropy (bit)", "Bit Accuracy")
axes.legend.set_title("temperature")
axes.legend.set_bbox_to_anchor((0.635, 0.45))
axes.legend.set_frame_on(True)
for ax in axes.axes.flat:
    ax.grid(True, linestyle='--', alpha=0.6)

df_undectectable = df[df['det'] == False]
max_det_false = df_undectectable[df_undectectable['correct_rate'] == df_undectectable['correct_rate'].max()].head(1)
max_det_false_sentence = max_det_false['sentence'].values[0].split(".")[4].strip()

import textwrap

axes.axes[0, 0].annotate(
    textwrap.fill("[...] (5rd sentence) "+max_det_false_sentence, width=35,max_lines=3),
    xy=(max_det_false['avg_entropy'].values[0], max_det_false['correct_rate'].values[0]),
    xytext=(max_det_false['avg_entropy'].values[0] - 5.5, max_det_false['correct_rate'].values[0] +0.07),
    arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
    fontsize=10, color=sns.color_palette('Set1')[2], ha='left', va='center',
    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.8),
)
df_tmp = df_undectectable[df_undectectable["temperature"]==1.6]
pp = df_tmp[df_tmp['correct_rate'] == df_tmp['correct_rate'].max()].head(1)
max_det_false_sentence = pp['sentence'].values[0].split(".")[8].strip()
axes.axes[0, 0].annotate(
    textwrap.fill("[...] (9th sentence) "+max_det_false_sentence, width=26,max_lines=5),
    xy=(pp['avg_entropy'].values[0], pp['correct_rate'].values[0]),
    xytext=(pp['avg_entropy'].values[0] - 1.35, pp['correct_rate'].values[0] -0.135),
    arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
    fontsize=10, color=sns.color_palette('Set1')[3], ha='left', va='center',
    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.8),
)


pp = df_undectectable[df_undectectable["temperature"]==1.4].sort_values('correct_rate')[5:6]
max_det_false_sentence = pp['sentence'].values[0].split(".")[29].strip().replace("\n", "")
axes.axes[0, 0].annotate(
    textwrap.fill("[...] (30th sentence) "+max_det_false_sentence, width=30,max_lines=4),
    xy=(pp['avg_entropy'].values[0], pp['correct_rate'].values[0]),
    xytext=(pp['avg_entropy'].values[0]+3 , pp['correct_rate'].values[0] -0.13),
    arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
    fontsize=10, color=sns.color_palette('Set1')[2], ha='left', va='center',
    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.8),
)
pp = df_undectectable[df_undectectable["temperature"]==1.0].sort_values('correct_rate')[5:6]
max_det_false_sentence = pp['sentence'].values[0].split(".")[39].strip().replace("\n", "")
axes.axes[0, 0].annotate(
    textwrap.fill("[...] (40th sentence) "+max_det_false_sentence, width=15,max_lines=4),
    xy=(pp['avg_entropy'].values[0], pp['correct_rate'].values[0]),
    xytext=(pp['avg_entropy'].values[0], pp['correct_rate'].values[0] +0.14),
    arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
    fontsize=10, color=sns.color_palette('Set1')[0], ha='left', va='center',
    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.8),
)

pp = df_undectectable[df_undectectable["temperature"]==1.0].sort_values('correct_rate')[-5:-4]
max_det_false_sentence = pp['sentence'].values[0].split(".")[9].strip().replace("\n", "")
axes.axes[0, 0].annotate(
    textwrap.fill("[...] (10th sentence) "+max_det_false_sentence, width=23,max_lines=3),
    xy=(pp['avg_entropy'].values[0], pp['correct_rate'].values[0]),
    xytext=(pp['avg_entropy'].values[0]+2, pp['correct_rate'].values[0]-0.02),
    arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
    fontsize=10, color=sns.color_palette('Set1')[0], ha='left', va='center',
    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.8),
)
pp= df_undectectable[df_undectectable["temperature"]==1.2].sort_values('correct_rate')[-10:-9]
max_det_false_sentence = pp['sentence'].values[0].split(".")[7].strip().replace("\n", "")
axes.axes[0, 0].annotate(
    textwrap.fill("[...] (8th sentence) "+max_det_false_sentence, width=20,max_lines=3),
    xy=(pp['avg_entropy'].values[0], pp['correct_rate'].values[0]),
    xytext=(pp['avg_entropy'].values[0] -5, pp['correct_rate'].values[0] +0.08),
    arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
    fontsize=10, color=sns.color_palette('Set1')[1], ha='left', va='center',
    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.8),
)
pp= df_undectectable[df_undectectable["temperature"]==1.2].sort_values('correct_rate')[400:401]
max_det_false_sentence = pp['sentence'].values[0].split(".")[12].strip().replace("\n", "")
axes.axes[0, 0].annotate(
    textwrap.fill("[...] (13th sentence) "+max_det_false_sentence + " [...]", width=14,max_lines=6),
    xy=(pp['avg_entropy'].values[0], pp['correct_rate'].values[0]),
    xytext=(pp['avg_entropy'].values[0] - 5.5, pp['correct_rate'].values[0] +0.2),
    arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
    fontsize=10, color=sns.color_palette('Set1')[1], ha='left', va='center',
    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.8),
)
df_dectectable = df[df['det'] == True]
pp = df_dectectable[df_dectectable['avg_entropy'] == df_dectectable['avg_entropy'].min()].head(1)
annot_sentence = pp['sentence'].values[0].split(".")[5].strip().replace("\n", "")
axes.axes[0, 1].annotate(
    textwrap.fill("[...] (6th sentence) "+annot_sentence+" [...]", width=40,max_lines=4),
    xy=(pp['avg_entropy'].values[0], pp['correct_rate'].values[0]),
    xytext=(pp['avg_entropy'].values[0] - 10, pp['correct_rate'].values[0] +0.09),
    arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
    fontsize=10, color=sns.color_palette('Set1')[2], ha='left', va='center',
    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.8),
)
df_tmp = df_dectectable[df_dectectable["temperature"]==1.6]
pp = df_tmp.sort_values('correct_rate')[50:51]
max_det_false_sentence = pp['sentence'].values[0].split(".")[3].strip().replace("\n", "")
axes.axes[0, 1].annotate(
    textwrap.fill("[...] (4th sentence) "+max_det_false_sentence, width=20,max_lines=3),
    xy=(pp['avg_entropy'].values[0], pp['correct_rate'].values[0]),
    xytext=(pp['avg_entropy'].values[0] - 6.5, pp['correct_rate'].values[0] -0.3),
    arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
    fontsize=10, color=sns.color_palette('Set1')[3], ha='left', va='center',
    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.8),
)
df_tmp = df_dectectable[df_dectectable["temperature"]==1.8]
pp = df_tmp.sort_values('correct_rate').head(1)
max_det_false_sentence = pp['sentence'].values[0].split(".")[6].strip().replace("\n", "")
axes.axes[0, 1].annotate(
    textwrap.fill("[...] (7th sentence) "+max_det_false_sentence, width=12,max_lines=4),
    xy=(pp['avg_entropy'].values[0], pp['correct_rate'].values[0]),
    xytext=(pp['avg_entropy'].values[0] - 6.5, pp['correct_rate'].values[0] -0.2),
    arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
    fontsize=10, color=sns.color_palette('Set1')[4], ha='left', va='center',
    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.8),
)
pp=df_tmp.sort_values('correct_rate')[-3:-2]
max_det_false_sentence = pp['sentence'].values[0].split(".")[0].strip().replace("\n", "")
axes.axes[0, 1].annotate(
    textwrap.fill("[...] (1st sentence) "+max_det_false_sentence, width=10,max_lines=8),
    xy=(pp['avg_entropy'].values[0], pp['correct_rate'].values[0]),
    xytext=(pp['avg_entropy'].values[0]-2.5, pp['correct_rate'].values[0] -0.23),
    arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
    fontsize=10, color=sns.color_palette('Set1')[4], ha='left', va='center',
    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.8),
)
print(df["temperature"].value_counts())
plt.savefig('llm/data/avg_entropy_vs_correct_rate.pdf', bbox_inches='tight')


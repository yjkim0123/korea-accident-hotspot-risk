"""Generate a clean pipeline/architecture schematic for the paper."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig,ax=plt.subplots(figsize=(7.2,2.5)); ax.set_xlim(0,12); ax.set_ylim(0,4); ax.axis('off')
def box(x,y,w,h,txt,fc):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.06",
        linewidth=1.1,edgecolor='#333',facecolor=fc))
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=8.2)
def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=11,lw=1.1,color='#555'))

# Data sources
box(0.1,2.5,2.5,1.1,'data.go.kr\nhotspots (3,956)','#dbeafe')
box(0.1,0.4,2.5,1.1,'OSM Overpass\nroad features','#dbeafe')
# Feature engineering
box(3.2,1.4,2.4,1.2,'Feature\nengineering\n(spatial + road)','#fef3c7')
# Models
box(6.2,1.4,2.5,1.2,'Models:\nLogit/XGB/RF/MLP\nkNN/GraphSAGE','#dcfce7')
# Evaluation
box(9.3,1.4,2.5,1.2,'Evaluation:\nhold-out/CV/LOPO\npersistence/SHAP','#fee2e2')

arrow(2.6,3.0,3.2,2.3); arrow(2.6,0.95,3.2,1.8)
arrow(5.6,2.0,6.2,2.0); arrow(8.7,2.0,9.3,2.0)
plt.tight_layout(); plt.savefig('paper/fig_pipeline.png',dpi=160,bbox_inches='tight'); print('OK')

import numpy as np
import plotly
import plotly.graph_objects as go

arr = np.load('cache/minimax.npy')
arr = arr.reshape((9, 3**9))

fig = go.Figure(go.Heatmap(z=arr, x=np.arange(0, 3**9, 1),
                           y=np.arange(0, 9, 1), colorscale='viridis'))
fig.update_layout(title='Minimax Cache visualization',
                  xaxis={'title': 'States'}, yaxis={'title': 'Actions'})

plotly.offline.plot(fig, filename='cache/minimax_visualized.html')

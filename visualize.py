import numpy as np
import plotly
import plotly.graph_objects as go


def visualize_q_table(dirname, high, iterate):
    """
    Visualizes the q_tables saved in the `dirname` directory

    params:

    dirname: Directory name where q_tables are stored

    high: Highest episode number

    iterate: Steps at which q_tables are saved
    """

    # Create figure
    fig = go.Figure()

    # Add traces, one for each slider step
    for step in np.arange(0, high, iterate):
        arr = np.load(dirname+'/q_table_'+str(step)+'.npy').reshape(9, 3**9)
        arr[arr == -100] = -1
        fig.add_trace(
            go.Heatmap(
                visible=False,
                name="episode = " + str(step),
                z=arr,
                x=np.arange(0, 3**9, 1),
                y=np.arange(0, 9, 1),
                colorscale='viridis'))

    # Make 10th trace visible
    fig.data[0].visible = True

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": "Slider switched to step: " + str(i*iterate)}],
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Frequency: "},
        pad={"t": len(np.arange(0, high, iterate))},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        xaxis={'title': 'States'},
        yaxis={'title': 'Actions'}
    )

    plotly.offline.plot(fig, filename=dirname+'/q_visualized.html')
    # fig.show()

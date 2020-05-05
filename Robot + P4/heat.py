import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    # ax.xaxis.set_ticklabel_position('t')
    ax.xaxis.set_label_position('top')


    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    ax.set_ylabel("Delay (ms)", fontsize=10,ha='center')

  #ax.set_xlabel(r'$Acceleration \ (^\circ/s^2 )$', fontsize=10,ha='center')
     
    ax.set_xlabel(r'$Step \ (mm)$', fontsize=10,ha='center')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.
    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

########################

# vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
#               "potato", "wheat", "barley"]

# farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
#            "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

# harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
#                     [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
#                     [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
#                     [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
#                     [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
#                     [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
#                     [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])


#vegetables = ["0","5","10","15","25","50","100","150","200","250","300","350","400","450","500"]
#farmers = ["1","5","10","20","30"]

#harvest = np.array([[0.00,0.00,0.00,0.00,0.00],
#                    [0.00,0.00,0.00,0.00,0.00],
#                    [0.00,0.01,0.00,0.00,0.00],
#                    [0.00,0.01,0.00,0.00,0.00],
#                    [0.01,0.01,0.01,0.01,0.01],
#                    [0.01,0.01,0.01,0.01,0.01],
#                    [0.01,0.01,0.02,0.02,0.02],
#                    [0.01,0.02,0.02,0.03,0.03],
#                    [0.01,0.02,0.02,0.04,0.04],
#                    [0.02,0.03,0.04,0.05,0.05],
#                    [0.02,0.03,0.04,0.05,0.06],
#                    [0.02,0.04,0.05,0.06,0.07],
#                    [0.02,0.04,0.06,0.07,0.08],
#                    [0.03,0.05,0.06,0.08,0.09],
#                    [0.03,0.05,0.07,0.09,0.10]])


vegetables = ["0","1","3","5","7.5","10","12.5","15","20","25","35","50","75","100","150","200","250","300"]
farmers = ["0.1","1"]

#Acc30 Step 0,1;1
#harvest = np.array([[0.0,0.0],
#                    [0.0,0.0],
#                    [0.0,0.0],
#                    [0.0,0.0],
#                    [0.0,0.0],
#                    [0.0,0.0],
#                    [0.0,0.0],
#                    [0.1,1.0],
#                    [0.1,1.0],
#                    [0.1,1.0],
#                    [0.1,1.0],
#                    [0.1,1.0],
#                    [0.1,1.0],
#                    [0.1,1.0],
#                    [0.1,1.0],
#                    [0.2,1.0],
#                    [0.2,1.0],
#                    [0.3,1.0]])

#Acc80 Step 0,1;1
#harvest = np.array([[0.0,0.0],
#                    [0.0,0.0],
#                    [0.0,0.0],
#                    [0.1,0.0],
#                    [0.1,1.0],
#                    [0.1,1.0],
#                    [0.1,1.0],
#                    [0.1,1.0],
#                    [0.1,1.0],
#                    [0.1,1.0],
#                    [0.1,1.0],
#                    [0.1,1.0],
#                    [0.1,1.0],
#                    [0.1,1.0],
#                    [0.2,1.0],
#                    [0.2,1.0],
#                    [0.3,1.0],
#                    [0.3,1.0]])

#Acc130 Step 0,1;1
harvest = np.array([[0.1,1.0],
                    [0.1,1.0],
                    [0.1,1.0],
                    [0.1,1.0],
                    [0.1,1.0],
                    [0.1,1.0],
                    [0.1,1.0],
                    [0.1,1.0],
                    [0.1,1.0],
                    [0.1,1.0],
                    [0.1,1.0],
                    [0.1,1.0],
                    [0.1,1.0],
                    [0.1,1.0],
                    [0.2,1.0],
                    [0.2,1.0],
                    [0.2,1.0],
                    [0.3,1.0]])


# fig, ax = plt.subplots()
# im = ax.imshow(harvest)

# # We want to show all ticks...
# ax.set_xticks(np.arange(len(farmers)))
# ax.set_yticks(np.arange(len(vegetables)))
# # ... and label them with the respective list entries
# ax.set_xticklabels(farmers)
# ax.set_yticklabels(vegetables)

# # Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")

# # Loop over data dimensions and create text annotations.
# for i in range(len(vegetables)):
#     for j in range(len(farmers)):
#         text = ax.text(j, i, harvest[i, j],
#                        ha="center", va="center", color="w")

# ax.set_title("Harvest of local farmers (in tons/year)")
# fig.tight_layout()
# plt.show()

# fig, ax = plt.subplots()
fig = plt.figure(figsize=(4,6))
ax = fig.add_subplot(111)


im, cbar = heatmap(harvest, vegetables, farmers, ax=ax,
                   cmap="YlGn", cbarlabel="Error (mm)")
texts = annotate_heatmap(im, valfmt="{x:.2f}",size=7)

fig.tight_layout()
# plt.show()
plt.savefig("heat.png")
plt.savefig("heat.eps")

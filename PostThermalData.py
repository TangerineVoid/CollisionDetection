import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colorbar as cbar
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mplcursors

# ========================
# FILTERING FUNCTIONS
# ========================
def filter_maxValues(file_path):
    df = pd.read_csv(file_path, header=None)
    # ------------------------
    ## Sweep and find the highest temperatures (deposition center)
    # xmax_values = df.max(axis=1)
    # xmax_indices = df.idxmax(axis=1)
    # ymax_values = df.max(axis=0)
    # ymax_indices = df.idxmax(axis=0)
    # Change all values to 0, except the maximum values
    # for row in range(np.size(df,0)):
    #     df.loc[row, :] = 0  # Set all values in the row to 0
    #     max_index = xmax_indices[row]
    #     df.loc[row, max_index] = xmax_values[row]  # Keep the maximum value
    # for column in range(np.size(df,1)):
    #     max_index = ymax_indices[column]
    #     df.loc[max_index,column] = ymax_values[column]  # Keep the maximum value
    # ------------------------
def filter_threshold(file_path,thr = 26, plot=False, c=0):
    print(f'Processing: {os.path.basename(file_path)}.csv')
    df = pd.read_csv(file_path, header=None)
    # Define temperature threshold
    threshold = 26  # Adjust this value as needed
    mask = df > threshold
    thresholded_values = df[mask]
    df = thresholded_values
    # n = 3
    # thrindex = thresholded_values.apply(lambda row: row.nlargest(n).index(n), axis=1)
    # # top_indexes = df.apply(lambda row: row.nlargest(n).index, axis=1)
    # mask = thresholded_values < thr
    # thresholded_values = df[mask]
    # thresholded_indexes = np.argwhere(mask.to_numpy())
    # Set second mask
    # mask2 = np.full_like(df,True)
    # mask2[15+c:,:] = False
    # mask = mask & mask2
    arr = np.where(mask, df, 0)
    df = pd.DataFrame(arr)

    # Create a boolean DataFrame where each entry is True if the corresponding value is different than zero
    boolean_df = df != 0

    # Sum the boolean DataFrame along the columns (axis=0) or rows (axis=1)
    # # Sum along columns
    # count_non_zero_per_column = boolean_df.sum(axis=0)
    #
    # # Sum along rows
    # count_non_zero_per_row = boolean_df.sum(axis=1)

    ## Total count of non-zero values in the entire DataFrame
    # total_non_zero_count = boolean_df.sum().sum()
    # print(total_non_zero_count)

    # mask = df < threshold
    # Measuring coldest temperature height, from top to bottom
    # sarr = []
    # x = []
    # y = []
    # for v,values in enumerate(arr):
    #     if len(values[values > 0]) > 0:
    #         im = np.where((values > 0))
    #         i = np.argmin(values[values > 0])
    #         x.append(im[0][i])
    #         y.append(v)
    #         sarr.append(values[im[0][i]])
    # df = pd.DataFrame([np.argmin(sarr),np.argmin(x),np.argmin(y)])
    # df.to_csv(f'{os.path.basename(file_path)}.csv', index=False)
    # print(min([min(values[values>0]) for values in arr]))

    if plot:
        # ========================
        # OUTPUT
        # ========================
        # ------------------------
        # Save the modified DataFrame to a new CSV file
        # df.to_csv('modified_matrix.csv', header=False, index=False)
        # ------------------------
        # ------------------------
        # Normalize the DataFrame to a range of 0-255 for grayscale representation
        df_normalized = (df - df.min().min()) / (df.max().max() - df.min().min()) * 255
        # Convert the DataFrame to a numpy array
        array = df_normalized.to_numpy().astype(np.uint8)
        # Invert the colors by subtracting from 255
        inverted_array = 255 - array
        # Create a PIL Image from the numpy array
        image = Image.fromarray(inverted_array, 'L')

        # Convert to a color map using matplotlib
        norm = Normalize(vmin=0, vmax=180) #vmax=max(max_values)
        color_mapped_array = plt.cm.rainbow(norm(array))  # You can choose other colormaps like 'plasma', 'inferno', 'magma', etc.
        color_mapped_array = (color_mapped_array[:, :, :3] * 255).astype(np.uint8)  # Remove alpha channel and convert to 8-bit
        # Create a PIL Image from the color-mapped array
        color_image = Image.fromarray(color_mapped_array, 'RGB')

        # Create the figure and axis
        fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(14,5))
        # fig, ax = plt.subplots(figsize=(10, 8))
        fig.subplots_adjust(right=0.85)
        ax[0].imshow(color_image, aspect="auto")
        label = os.path.basename(file_path).split('.')[0]
        ax[0].set_xlabel(f'{label}')
        # ax[0].imshow(color_image)

        # Add a colorbar
        divider = make_axes_locatable(ax[0])
        # cbar_ax = fig.add_axes([0.5, 0.15, 0.03, 0.7])
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(sm, cax=cax)

        # Find the top 5 values and their indexes in each row
        n = 500
        top_values = df.apply(lambda row: row.nlargest(n), axis=1)
        top_indexes = df.apply(lambda row: row.nlargest(n).index, axis=1)
        top_values_array = np.zeros_like(df)
        for i in range(df.shape[0]):
            top_values_array[i, top_indexes.iloc[i]] = top_values.loc[i,top_indexes.iloc[i]]
        df = top_values_array
        # top_values_array = top_values_array.astype(np.uint8)

        # Plot non-zero temperatures next to the image
        # ax[1] = fig.add_axes([0.92, 0.15, 0.1, 0.7])
        maxv = max([max(values) for values in df])
        ax[1].set_xlim(0, maxv + maxv/10)
        ax[1].set_ylim(ax[0].get_ylim())
        # scatter_points = []
        # ax[1].set_ylim(0, max([len(values) for values in top_values_array]))
        for i, values in enumerate(df):
            if len(values) > 0:
                scatter = ax[1].scatter(values[values>0], np.full_like(values[values>0], i),marker='.',color='k',s=0.5)
                # scatter_points.append(scatter)
                # ax[1].plot(values[values>0], np.full_like(values[values>0], i), 'k.')
        ax[1].set_xlabel("Temperature")
        ax[1].axvline(x=50,color='r')
        ax[1].axvline(x=60,color='r')
        # plt.autoscale(True,axis='both')
        # Save the final image with colorbar
        # fig.savefig('colored_matrix_image_with_colorbar.png', bbox_inches='tight')
        # Make the plot interactive with mplcursors
        # mplcursors.cursor(scatter_points, hover=True).connect(
        #     "add", lambda sel: sel.annotation.set_text(f"Value: {sel.target[0]:.2f}, Row: {int(sel.target[1])}"))
        filename = os.path.basename(file_path).split('.')[0]
        plt.savefig(f'{filename}.png')
        # plt.show()
        plt.close(fig)
        # Save the image
        # image.save('matrix_image.png')

        # Show the image
        # image.show()
def process_data(folder_path,filename):
    file_path = folder_path + filename
    if filename == '':
        counter = 0
        for file_path in glob.glob(os.path.join(folder_path, '*.csv')):
            filter_threshold(file_path,50,plot=True,c=counter)
            counter+=9
    else: filter_threshold(file_path,46)

if __name__ == "__main__":
    path = 'D:/Doctorado/Thermal Camera/New folder'
    # path = 'D:/Doctorado/Thermal Camera/C Shape/frame by frame csv'
    # path = 'D:/Doctorado/Thermal Camera/C Shape/'
    filename = ''
    # filename = '3-C_shape_185C_0035_nfix_02.csv'
    process_data(path,filename)

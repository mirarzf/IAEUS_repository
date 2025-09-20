import matplotlib.pyplot as plt
import numpy as np 

def plot_img_and_mask(img, mask, classes):
    nb_class_columns = 1 if classes <= 2 else classes - 1 
    # We display the RGB image and each predicted class
    # And ignore background class if classes > 1 
    fig, ax = plt.subplots(1, nb_class_columns + 1) 
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[0].set_xticks([]), ax[0].set_yticks([])
    if nb_class_columns > 1:
        ax[1].set_title('All classes in one')
        ax[1].imshow(mask, cmap='gray', vmin=0, vmax=nb_class_columns)
        ax[1].set_xticks([]), ax[1].set_yticks([])
        for i in range(nb_class_columns): # We ignore background class     
            ax[i + 2].set_title(f'Output mask (class {i + 1})')
            ax[i + 2].imshow(np.where(mask == i+1, 255, 0), cmap='gray', vmin=0, vmax=255)
            ax[i + 2].set_xticks([]), ax[i + 2].set_yticks([])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
        ax[1].set_xticks([]), ax[1].set_yticks([])
    plt.show()


def plot_img_and_mask_and_gt(img, gt, mask, classes):
    nb_class_columns = 1 if classes == 1 else classes - 1 
    # We display the RGB image and each predicted class
    # And ignore background class if classes > 1 
    fig, ax = plt.subplots(1, nb_class_columns + 2) 
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[0].set_xticks([]), ax[0].set_yticks([])
    ax[1].set_title('Ground Truth')
    ax[1].imshow(gt, cmap='gray', vmin=0, vmax=1)
    ax[1].set_xticks([]), ax[1].set_yticks([])
    if classes > 1:
        for i in range(nb_class_columns): # We ignore background class     
            ax[i + 2].set_title(f'Output mask (class {i + 1})')
            ax[i + 2].imshow(np.where(mask == i+1, 255, 0), cmap='gray', vmin=0, vmax=255)
            ax[i + 2].set_xticks([]), ax[i + 2].set_yticks([])
    else:
        ax[2].set_title(f'Output mask')
        ax[2].imshow(mask, cmap='gray', vmin=0, vmax=1)
        ax[2].set_xticks([]), ax[2].set_yticks([])
    plt.show()
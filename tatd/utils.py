
import os
import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

plt.switch_backend('agg')
cmap = sns.diverging_palette(220, 10, as_cmap=True)


def get_path(name, weightf, sparse, rrank, window, penalty, scheme, lr , count, exp):
    
    path = f'out/{scheme}/{name}/training/'
    path1= f'out/{scheme}/{name}/model/'
    path2= f'out/{scheme}/{name}/loss/'
    path3= f'out/{scheme}/{name}/factors/'

    for p in [path, path1, path2, path3]:
        if not os.path.isdir(p):
            os.makedirs(p)
    
    info = f'{weightf}_{sparse}_r_{rrank}_w_{window}_p_{penalty}_lr_{lr}_{count}_{exp}'
    
    train_path = os.path.join(path, info + '.txt')
    model_path = os.path.join(path1, info)
    loss_path = os.path.join(path2, info + '.txt')
    f_path = os.path.join(path3, info)
        
    best_path = f'out/{scheme}/{name}/best.txt'
    
    if not os.path.exists(best_path):
        with open(best_path, 'w') as f:
            f.write('No.\titers\ttime\tweightf\tsparse\trank\twindow\tpenalty\tscheme\tlr\trmse\tmae\texp\n')
        f.close()

    return train_path, model_path, loss_path, f_path, best_path


def save_checkpoints(model, path):
    """
    Save a trained model.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(dict(model_state=model.state_dict()), path)


def draw_factors(nmode, tmode, factors, num, path):
    path_img = os.path.join(path, 'img')
    path_p = os.path.join(path, 'p')

    factors = [f.detach().cpu().numpy() for f in factors]

    fig = plt.figure(figsize = (10, 6), dpi = 100)

    sns.heatmap(factors[tmode], cmap = cmap)

    if not os.path.exists(path_img):
        os.makedirs(path_img)
    if not os.path.exists(path_p):
        os.makedirs(path_p)

    plt.savefig(f"{path_img}/{num}.png", bbox_indches = 'tight')
    plt.close(fig)

    np.savetxt(f"{path_p}/{num}.txt", factors[tmode], delimiter='\t', )


def make_images(path):

    image_folder = os.fsencode(path)

    filenames = []

    for file in os.listdir(image_folder):
        filename = os.fsdecode(file)
        if filename.endswith( ('.pdf', '.png') ):
            filenames.append(filename)
    filenames.sort() # this iteration technique has no built in order, so sort the frames

    images = list(map(lambda filename: imageio.imread(f'{path}/{filename}'), filenames))

    imageio.mimsave(os.path.join(f'{path}/tfactor.gif'), images, duration = 0.3) # modify duration as needed

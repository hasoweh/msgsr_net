"""Functions used for creating the pseudolabels from CAMs
"""

from skimage.segmentation import slic
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# get the cam for a single level in the model
def gradcampp(model, target_layer, img, target_class):
    """Obtains a GradCAM++.
    
    Parameters
    ----------
    model: A pre-trained pytorch model.
    target_layer: The specific layer of the model.
    img: torch.tensor, the input image for which to generate the CAM.
    target_class: int, the index of the class which we want to generate a CAM for.
    """
    target_layers = [target_layer] 
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=False)
    targets = [ClassifierOutputTarget(target_class)]
    # get CAM for the input image and the target class
    cam_out = cam(img, targets)

    return cam_out

def superpixel_cam(cam, segments):
    """Combines CAM with superpixel segmentation.
    
    Parameters
    ----------
    cam: np.array(shape = (H, W)), a CAM.
    segments: np.array(shape = (H, W)), the segmented mask.
    """
    # so get the value of each superpixel label
    labels = np.unique(segments)
    updated_cam = np.zeros(segments.shape)
    for label in labels:
        mask = np.where(segments == label, 1, 0)
        # get the average value of the CAM in the segment
        mean_val = mean_nonzero(cam, mask)
        updated_cam[np.where(segments == label)] = mean_val
        
    return updated_cam

def mean_nonzero(array, mask):
    return np.mean(array[np.where(mask != 0)])
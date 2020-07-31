# Differentiable color conversions from https://github.com/TheZino/pytorch-color-conversions

from warnings import warn

import numpy as np
import torch
from scipy import linalg

xyz_from_rgb = torch.Tensor([[0.412453, 0.357580, 0.180423],
                         [0.212671, 0.715160, 0.072169],
                         [0.019334, 0.119193, 0.950227]])

rgb_from_xyz = torch.Tensor(linalg.inv(xyz_from_rgb))

illuminants = \
    {"A": {'2': torch.Tensor([(1.098466069456375, 1, 0.3558228003436005)]),
           '10': torch.Tensor([(1.111420406956693, 1, 0.3519978321919493)])},
     "D50": {'2': torch.Tensor([(0.9642119944211994, 1, 0.8251882845188288)]),
             '10': torch.Tensor([(0.9672062750333777, 1, 0.8142801513128616)])},
     "D55": {'2': torch.Tensor([(0.956797052643698, 1, 0.9214805860173273)]),
             '10': torch.Tensor([(0.9579665682254781, 1, 0.9092525159847462)])},
     "D65": {'2': torch.Tensor([(0.95047, 1., 1.08883)]),   # This was: `lab_ref_white`
             '10': torch.Tensor([(0.94809667673716, 1, 1.0730513595166162)])},
     "D75": {'2': torch.Tensor([(0.9497220898840717, 1, 1.226393520724154)]),
             '10': torch.Tensor([(0.9441713925645873, 1, 1.2064272211720228)])},
     "E": {'2': torch.Tensor([(1.0, 1.0, 1.0)]),
           '10': torch.Tensor([(1.0, 1.0, 1.0)])}}


# -------------------------------------------------------------
# The conversion functions that make use of the matrices above
# -------------------------------------------------------------


##### RGB - YCbCr

# Helper for the creation of module-global constant tensors
def _t(data):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # TODO inherit this
    device = torch.device("cpu") # TODO inherit this
    return torch.tensor(data, requires_grad=False, dtype=torch.float32, device=device)

# Helper for color matrix multiplication
def _mul(coeffs, image):
    # This is implementation is clearly suboptimal.  The function will
    # be implemented with 'einsum' when a bug in pytorch 0.4.0 will be
    # fixed (Einsum modifies variables in-place #7763).
    coeffs = coeffs.to(image.device)
    r0 = image[:, 0:1, :, :].repeat(1, 3, 1, 1) * coeffs[:, 0].view(1, 3, 1, 1)
    r1 = image[:, 1:2, :, :].repeat(1, 3, 1, 1) * coeffs[:, 1].view(1, 3, 1, 1)
    r2 = image[:, 2:3, :, :].repeat(1, 3, 1, 1) * coeffs[:, 2].view(1, 3, 1, 1)
    return r0 + r1 + r2
    # return torch.einsum("dc,bcij->bdij", (coeffs.to(image.device), image))

_RGB_TO_YCBCR = _t([[0.257, 0.504, 0.098], [-0.148, -0.291, 0.439], [0.439 , -0.368, -0.071]])
_YCBCR_OFF = _t([0.063, 0.502, 0.502]).view(1, 3, 1, 1)


def rgb2ycbcr(rgb):
    """sRGB to YCbCr conversion."""
    clip_rgb=False
    if clip_rgb:
        rgb = torch.clamp(rgb, 0, 1)
    return _mul(_RGB_TO_YCBCR, rgb) + _YCBCR_OFF.to(rgb.device)


def ycbcr2rgb(rgb):
    """YCbCr to sRGB conversion."""
    clip_rgb=False
    rgb = _mul(torch.inverse(_RGB_TO_YCBCR), rgb - _YCBCR_OFF.to(rgb.device))
    if clip_rgb:
        rgb = torch.clamp(rgb, 0, 1)
    return rgb


##### HSV - RGB

def rgb2hsv(rgb):
    """
    R, G and B input range = 0 ÷ 1.0
    H, S and V output range = 0 ÷ 1.0
    """
    eps = 1e-7

    var_R = rgb[:,0,:,:]
    var_G = rgb[:,1,:,:]
    var_B = rgb[:,2,:,:]

    var_Min = rgb.min(1)[0]    #Min. value of RGB
    var_Max = rgb.max(1)[0]    #Max. value of RGB
    del_Max = var_Max - var_Min             ##Delta RGB value

    H = torch.zeros([rgb.shape[0], rgb.shape[2], rgb.shape[3]]).to(rgb.device)
    S = torch.zeros([rgb.shape[0], rgb.shape[2], rgb.shape[3]]).to(rgb.device)
    V = torch.zeros([rgb.shape[0], rgb.shape[2], rgb.shape[3]]).to(rgb.device)

    V = var_Max

    #This is a gray, no chroma...
    mask = del_Max == 0
    H[mask] = 0
    S[mask] = 0

    #Chromatic data...
    S = del_Max / (var_Max + eps)

    del_R = ( ( ( var_Max - var_R ) / 6 ) + ( del_Max / 2 ) ) / (del_Max + eps)
    del_G = ( ( ( var_Max - var_G ) / 6 ) + ( del_Max / 2 ) ) / (del_Max + eps)
    del_B = ( ( ( var_Max - var_B ) / 6 ) + ( del_Max / 2 ) ) / (del_Max + eps)

    H = torch.where( var_R == var_Max , del_B - del_G, H)
    H = torch.where( var_G == var_Max , ( 1 / 3 ) + del_R - del_B, H)
    H = torch.where( var_B == var_Max ,( 2 / 3 ) + del_G - del_R, H)

    # if ( H < 0 ) H += 1
    # if ( H > 1 ) H -= 1

    return torch.stack([H, S, V], 1)

def hsv2rgb(hsv):
    """
    H, S and V input range = 0 ÷ 1.0
    R, G and B output range = 0 ÷ 1.0
    """

    eps = 1e-7

    bb,cc,hh,ww = hsv.shape
    H = hsv[:,0,:,:]
    S = hsv[:,1,:,:]
    V = hsv[:,2,:,:]

    # var_h = torch.zeros(bb,hh,ww)
    # var_s = torch.zeros(bb,hh,ww)
    # var_v = torch.zeros(bb,hh,ww)

    # var_r = torch.zeros(bb,hh,ww)
    # var_g = torch.zeros(bb,hh,ww)
    # var_b = torch.zeros(bb,hh,ww)

    # Grayscale
    if (S == 0).all():

        R = V
        G = V
        B = V

    # Chromatic data
    else:

        var_h = H * 6

        var_h[var_h == 6] = 0      #H must be < 1
        var_i = var_h.floor()                           #Or ... var_i = floor( var_h )
        var_1 = V * ( 1 - S )
        var_2 = V * ( 1 - S * ( var_h - var_i ) )
        var_3 = V * ( 1 - S * ( 1 - ( var_h - var_i ) ) )

        # else                   { var_r = V     ; var_g = var_1 ; var_b = var_2 }
        var_r = V
        var_g = var_1
        var_b = var_2

        # var_i == 0 { var_r = V     ; var_g = var_3 ; var_b = var_1 }
        var_r = torch.where(var_i == 0, V, var_r)
        var_g = torch.where(var_i == 0, var_3, var_g)
        var_b = torch.where(var_i == 0, var_1, var_b)

        # else if ( var_i == 1 ) { var_r = var_2 ; var_g = V     ; var_b = var_1 }
        var_r = torch.where(var_i == 1, var_2, var_r)
        var_g = torch.where(var_i == 1, V, var_g)
        var_b = torch.where(var_i == 1, var_1, var_b)

        # else if ( var_i == 2 ) { var_r = var_1 ; var_g = V     ; var_b = var_3 }
        var_r = torch.where(var_i == 2, var_1, var_r)
        var_g = torch.where(var_i == 2, V, var_g)
        var_b = torch.where(var_i == 2, var_3, var_b)

        # else if ( var_i == 3 ) { var_r = var_1 ; var_g = var_2 ; var_b = V     }
        var_r = torch.where(var_i == 3, var_1, var_r)
        var_g = torch.where(var_i == 3, var_2, var_g)
        var_b = torch.where(var_i == 3, V, var_b)

        # else if ( var_i == 4 ) { var_r = var_3 ; var_g = var_1 ; var_b = V     }
        var_r = torch.where(var_i == 4, var_3, var_r)
        var_g = torch.where(var_i == 4, var_1, var_g)
        var_b = torch.where(var_i == 4, V, var_b)


        R = var_r #* 255
        G = var_g #* 255
        B = var_b #* 255


    return torch.stack([R, G, B], 1)


##### LAB - RGB

def _convert(matrix, arr):
    """Do the color space conversion.
    Parameters
    ----------
    matrix : array_like
        The 3x3 matrix to use.
    arr : array_like
        The input array.
    Returns
    -------
    out : ndarray, dtype=float
        The converted array.
    """

    if arr.is_cuda:
        matrix = matrix.cuda()

    bs, ch, h, w = arr.shape

    arr = arr.permute((0,2,3,1))
    arr = arr.contiguous().view(-1,1,3)

    matrix = matrix.transpose(0,1).unsqueeze(0)
    matrix = matrix.repeat(arr.shape[0],1,1)

    res = torch.bmm(arr,matrix)

    res = res.view(bs,h,w,ch)
    res = res.transpose(3,2).transpose(2,1)


    return res

def get_xyz_coords(illuminant, observer):
    """Get the XYZ coordinates of the given illuminant and observer [1]_.
    Parameters
    ----------
    illuminant : {"A", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10"}, optional
        The aperture angle of the observer.
    Returns
    -------
    (x, y, z) : tuple
        A tuple with 3 elements containing the XYZ coordinates of the given
        illuminant.
    Raises
    ------
    ValueError
        If either the illuminant or the observer angle are not supported or
        unknown.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Standard_illuminant
    """
    illuminant = illuminant.upper()
    try:
        return illuminants[illuminant][observer]
    except KeyError:
        raise ValueError("Unknown illuminant/observer combination\
        (\'{0}\', \'{1}\')".format(illuminant, observer))





def rgb2xyz(rgb):

    mask = rgb > 0.04045
    rgbm = rgb.clone()
    tmp = torch.pow((rgb + 0.055) / 1.055, 2.4)
    rgb = torch.where(mask, tmp, rgb)

    rgbm = rgb.clone()
    rgb[~mask] = rgbm[~mask]/12.92
    return _convert(xyz_from_rgb, rgb)

def xyz2lab(xyz, illuminant="D65", observer="2"):

    # arr = _prepare_colorarray(xyz)
    xyz_ref_white = get_xyz_coords(illuminant, observer)
    #cuda
    if xyz.is_cuda:
        xyz_ref_white = xyz_ref_white.cuda()

    # scale by CIE XYZ tristimulus values of the reference white point
    xyz = xyz / xyz_ref_white.view(1,3,1,1)
    # Nonlinear distortion and linear transformation
    mask = xyz > 0.008856
    xyzm = xyz.clone()
    xyz[mask] = torch.pow(xyzm[mask], 1/3)
    xyzm = xyz.clone()
    xyz[~mask] = 7.787 * xyzm[~mask] + 16. / 116.
    x, y, z = xyz[:, 0, :, :], xyz[:, 1, :, :], xyz[:, 2, :, :]
    # Vector scaling
    L = (116. * y) - 16.
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)
    return torch.stack((L,a,b), 1)

def rgb2lab(rgb, illuminant="D65", observer="2"):
    """RGB to lab color space conversion.
    Parameters
    ----------
    rgb : array_like
        The image in RGB format, in a 3- or 4-D array of shape
        ``(.., ..,[ ..,] 3)``.
    illuminant : {"A", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10"}, optional
        The aperture angle of the observer.
    Returns
    -------
    out : ndarray
        The image in Lab format, in a 3- or 4-D array of shape
        ``(.., ..,[ ..,] 3)``.
    Raises
    ------
    ValueError
        If `rgb` is not a 3- or 4-D array of shape ``(.., ..,[ ..,] 3)``.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Standard_illuminant
    Notes
    -----
    This function uses rgb2xyz and xyz2lab.
    By default Observer= 2A, Illuminant= D65. CIE XYZ tristimulus values
    x_ref=95.047, y_ref=100., z_ref=108.883. See function `get_xyz_coords` for
    a list of supported illuminants.
    """
    return xyz2lab(rgb2xyz(rgb), illuminant, observer)





def lab2xyz(lab, illuminant="D65", observer="2"):
    arr = lab.clone()
    L, a, b = arr[:, 0, :, :], arr[:, 1, :, :], arr[:, 2, :, :]
    y = (L + 16.) / 116.
    x = (a / 500.) + y
    z = y - (b / 200.)

    # if (z < 0).sum() > 0:
    #     warn('Color data out of range: Z < 0 in %s pixels' % (z < 0).sum().item())
    #     z[z < 0] = 0 # NO GRADIENT!!!!

    out = torch.stack((x, y, z),1)

    mask = out > 0.2068966
    outm = out.clone()
    out[mask] = torch.pow(outm[mask], 3.)
    outm = out.clone()
    out[~mask] = (outm[~mask] - 16.0 / 116.) / 7.787

    # rescale to the reference white (illuminant)
    xyz_ref_white = get_xyz_coords(illuminant, observer)
    # cuda
    if lab.is_cuda:
        xyz_ref_white = xyz_ref_white.cuda()
    xyz_ref_white = xyz_ref_white.unsqueeze(2).unsqueeze(2).repeat(1,1,out.shape[2],out.shape[3])
    out = out * xyz_ref_white
    return out

def xyz2rgb(xyz):
    arr = _convert(rgb_from_xyz, xyz)
    mask = arr > 0.0031308
    arrm = arr.clone()
    arr[mask] = 1.055 * torch.pow(arrm[mask], 1 / 2.4) - 0.055
    arrm = arr.clone()
    arr[~mask] = arrm[~mask] * 12.92

    # CLAMP KILLS GRADIENTS
    # mask_z = arr < 0
    # arr[mask_z] = 0
    # mask_o = arr > 1
    # arr[mask_o] = 1

    # torch.clamp(arr, 0, 1, out=arr)
    return arr

def lab2rgb(lab, illuminant="D65", observer="2"):
    """Lab to RGB color space conversion.
    Parameters
    ----------
    lab : array_like
        The image in Lab format, in a 3-D array of shape ``(.., .., 3)``.
    illuminant : {"A", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10"}, optional
        The aperture angle of the observer.
    Returns
    -------
    out : ndarray
        The image in RGB format, in a 3-D array of shape ``(.., .., 3)``.
    Raises
    ------
    ValueError
        If `lab` is not a 3-D array of shape ``(.., .., 3)``.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Standard_illuminant
    Notes
    -----
    This function uses lab2xyz and xyz2rgb.
    By default Observer= 2A, Illuminant= D65. CIE XYZ tristimulus values
    x_ref=95.047, y_ref=100., z_ref=108.883. See function `get_xyz_coords` for
    a list of supported illuminants.
    """
    return xyz2rgb(lab2xyz(lab, illuminant, observer))
